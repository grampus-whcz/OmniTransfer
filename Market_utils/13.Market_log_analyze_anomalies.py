import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import argparse

def infer_bucket_sec(timestamps):
    """Infer bucket_sec from sorted timestamps (assume uniform)."""
    if len(timestamps) < 2:
        return 60  # fallback
    diffs = np.diff(timestamps)
    bucket_sec = int(np.min(diffs))  # smallest gap is likely the bucket size
    return bucket_sec

def analyze_log_anomalies_in_time_range(
    log_meta_path: str,
    log_patterns_json_path: str,
    predictions_dir: str,
    pred_time_window: tuple,      # (pred_start_unix, pred_end_unix) — full raw time range used for inference
    query_time_range: tuple,      # (query_start_unix, query_end_unix)
    output_npy_path: str,
    anomaly_report: str,
    window_size: int,             # e.g., 12 — number of historical points used by model
    args
):
    # --- Step 1: Load log metadata ---
    meta = np.load(log_meta_path, allow_pickle=True).item()
    pods = meta["pods"]
    all_global_timestamps = np.array(meta["timestamps"])  # full global timeline (e.g., 1440 points for a day)
    max_attributes = meta["max_attributes"]
    service_lang_map = meta.get("service_language_map", {})

    # Infer bucket_sec from global timestamps
    bucket_sec = infer_bucket_sec(all_global_timestamps)
    print(f"ℹ️ Inferred log bucket_sec = {bucket_sec} seconds")

    pred_start, pred_end = pred_time_window
    query_start, query_end = query_time_range

    # Validate that query is within pred window
    if not (pred_start <= query_start <= query_end <= pred_end):
        raise ValueError("Query range must be within the prediction window.")

    # --- Step 2: Compute slice indices in global timeline ---
    # Align to bucket boundaries
    global_start_sec = all_global_timestamps[0]
    total_buckets = len(all_global_timestamps)

    # Compute start/end bucket indices for the PREDICTION INPUT SLICE
    i_start = (pred_start - global_start_sec) // bucket_sec
    i_end_excl = (pred_end - global_start_sec) // bucket_sec + 1

    # Clamp to valid range
    i_start = max(0, i_start)
    i_end_excl = min(i_end_excl, total_buckets)
    L_raw = i_end_excl - i_start

    if L_raw <= 0:
        raise ValueError(f"No valid buckets found in prediction window [{pred_start}, {pred_end}]")

    # Extract the actual raw timestamps used as input to the model
    raw_input_timestamps = all_global_timestamps[i_start:i_end_excl]  # shape: (L_raw,)

    # Verify alignment (optional)
    expected_first = global_start_sec + i_start * bucket_sec
    expected_last = global_start_sec + (i_end_excl - 1) * bucket_sec
    if raw_input_timestamps[0] != expected_first or raw_input_timestamps[-1] != expected_last:
        print("⚠️ Warning: Raw input timestamps may not align perfectly with bucket grid. Proceeding.")

    # --- Step 3: Compute prediction timestamps ---
    if L_raw <= window_size:
        raise ValueError(f"Raw input length ({L_raw}) <= window_size ({window_size}) — no predictions possible.")

    # Predictions correspond to timestamps starting at index = i_start + window_size
    pred_bucket_indices = np.arange(i_start + window_size, i_start + L_raw)
    pred_timestamps = global_start_sec + pred_bucket_indices * bucket_sec  # shape: (L_pred,)
    expected_pred_len = len(pred_timestamps)

    print(f"ℹ️ Input slice: {L_raw} buckets → Predictions: {expected_pred_len} points "
          f"(window_size={window_size}, bucket_sec={bucket_sec})")

    # --- Step 4: Load patterns ---
    with open(log_patterns_json_path, 'r') as f:
        patterns_dict = json.load(f)
    java_patterns = patterns_dict.get("java") or patterns_dict.get("Java", [])
    if len(java_patterns) != max_attributes:
        print(f"⚠️ Warning: Expected {max_attributes} Java patterns, got {len(java_patterns)}. Padding/truncating.")
        if len(java_patterns) < max_attributes:
            java_patterns += [f"<UNKNOWN_PATTERN_{i}>" for i in range(len(java_patterns), max_attributes)]
        else:
            java_patterns = java_patterns[:max_attributes]

    anomalies = []

    # --- Step 5: Process each pod ---
    for idx, pod in enumerate(pods):
        pred_file = os.path.join(predictions_dir, f"pred_per_feature_{idx}.npy")
        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found. Skipping pod {pod}.")
            continue

        pred = np.load(pred_file)  # shape (L_pred_actual, num_features)
        L_pred_actual, num_features = pred.shape

        # Use actual prediction length, but align with our computed pred_timestamps
        use_len = min(L_pred_actual, expected_pred_len)
        if use_len == 0:
            print(f"Warning: No overlapping predictions for pod {pod}. Skipping.")
            continue

        use_pred = pred[:use_len]
        use_timestamps = pred_timestamps[:use_len]

        if L_pred_actual != expected_pred_len:
            print(f"Warning: Pod {pod}: prediction length {L_pred_actual} ≠ expected {expected_pred_len}. "
                  f"Using first {use_len} points.")

        # Apply user query time filter
        mask_query = (use_timestamps >= query_start) & (use_timestamps <= query_end)

        actual_features = min(num_features, max_attributes)
        for feat_idx in range(actual_features):
            anomaly_mask = (use_pred[:, feat_idx] == 1) & mask_query
            for t_idx in np.where(anomaly_mask)[0]:
                ts = int(use_timestamps[t_idx])
                anomalies.append({
                    "pod": pod,
                    "pattern_id": feat_idx,
                    "pattern_str": java_patterns[feat_idx],
                    "timestamp": ts
                })

    # --- Step 6: Save .npy ---
    dt = np.dtype([
        ('pod', 'U32'),
        ('pattern_id', 'i4'),
        ('pattern_str', 'U512'),
        ('timestamp', 'i8')
    ])
    if anomalies:
        arr = np.array([
            (a['pod'], a['pattern_id'], a['pattern_str'][:511], a['timestamp']) for a in anomalies
        ], dtype=dt)
    else:
        arr = np.empty(0, dtype=dt)

    np.save(output_npy_path, arr)
    print(f"\n✅ Saved {len(anomalies)} log anomaly records to {output_npy_path}")

    # --- Step 7: Generate human-readable report ---
    if not anomalies:
        print("\n🔍 No log anomalies found in the specified time range.")
        return

    report = defaultdict(lambda: defaultdict(list))
    for a in anomalies:
        report[a['pod']][a['pattern_id']].append(a['timestamp'])

    report_output_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.pred_dir}"
    os.makedirs(report_output_dir, exist_ok=True)
    report_file_path = os.path.join(report_output_dir, f"Market_log_anomaly_report_{anomaly_report}.txt")

    report_lines = []
    report_lines.append("\n📝 Detailed Log Anomaly Report (Beijing Time = UTC+8):")
    report_lines.append("=" * 80)

    for pod in sorted(report.keys()):
        report_lines.append(f"\nPod: {pod}")
        for pid in sorted(report[pod].keys()):
            timestamps = sorted(report[pod][pid])
            readable_entries = []
            for ts in timestamps:
                beijing_time = datetime.utcfromtimestamp(ts) + timedelta(hours=8)
                readable = f"{ts} ({beijing_time.strftime('%Y-%m-%d %H:%M:%S')} CST)"
                readable_entries.append(readable)
            pattern_preview = java_patterns[pid][:200].replace('\n', ' ').strip()
            if len(java_patterns[pid]) > 200:
                pattern_preview += " ..."
            report_lines.append(f"  - Pattern ID {pid} ({len(timestamps)} anomalies):")
            report_lines.append(f"      Template: {pattern_preview}")
            report_lines.append("      " + ", ".join(readable_entries))

    report_lines.append("\n💡 Note: 'CST' = China Standard Time (UTC+8).")

    full_report_text = "\n".join(report_lines)
    print(full_report_text)

    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    print(f"\n✅ Full report saved to: {report_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LOG anomalies in a given time range (Beijing Time).")
    parser.add_argument("--log_meta", type=str,
                        default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Market/log/temp_data/raw_data/train_valid/2022_03_20/cloudbed-1/raw_log/service_log_patterns_count_meta.npy")
    parser.add_argument("--log_patterns", type=str,
                        default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Market/log/analysis/log_istio_patterns.json")
    parser.add_argument("--pred_dir", type=str, default="1116")
    parser.add_argument("--pred_start", type=int, required=True)
    parser.add_argument("--pred_end", type=int, required=True)
    parser.add_argument("--query_start", type=int, required=True)
    parser.add_argument("--query_end", type=int, required=True)
    parser.add_argument("--output", type=str, default="log_anomalies_in_range.npy")
    parser.add_argument("--anomaly_report", type=str, default="custom_range")
    parser.add_argument("--window_size", type=int, default=12,
                        help="Sliding window size used by the anomaly detection model (default: 12)")

    args = parser.parse_args()

    predictions_dir = (
        f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.pred_dir}/"
        f"TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/"
        f"evaluation_result/predictions_g"
    )

    analyze_log_anomalies_in_time_range(
        log_meta_path=args.log_meta,
        log_patterns_json_path=args.log_patterns,
        predictions_dir=predictions_dir,
        pred_time_window=(args.pred_start, args.pred_end),
        query_time_range=(args.query_start, args.query_end),
        output_npy_path=args.output,
        anomaly_report=args.anomaly_report,
        window_size=args.window_size,
        args=args
    )
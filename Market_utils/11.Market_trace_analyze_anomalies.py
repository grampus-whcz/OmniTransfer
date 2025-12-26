import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from collections import defaultdict

def analyze_market_trace_anomalies_in_time_range(
    cloudbed: str,
    date_str: str,
    predictions_dir: str,
    time_range_query: tuple,
    output_npy_path: str,
    anomaly_report_suffix: str,
    bucket_sec: int,
    window_size: int,          # <-- NEW: sliding window size used by model
    pred_dir_for_report: str   # <-- needed for report path (since args not global)
):
    # --- Step 1: Load trace metadata ---
    meta_path = Path("/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Market/trace") / f"{cloudbed}_{date_str.replace('-', '_')}_trace_edge_bucket_{bucket_sec}.meta.json"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    edges = meta["edges"]
    columns = meta["features"]
    global_start_sec = meta["global_start_sec"]
    bucket_sec_meta = meta["bucket_sec"]
    num_buckets_total = meta["num_buckets"]

    assert bucket_sec_meta == bucket_sec, f"Bucket mismatch: expected {bucket_sec}, got {bucket_sec_meta}"

    query_start, query_end = time_range_query

    if query_start < global_start_sec:
        raise ValueError(f"Query start ({query_start}) is before data start ({global_start_sec})")
    
    # Compute the RAW slice indices (before model windowing)
    i_start = (query_start - global_start_sec) // bucket_sec
    i_end_excl = (query_end - global_start_sec) // bucket_sec + 1
    i_start = max(0, min(i_start, num_buckets_total))
    i_end_excl = min(i_end_excl, num_buckets_total)
    if i_start >= i_end_excl:
        raise ValueError("Query range results in empty slice.")

    L_raw = i_end_excl - i_start
    print(f"Original extracted slice: [{i_start}, {i_end_excl}) → {L_raw} buckets")

    # Expected prediction length after windowing
    L_pred_expected = max(0, L_raw - window_size)
    if L_pred_expected <= 0:
        raise ValueError(f"Slice too short ({L_raw} buckets) for window_size={window_size}. No predictions possible.")

    print(f"With window_size={window_size}, expected prediction length: {L_pred_expected}")

    anomalies = []

    # --- Step 2: Process each edge ---
    for idx, edge_name in enumerate(edges):
        pred_file = os.path.join(predictions_dir, f"pred_per_feature_{idx}.npy")
        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found. Skipping edge {edge_name}.")
            continue

        pred = np.load(pred_file)  # shape (L_pred, F)
        L_pred, num_features = pred.shape

        # Align predictions to correct timestamps
        if L_pred != L_pred_expected:
            print(f"Warning: Edge {edge_name}: prediction length {L_pred} ≠ expected {L_pred_expected}. "
                  f"Using min length.")
            use_len = min(L_pred, L_pred_expected)
        else:
            use_len = L_pred

        # Compute actual timestamps for predictions:
        # pred[t] corresponds to bucket index = i_start + window_size + t
        pred_timestamps = np.array([
            global_start_sec + (i_start + window_size + t) * bucket_sec
            for t in range(use_len)
        ])

        # Apply user query time filter (should be redundant but safe)
        mask_time = (pred_timestamps >= query_start) & (pred_timestamps <= query_end)

        for feat_idx in range(min(num_features, len(columns))):
            attr_name = columns[feat_idx]
            anomaly_mask = (pred[:use_len, feat_idx] == 1) & mask_time

            for t in np.where(anomaly_mask)[0]:
                ts = int(pred_timestamps[t])
                anomalies.append({
                    "edge": edge_name,
                    "attribute": attr_name,
                    "timestamp": ts
                })

    # --- Step 3: Save .npy ---
    dt = np.dtype([('edge', 'U128'), ('attribute', 'U32'), ('timestamp', 'i8')])
    if anomalies:
        arr = np.array([(a['edge'], a['attribute'], a['timestamp']) for a in anomalies], dtype=dt)
    else:
        arr = np.empty(0, dtype=dt)

    np.save(output_npy_path, arr)
    print(f"\n✅ Saved {len(anomalies)} Market trace anomaly records to {output_npy_path}")

    # --- Step 4: Generate report ---
    if not anomalies:
        print("\n🔍 No Market trace anomalies found in the specified time range.")
        return

    report = defaultdict(lambda: defaultdict(list))
    for a in anomalies:
        report[a['edge']][a['attribute']].append(a['timestamp'])

    report_output_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{pred_dir_for_report}"
    os.makedirs(report_output_dir, exist_ok=True)
    report_file_path = os.path.join(report_output_dir, f"Market_trace_anomaly_report_{anomaly_report_suffix}.txt")

    report_lines = []
    report_lines.append("\n📝 Market Trace Anomaly Report (Beijing Time = UTC+8):")
    report_lines.append("=" * 90)

    for edge in sorted(report.keys()):
        report_lines.append(f"\nEdge: {edge}")
        for attr in sorted(report[edge].keys()):
            timestamps = sorted(report[edge][attr])
            readable_entries = []
            for ts in timestamps:
                beijing_time = datetime.utcfromtimestamp(ts) + timedelta(hours=8)
                readable = f"{ts} ({beijing_time.strftime('%Y-%m-%d %H:%M:%S')} CST)"
                readable_entries.append(readable)
            report_lines.append(f"  - Attribute '{attr}': {len(timestamps)} anomalies at:")
            report_lines.append("      " + ", ".join(readable_entries))

    report_lines.append("\n💡 Note: Edge format is 'source->target:method'. CST = China Standard Time (UTC+8).")

    full_report_text = "\n".join(report_lines)
    print(full_report_text)

    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    print(f"\n✅ Full report saved to: {report_file_path}")


# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Market TRACE anomalies in a given time range.")
    parser.add_argument("--cloudbed", type=str, default="cloudbed-1")
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--start", type=int, required=True, help="Query start Unix timestamp")
    parser.add_argument("--end", type=int, required=True, help="Query end Unix timestamp")
    parser.add_argument("--output", type=str, default="market_trace_anomalies.npy")
    parser.add_argument("--report_suffix", type=str, default="custom")
    parser.add_argument("--bucket_sec", type=int, default=60)
    parser.add_argument("--output_folder_name", type=str, default="1215")
    parser.add_argument("--window_size", type=int, default=12,
                        help="Sliding window size used during TranAD inference (e.g., 12 from '12ws' in path)")

    args = parser.parse_args()

    pred_dir_path = Path(args.pred_dir).resolve()

    analyze_market_trace_anomalies_in_time_range(
        cloudbed=args.cloudbed,
        date_str=args.date,
        predictions_dir=str(pred_dir_path),
        time_range_query=(args.start, args.end),
        output_npy_path=args.output,
        anomaly_report_suffix=args.report_suffix,
        bucket_sec=args.bucket_sec,
        window_size=args.window_size,
        pred_dir_for_report=args.output_folder_name  # pass for report path
    )
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import argparse


def analyze_anomalies_in_time_range(
    metric_meta_path: str,
    predictions_dir: str,
    time_range_query: tuple,  # (start_unix, end_unix)
    output_npy_path: str,
    anomaly_report: str,
    pred_dir_name: str  # for report path
):
    # --- Step 1: Load metadata ---
    with open(metric_meta_path, 'r') as f:
        meta = json.load(f)

    entities = meta["entities"]          # e.g., ["db_001", "db_002", ...]
    features = meta["features"]          # e.g., ["ZJ-001-001_CPU_util_pct", ...]
    start_ts = meta["global_start_sec"]  # e.g., 1586534400
    bucket_sec = meta["bucket_sec"]      # e.g., 60
    num_buckets = meta["num_buckets"]    # e.g., 360

    anomalies = []  # list of dicts: {entity, attribute, timestamp}

    # --- Step 2: Process each entity ---
    for idx, entity in enumerate(entities):
        pred_file = os.path.join(predictions_dir, f"pred_per_feature_{idx}.npy")
        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found. Skipping entity {entity} (index {idx}).")
            continue

        pred = np.load(pred_file)  # shape (L, num_features), L=348
        L = pred.shape[0]
        num_features = pred.shape[1]

        if num_features != len(features):
            print(f"Warning: pred_per_feature_{idx} has {num_features} features, expected {len(features)}. "
                  f"Will only process first {min(num_features, len(features))}.")

        # Predictions cover the last L buckets of the full 360-bucket window
        offset = num_buckets - L
        if offset < 0:
            print(f"Warning: Prediction length ({L}) exceeds total buckets ({num_buckets}) for {entity}. Skipping.")
            continue

        # Generate timestamps for each prediction point
        pred_timestamps = np.array([
            start_ts + (offset + t) * bucket_sec for t in range(L)
        ])

        # Filter timestamps within query range
        mask_time = (pred_timestamps >= time_range_query[0]) & (pred_timestamps <= time_range_query[1])

        # For each feature (attribute)
        for feat_idx in range(min(num_features, len(features))):
            attr_name = features[feat_idx]
            anomaly_mask = (pred[:, feat_idx] == 1) & mask_time

            for t in np.where(anomaly_mask)[0]:
                ts = int(pred_timestamps[t])
                anomalies.append({
                    "entity": entity,
                    "attribute": attr_name,
                    "timestamp": ts
                })

    # --- Step 3: Save as structured NumPy array ---
    dt = np.dtype([('entity', 'U64'), ('attribute', 'U128'), ('timestamp', 'i8')])
    if anomalies:
        arr = np.array([
            (a['entity'], a['attribute'], a['timestamp']) for a in anomalies
        ], dtype=dt)
    else:
        arr = np.empty(0, dtype=dt)

    np.save(output_npy_path, arr)
    print(f"\n✅ Saved {len(anomalies)} anomaly records to {output_npy_path}")

    # --- Step 4: Generate human-readable report (Beijing Time = UTC+8) ---
    if not anomalies:
        print("\n🔍 No anomalies found in the specified time range.")
        return

    report = defaultdict(lambda: defaultdict(list))
    for a in anomalies:
        report[a['entity']][a['attribute']].append(a['timestamp'])

    # Output report directory
    report_output_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{pred_dir_name}"
    os.makedirs(report_output_dir, exist_ok=True)
    report_file_path = os.path.join(report_output_dir, f"Telecom_metric_A_anomaly_report_{anomaly_report}.txt")

    # Build report lines
    report_lines = []
    report_lines.append("\n📝 Detailed Telecom Metric CMDB Anomaly Report (Beijing Time = UTC+8):")
    report_lines.append("=" * 80)

    for entity in sorted(report.keys()):
        report_lines.append(f"\nEntity: {entity}")
        for attr in sorted(report[entity].keys()):
            timestamps = sorted(report[entity][attr])
            readable_entries = []
            for ts in timestamps:
                beijing_time = datetime.utcfromtimestamp(ts) + timedelta(hours=8)
                readable = f"{ts} ({beijing_time.strftime('%Y-%m-%d %H:%M:%S')} CST)"
                readable_entries.append(readable)
            report_lines.append(f"  - Attribute '{attr}': {len(timestamps)} anomalies at timestamps:")
            report_lines.append("      " + ", ".join(readable_entries))

    report_lines.append("\n💡 Note: 'CST' = China Standard Time (UTC+8).")

    full_report_text = "\n".join(report_lines)
    print(full_report_text)

    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    print(f"\n✅ Full report saved to: {report_file_path}")


# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Telecom anomalies in a given time range (Beijing Time).")
    parser.add_argument("--meta", type=str,
                        default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Telecom/metric/metadata_A_2020_04_11_60s.json",
                        help="Path to Telecom metadata JSON file")
    parser.add_argument("--pred_dir", type=str,
                        default="1120",
                        help="Base directory name (e.g., '1120') used in path construction")
    parser.add_argument("--start", type=int,
                        help="Query start time (Unix timestamp)",
                        default=1586534400)  # 2020-04-11 00:00:00 UTC
    parser.add_argument("--end", type=int,
                        help="Query end time (Unix timestamp)",
                        default=1586555940)  # 2020-04-11 05:59:00 UTC (approx)
    parser.add_argument("--output", type=str,
                        default="telecom_anomalies_in_range.npy",
                        help="Output .npy file path")
    parser.add_argument("--anomaly_report", type=str,
                        default="2020_04_11",
                        help="Suffix for report filename")

    args = parser.parse_args()

    predictions_dir = (
        f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/"
        f"{args.pred_dir}/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"
    )

    analyze_anomalies_in_time_range(
        metric_meta_path=args.meta,
        predictions_dir=predictions_dir,
        time_range_query=(args.start, args.end),
        output_npy_path=args.output,
        anomaly_report=args.anomaly_report,
        pred_dir_name=args.pred_dir
    )
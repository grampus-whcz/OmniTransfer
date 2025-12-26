import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


def analyze_anomalies_in_time_range(
    metric_meta_path: str,
    predictions_dir: str,
    time_range_query: tuple,  # (start_unix, end_unix)
    output_npy_path: str,
    anomaly_report: str
):
    # --- Step 1: Load metadata ---
    with open(metric_meta_path, 'r') as f:
        meta = json.load(f)

    tc_list = meta["cmdb_list"]
    columns = meta["kpi_list"]  # e.g., ['rr', 'sr', 'cnt', 'mrt']
    start_ts = meta["time_range_sec"][0]  # e.g., 1614787200
    bucket_sec = meta["bucket_sec"]       # e.g., 60
    num_buckets = meta["num_buckets"]     # e.g., 1444

    anomalies = []  # list of dicts: {entity, attribute, timestamp}

    # --- Step 2: Process each entity ---
    for idx, entity in enumerate(tc_list):
        pred_file = os.path.join(predictions_dir, f"pred_per_feature_{idx}.npy")
        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found. Skipping.")
            continue

        pred = np.load(pred_file)  # shape (L, num_features)
        L = pred.shape[0]
        num_features = pred.shape[1]

        # Validate feature count
        if num_features != len(columns):
            print(f"Warning: pred_per_feature_{idx} has {num_features} features, expected {len(columns)}. "
                  f"Will only process first {min(num_features, len(columns))}.")

        # Infer offset assuming predictions cover the last L buckets
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
        for feat_idx in range(min(num_features, len(columns))):
            attr_name = columns[feat_idx]
            anomaly_mask = (pred[:, feat_idx] == 1) & mask_time

            for t in np.where(anomaly_mask)[0]:
                ts = int(pred_timestamps[t])
                anomalies.append({
                    "entity": entity,
                    "attribute": attr_name,
                    "timestamp": ts
                })

    # --- Step 3: Save as structured NumPy array ---
    if anomalies:
        # Compute safe string lengths
        max_ent_len = max(len(a['entity']) for a in anomalies)
        max_attr_len = max(len(a['attribute']) for a in anomalies)
        ent_dtype = f'U{max(max_ent_len + 16, 64)}'
        attr_dtype = f'U{max(max_attr_len + 8, 32)}'
    else:
        ent_dtype, attr_dtype = 'U64', 'U32'

    dt = np.dtype([('entity', ent_dtype), ('attribute', attr_dtype), ('timestamp', 'i8')])

    if anomalies:
        arr = np.array([
            (a['entity'], a['attribute'], a['timestamp']) for a in anomalies
        ], dtype=dt)
    else:
        arr = np.empty(0, dtype=dt)

    np.save(output_npy_path, arr)
    print(f"\n✅ Saved {len(anomalies)} anomaly records to {output_npy_path}")

    # --- Step 4: Generate human-readable report with Beijing Time (UTC+8) ---
    if not anomalies:
        print("\n🔍 No anomalies found in the specified time range.")
        return

    report = defaultdict(lambda: defaultdict(list))
    for a in anomalies:
        report[a['entity']][a['attribute']].append(a['timestamp'])

    # Define output report file path
    report_output_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.pred_dir}"
    os.makedirs(report_output_dir, exist_ok=True)
    report_file_path = os.path.join(report_output_dir, f"Bank_metric_container_anomaly_report_{anomaly_report}.txt")

    # Prepare full report as a list of lines
    report_lines = []
    report_lines.append("\n📝 Detailed metric container Anomaly Report (Beijing Time = UTC+8):")
    report_lines.append("=" * 80)

    for entity in sorted(report.keys()):
        report_lines.append(f"\nEntity: {entity}")
        for attr in sorted(report[entity].keys()):
            timestamps = sorted(report[entity][attr])
            readable_entries = []
            for ts in timestamps:
                # Convert to Beijing Time: UTC + 8 hours
                beijing_time = datetime.utcfromtimestamp(ts) + timedelta(hours=8)
                readable = f"{ts} ({beijing_time.strftime('%Y-%m-%d %H:%M:%S')} CST)"
                readable_entries.append(readable)
            report_lines.append(f"  - Attribute '{attr}': {len(timestamps)} anomalies at timestamps:")
            report_lines.append("      " + ", ".join(readable_entries))

    report_lines.append("\n💡 Note: 'CST' = China Standard Time (UTC+8).")

    # Output to both console and file
    full_report_text = "\n".join(report_lines)
    print(full_report_text)

    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    print(f"\n✅ Full report saved to: {report_file_path}")


# -----------------------------
# Example Usage (Command-line Interface)
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract anomalies in a given time range (displayed in Beijing Time).")
    parser.add_argument("--meta", type=str,
                        help="Path to metadata JSON file",
                        default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/metric/metadata_container_2021_03_06_120s.json")
    parser.add_argument("--pred_dir", type=str,
                        default="1116",
                        help="Directory with pred_per_feature_*.npy for logs")
    parser.add_argument("--start", type=int,
                        help="Query start time (Unix timestamp)",
                        default=1614960000)
    parser.add_argument("--end", type=int,
                        help="Query end time (Unix timestamp)",
                        default=1615046400)
    parser.add_argument("--output", type=str, default="anomalies_in_range.npy",
                        help="Output .npy file path")
    parser.add_argument("--anomaly_report", type=str, default="2021_03_06_14_to_15",
                        help="Output .txt file name")

    args = parser.parse_args()
    
    predictions_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.pred_dir}/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"

    analyze_anomalies_in_time_range(
        metric_meta_path=args.meta,
        predictions_dir=predictions_dir,
        time_range_query=(args.start, args.end),
        output_npy_path=args.output,
        anomaly_report=args.anomaly_report
    )
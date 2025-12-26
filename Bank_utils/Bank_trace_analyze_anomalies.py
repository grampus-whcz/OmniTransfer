import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def analyze_trace_anomalies_in_time_range(
    trace_meta_path: str,
    predictions_dir: str,
    time_range_query: tuple,  # (start_unix, end_unix)
    output_npy_path: str,
    anomaly_report: str
):
    # --- Step 1: Load trace metadata ---
    with open(trace_meta_path, 'r') as f:
        meta = json.load(f)

    edges = meta["edges"]                 # list of "src->dst"
    columns = meta["features"]            # ["duration", "frequency"]
    start_ts = meta["global_start_sec"]   # 1614787199
    bucket_sec = meta["bucket_sec"]       # 60
    num_buckets = meta["num_buckets"]     # 1446

    num_edges = len(edges)
    print(f"Loaded {num_edges} edges from metadata.")

    anomalies = []  # list of dicts: {edge, attribute, timestamp}

    # --- Step 2: Process each edge (indexed 0 to 55) ---
    for idx in range(num_edges):
        edge_name = edges[idx]
        pred_file = os.path.join(predictions_dir, f"pred_per_feature_{idx}.npy")
        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found. Skipping edge {edge_name}.")
            continue

        pred = np.load(pred_file)  # shape (L, ?)
        L = pred.shape[0]
        num_features = pred.shape[1]

        if num_features != len(columns):
            print(f"Warning: pred_per_feature_{idx} has {num_features} features, expected {len(columns)}. "
                  f"Will process first {min(num_features, len(columns))}.")

        # Infer alignment: assume prediction covers last L buckets
        offset = num_buckets - L  # e.g., if L=1434, offset=12

        # Generate timestamps for each prediction point
        pred_timestamps = np.array([
            start_ts + (offset + t) * bucket_sec for t in range(L)
        ])

        # Filter by query time range
        mask_time = (pred_timestamps >= time_range_query[0]) & (pred_timestamps <= time_range_query[1])

        for feat_idx in range(min(num_features, len(columns))):
            attr_name = columns[feat_idx]
            anomaly_mask = (pred[:, feat_idx] == 1) & mask_time

            for t in np.where(anomaly_mask)[0]:
                ts = int(pred_timestamps[t])
                anomalies.append({
                    "edge": edge_name,
                    "attribute": attr_name,
                    "timestamp": ts
                })

    # --- Step 3: Save as structured NumPy array ---
    if anomalies:
        dt = np.dtype([
            ('edge', 'U64'),          # "IG01->Tomcat01" fits in 64 chars
            ('attribute', 'U16'),     # "duration", "frequency"
            ('timestamp', 'i8')
        ])
        arr = np.array([
            (a['edge'], a['attribute'], a['timestamp']) for a in anomalies
        ], dtype=dt)
    else:
        dt = np.dtype([('edge', 'U64'), ('attribute', 'U16'), ('timestamp', 'i8')])
        arr = np.empty(0, dtype=dt)

    np.save(output_npy_path, arr)
    print(f"\n✅ Saved {len(anomalies)} trace anomaly records to {output_npy_path}")

    # --- Step 4: Generate human-readable report ---
    if not anomalies:
        print("\n🔍 No trace anomalies found in the specified time range.")
        return

    from collections import defaultdict
    report = defaultdict(lambda: defaultdict(list))
    for a in anomalies:
        report[a['edge']][a['attribute']].append(a['timestamp'])
   
   # Define output report file path
    report_output_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.pred_dir}"
    os.makedirs(report_output_dir, exist_ok=True)
    report_file_path = os.path.join(report_output_dir, f"Bank_trace_anomaly_report_{anomaly_report}.txt")
   
    # Prepare full report as a list of lines
    report_lines = []
    report_lines.append("\n📝 Detailed Trace Anomaly Report (Beijing Time = UTC+8):")
    report_lines.append("=" * 80)

    for edge in sorted(report.keys()):
        report_lines.append(f"\nEdge: {edge}")
        for attr in sorted(report[edge].keys()):
            timestamps = sorted(report[edge][attr])
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
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract TRACE anomalies in a given time range.")
    parser.add_argument("--meta", type=str, required=True,
                        help="Path to trace metadata JSON file")
    parser.add_argument("--pred_dir", type=str,
                        default="1116",
                        help="Directory with pred_per_feature_*.npy for logs")
    parser.add_argument("--start", type=int, required=True,
                        help="Query start time (Unix timestamp)")
    parser.add_argument("--end", type=int, required=True,
                        help="Query end time (Unix timestamp)")
    parser.add_argument("--output", type=str, default="trace_anomalies_in_range.npy",
                        help="Output .npy file path")
    parser.add_argument("--anomaly_report", type=str, default="2021_03_06_14_to_15",
                        help="Output .txt file name")

    args = parser.parse_args()
    
    predictions_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.pred_dir}/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"

    analyze_trace_anomalies_in_time_range(
        trace_meta_path=args.meta,
        predictions_dir=predictions_dir,
        time_range_query=(args.start, args.end),
        output_npy_path=args.output,
        anomaly_report=args.anomaly_report
    )
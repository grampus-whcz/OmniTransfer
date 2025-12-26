import os
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import argparse


def analyze_telecom_trace_anomalies(
    info_json_path: str,
    predictions_dir: str,
    time_range_query: tuple,
    output_npy_path: str,
    anomaly_report: str,
    pred_dir_name: str
):
    # --- Step 1: Load trace info ---
    with open(info_json_path, 'r') as f:
        info = json.load(f)

    entities = info["common_edges"]      # list of 132 edge strings
    features = info["features"]          # ["duration", "frequency"]
    bucket_sec = info["bucket_sec"]      # 60
    duration_minutes = info["duration_minutes"]  # 30
    num_buckets = duration_minutes       # because aligned_shape[1] = 30
    fault_start_time_str = info["fault_start_time"]  # "2020-04-11 00:00:00"

    # Parse start timestamp (UTC)
    fault_start_dt = datetime.strptime(fault_start_time_str, "%Y-%m-%d %H:%M:%S")
    start_ts = int(fault_start_dt.timestamp())  # 1586534400

    anomalies = []

    # --- Step 2: Process each entity (edge) ---
    for idx, entity in enumerate(entities):
        pred_file = os.path.join(predictions_dir, f"pred_per_feature_{idx}.npy")
        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found. Skipping edge {idx}.")
            continue

        pred = np.load(pred_file)  # Expected shape: (L, 2), L <= 30
        L = pred.shape[0]
        num_feat = pred.shape[1]

        if num_feat != len(features):
            print(f"Warning: Entity {idx} has {num_feat} features, expected {len(features)}. Using min.")

        # Since data is already aligned to [0, 30) minutes, offset = 0
        offset = 0
        if L > num_buckets:
            print(f"Warning: Prediction length {L} > expected {num_buckets}. Clipping.")
            L = num_buckets

        # Generate timestamps: from start_ts, every 60s, total L points
        pred_timestamps = np.array([
            start_ts + (offset + t) * bucket_sec for t in range(L)
        ])

        # Time range filter
        mask_time = (pred_timestamps >= time_range_query[0]) & (pred_timestamps <= time_range_query[1])

        # Check per feature
        for feat_idx in range(min(num_feat, len(features))):
            attr = features[feat_idx]
            anomaly_mask = (pred[:, feat_idx] == 1) & mask_time
            for t in np.where(anomaly_mask)[0]:
                ts = int(pred_timestamps[t])
                anomalies.append({
                    "entity": entity,
                    "attribute": attr,
                    "timestamp": ts
                })

    # --- Step 3: Save structured array ---
    dt = np.dtype([('entity', 'U256'), ('attribute', 'U16'), ('timestamp', 'i8')])
    if anomalies:
        arr = np.array([(a['entity'], a['attribute'], a['timestamp']) for a in anomalies], dtype=dt)
    else:
        arr = np.empty(0, dtype=dt)

    np.save(output_npy_path, arr)
    print(f"\n✅ Saved {len(anomalies)} anomaly records to {output_npy_path}")

    # --- Step 4: Generate human-readable report (Beijing Time) ---
    if not anomalies:
        print("\n🔍 No anomalies found in the specified time range.")
        return

    report = defaultdict(lambda: defaultdict(list))
    for a in anomalies:
        report[a['entity']][a['attribute']].append(a['timestamp'])

    report_output_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{pred_dir_name}"
    os.makedirs(report_output_dir, exist_ok=True)
    report_file_path = os.path.join(report_output_dir, f"Telecom_trace_anomaly_report_{anomaly_report}.txt")

    report_lines = []
    report_lines.append("\n📝 Telecom Trace (Call Edge) Anomaly Report (Beijing Time = UTC+8):")
    report_lines.append("=" * 80)

    for entity in sorted(report.keys()):
        report_lines.append(f"\nEdge: {entity}")
        for attr in sorted(report[entity].keys()):
            timestamps = sorted(report[entity][attr])
            readable_entries = []
            for ts in timestamps:
                beijing_time = datetime.utcfromtimestamp(ts) + timedelta(hours=8)
                readable = f"{ts} ({beijing_time.strftime('%Y-%m-%d %H:%M:%S')} CST)"
                readable_entries.append(readable)
            report_lines.append(f"  - Feature '{attr}': {len(timestamps)} anomalies at:")
            report_lines.append("      " + ", ".join(readable_entries))

    report_lines.append("\n💡 Note: Entities are call graph edges. Time aligned to fault start (2020-04-11 00:00:00 UTC).")

    full_report_text = "\n".join(report_lines)
    print(full_report_text)

    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    print(f"\n✅ Full report saved to: {report_file_path}")


# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Telecom trace (call edge) anomalies.")
    parser.add_argument("--info_json", type=str,
                        default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/Telecom_utils/telecom_trace_case_set/fault_case_03_pod_CPU_fault_docker_003_2020-04-11/info.json",
                        help="Path to trace info.json")
    parser.add_argument("--pred_dir", type=str,
                        default="1120",
                        help="Base experiment directory name (e.g., '1120')")
    parser.add_argument("--start", type=int,
                        default=1586534400,  # 2020-04-11 00:00:00 UTC
                        help="Query start Unix timestamp")
    parser.add_argument("--end", type=int,
                        default=1586534400 + 30 * 60,  # 30 minutes later
                        help="Query end Unix timestamp")
    parser.add_argument("--output", type=str,
                        default="telecom_trace_anomalies.npy",
                        help="Output .npy path")
    parser.add_argument("--anomaly_report", type=str,
                        default="fault_case_03_2020_04_11",
                        help="Report filename suffix")

    args = parser.parse_args()

    predictions_dir = (
        f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/"
        f"{args.pred_dir}/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"
    )

    analyze_telecom_trace_anomalies(
        info_json_path=args.info_json,
        predictions_dir=predictions_dir,
        time_range_query=(args.start, args.end),
        output_npy_path=args.output,
        anomaly_report=args.anomaly_report,
        pred_dir_name=args.pred_dir
    )
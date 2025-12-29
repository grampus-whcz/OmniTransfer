import os
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import argparse


def analyze_entityB_anomalies(
    metric_meta_path: str,
    predictions_dir: str,
    time_range_query: tuple,
    output_npy_path: str,
    anomaly_report: str,
    pred_dir_name: str
):
    # --- Step 1: Load metadata ---
    with open(metric_meta_path, 'r') as f:
        meta = json.load(f)

    # Only one real entity
    real_entity = meta["entities"][0]      # "osb_001"
    features = meta["features"]            # ["avg_time", "num", "succee_num", "succee_rate"]
    start_ts = meta["global_start_sec"]    # 1586534400
    bucket_sec = meta["bucket_sec"]        # 60
    num_buckets = meta["num_buckets"]      # 361

    anomalies = []

    # --- Step 2: ONLY process index 0 (real entity) ---
    pred_file = os.path.join(predictions_dir, "pred_per_feature_0.npy")
    if not os.path.exists(pred_file):
        print(f"Error: {pred_file} not found. Aborting.")
        return

    pred = np.load(pred_file)  # shape (L, 4), L should be <= 361
    L = pred.shape[0]
    num_features = pred.shape[1]

    if num_features != len(features):
        print(f"Warning: Expected {len(features)} features, got {num_features}. Using min.")

    # Predictions cover last L buckets
    offset = num_buckets - L
    if offset < 0:
        print(f"Warning: Prediction length ({L}) > total buckets ({num_buckets}). Clipping offset to 0.")
        offset = 0

    # Generate timestamps
    pred_timestamps = np.array([
        start_ts + (offset + t) * bucket_sec for t in range(L)
    ])

    # Time range mask
    mask_time = (pred_timestamps >= time_range_query[0]) & (pred_timestamps <= time_range_query[1])

    # Process each feature
    for feat_idx in range(min(num_features, len(features))):
        attr_name = features[feat_idx]
        anomaly_mask = (pred[:, feat_idx] == 1) & mask_time
        for t in np.where(anomaly_mask)[0]:
            ts = int(pred_timestamps[t])
            anomalies.append({
                "entity": real_entity,
                "attribute": attr_name,
                "timestamp": ts
            })

    # --- Step 3: Save structured array ---
    dt = np.dtype([('entity', 'U64'), ('attribute', 'U32'), ('timestamp', 'i8')])
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
    report_file_path = os.path.join(report_output_dir, f"Telecom_metric_B_anomaly_report_{anomaly_report}.txt")

    report_lines = []
    report_lines.append("\n📝 Telecom Entity-B (app_service) Anomaly Report (Beijing Time = UTC+8):")
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
            report_lines.append(f"  - Attribute '{attr}': {len(timestamps)} anomalies at:")
            report_lines.append("      " + ", ".join(readable_entries))

    report_lines.append("\n💡 Note: Only real entity 'osb_001' is analyzed (index 0). Replicated entities ignored.")

    full_report_text = "\n".join(report_lines)
    print(full_report_text)

    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    print(f"\n✅ Full report saved to: {report_file_path}")


# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze anomalies for Telecom Entity-B (single-entity case).")
    parser.add_argument("--meta", type=str,
                        default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Telecom/metric/metadata_B_2020_04_11_60s.json",
                        help="Path to metadata_B_*.json")
    parser.add_argument("--pred_dir", type=str,
                        default="1120",
                        help="Base experiment directory name (e.g., '1120')")
    parser.add_argument("--start", type=int,
                        default=1586534400,  # 2020-04-11 00:00:00 UTC
                        help="Query start Unix timestamp")
    parser.add_argument("--end", type=int,
                        default=1586556000,  # 2020-04-11 06:00:00 UTC
                        help="Query end Unix timestamp")
    parser.add_argument("--output", type=str,
                        default="telecom_entityB_anomalies.npy",
                        help="Output .npy path")
    parser.add_argument("--anomaly_report", type=str,
                        default="2020_04_11",
                        help="Report filename suffix")

    args = parser.parse_args()

    predictions_dir = (
        f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/"
        f"{args.pred_dir}/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"
    )

    analyze_entityB_anomalies(
        metric_meta_path=args.meta,
        predictions_dir=predictions_dir,
        time_range_query=(args.start, args.end),
        output_npy_path=args.output,
        anomaly_report=args.anomaly_report,
        pred_dir_name=args.pred_dir
    )
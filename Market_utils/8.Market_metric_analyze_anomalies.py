#!/usr/bin/env python3

import os
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import argparse


def analyze_anomalies_in_time_range(
    metric_meta_path: str,
    predictions_dir: str,
    time_range_query: tuple,  # (start_unix, end_unix) — this is the ONLINE window
    output_npy_path: str,
    anomaly_report: str,
    pred_dir_name: str,
    bucket_sec: int,  # we can get it from meta, but ensure consistency
    metric_type: str
):
    # --- Step 1: Load metadata ---
    with open(metric_meta_path, 'r') as f:
        meta = json.load(f)

    entity_list = meta["entity_list"]
    kpi_list = meta.get("kpi_list", meta.get("columns", []))
    meta_bucket_sec = meta["bucket_sec"]
    
    if meta_bucket_sec != bucket_sec:
        print(f"⚠️ Warning: metadata bucket_sec ({meta_bucket_sec}) != inferred bucket_sec ({bucket_sec})")

    anomalies = []

    # The ONLINE window is exactly [start_ts, end_ts]
    start_ts = time_range_query[0]
    end_ts = time_range_query[1]
    expected_num_buckets = (end_ts - start_ts) // bucket_sec

    # --- Step 2: Process each entity ---
    for idx, entity in enumerate(entity_list):
        pred_file = os.path.join(predictions_dir, f"pred_per_feature_{idx}.npy")
        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found. Skipping entity {entity}.")
            continue

        pred = np.load(pred_file)  # shape (L, num_features)
        L = pred.shape[0]
        num_features = pred.shape[1]

        # In Market short-window setting, L should equal expected_num_buckets
        if L != expected_num_buckets:
            print(f"Info: Entity {entity}: prediction length {L}, expected {expected_num_buckets} based on time range.")

        # Generate timestamps starting from --start
        pred_timestamps = np.array([
            start_ts + t * bucket_sec for t in range(L)
        ])

        # Only consider timestamps within [start, end] (should be all, but safe)
        mask_time = (pred_timestamps >= start_ts) & (pred_timestamps <= end_ts)

        for feat_idx in range(min(num_features, len(kpi_list))):
            attr_name = kpi_list[feat_idx]
            anomaly_mask = (pred[:, feat_idx] == 1) & mask_time

            for t in np.where(anomaly_mask)[0]:
                ts = int(pred_timestamps[t])
                anomalies.append({
                    "entity": entity,
                    "attribute": attr_name,
                    "timestamp": ts
                })

    # --- Step 3: Save structured array ---
    dt = np.dtype([('entity', 'U64'), ('attribute', 'U64'), ('timestamp', 'i8')])
    arr = np.array([
        (a['entity'], a['attribute'], a['timestamp']) for a in anomalies
    ], dtype=dt) if anomalies else np.empty(0, dtype=dt)

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
    report_file_path = os.path.join(report_output_dir, f"Market_metric_{metric_type}_anomaly_report_{anomaly_report}.txt")

    report_lines = []
    report_lines.append("\n📝 Market Container Metric Anomaly Report (Beijing Time = UTC+8)")
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

    report_lines.append("\n💡 Note: 'CST' = China Standard Time (UTC+8).")

    full_report_text = "\n".join(report_lines)
    print(full_report_text)

    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)

    print(f"\n✅ Full report saved to: {report_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Market container metric anomalies (short-window mode).")
    parser.add_argument("--meta", type=str, required=True,
                        help="Path to Market container metadata JSON file (e.g., metadata_container_cloudbed-1_2022_03_20_60s.json)")
    parser.add_argument("--pred_dir", type=str, default="1120",
                        help="Experiment folder name (e.g., 1120)")
    parser.add_argument("--start", type=int, required=True,
                        help="Start of ONLINE window (Unix timestamp)")
    parser.add_argument("--end", type=int, required=True,
                        help="End of ONLINE window (Unix timestamp)")
    parser.add_argument("--output", type=str, default="market_container_anomalies.npy",
                        help="Output .npy file path")
    parser.add_argument("--anomaly_report", type=str, default="result",
                        help="Suffix for report filename")
    parser.add_argument("--metric_type", type=str, default="container",
                        help="Metrics types, e.g., 'container', 'mesh', 'node', 'runtime', 'service'")

    args = parser.parse_args()

    # Infer bucket_sec from meta filename (robust way)
    meta_basename = os.path.basename(args.meta)
    # Example: metadata_container_cloudbed-1_2022_03_20_60s.json → extract '60'
    try:
        bucket_sec = int(meta_basename.split('_')[-1].replace('s.json', ''))
    except Exception as e:
        raise ValueError(f"Cannot infer bucket_sec from filename: {meta_basename}") from e

    predictions_dir = (
        f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/"
        f"{args.pred_dir}/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"
    )

    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    analyze_anomalies_in_time_range(
        metric_meta_path=args.meta,
        predictions_dir=predictions_dir,
        time_range_query=(args.start, args.end),
        output_npy_path=args.output,
        anomaly_report=args.anomaly_report,
        pred_dir_name=args.pred_dir,
        bucket_sec=bucket_sec,
        metric_type=args.metric_type
    )
# import os
# import json
# import numpy as np
# from datetime import datetime, timedelta
# from collections import defaultdict
# from pathlib import Path


# def detect_log_anomalies_zscore(
#     data_npy_path: str,
#     meta_npy_path: str,
#     log_pattern_json_path: str,
#     time_range_query: tuple = None,  # (start_unix, end_unix), optional
#     z_threshold: float = 3.0,
#     output_npy_path: str = "log_anomalies_zscore.npy"
# ):
#     """
#     Detect anomalies in log count time series using Z-score.
    
#     Args:
#         data_npy_path: path to (14, 1440, 175) .npy file
#         meta_npy_path: path to metadata .npy (contains pods, timestamps, etc.)
#         log_pattern_json_path: path to log_istio_patterns.json
#         time_range_query: optional (start_ts, end_ts) to filter output
#         z_threshold: threshold for |z-score| to flag anomaly
#         output_npy_path: where to save anomaly records
#     """
#     # --- Step 1: Load data and metadata ---
#     data = np.load(data_npy_path)  # (14, 1440, 175)
#     meta = np.load(meta_npy_path, allow_pickle=True).item()
    
#     pods = meta['pods']  # list of 14 entity names
#     timestamps = np.array(meta['timestamps'])  # shape (1440,)
#     num_entities, num_timesteps, num_attrs = data.shape

#     assert len(pods) == num_entities
#     assert len(timestamps) == num_timesteps

#     # Load log patterns for interpretation
#     with open(log_pattern_json_path, 'r') as f:
#         log_patterns = json.load(f)
#     java_patterns = log_patterns.get("Java", [])
#     if len(java_patterns) != num_attrs:
#         print(f"⚠️ Warning: Expected {num_attrs} Java patterns, got {len(java_patterns)}")

#     # --- Step 2: Compute Z-scores per (entity, attribute) ---
#     anomalies = []

#     for e_idx in range(num_entities):
#         entity = pods[e_idx]
#         entity_data = data[e_idx]  # (1440, 175)

#         for a_idx in range(num_attrs):
#             ts_series = entity_data[:, a_idx]  # (1440,)

#             # Skip if all zeros (no logs for this pattern)
#             if np.all(ts_series == 0):
#                 continue

#             mean = np.mean(ts_series)
#             std = np.std(ts_series)

#             if std == 0:
#                 z_scores = np.zeros_like(ts_series)
#             else:
#                 z_scores = (ts_series - mean) / std

#             # Find anomalies
#             anomaly_indices = np.where(np.abs(z_scores) > z_threshold)[0]

#             for t_idx in anomaly_indices:
#                 ts_unix = int(timestamps[t_idx])
                
#                 # Optional: filter by time range
#                 if time_range_query is not None:
#                     if not (time_range_query[0] <= ts_unix <= time_range_query[1]):
#                         continue

#                 anomalies.append({
#                     "entity": entity,
#                     "attribute": f"log_pattern_{a_idx}",  # or use actual pattern if needed
#                     "timestamp": ts_unix,
#                     "z_score": float(z_scores[t_idx]),
#                     "count": int(ts_series[t_idx])
#                 })

#     # --- Step 3: Save structured array ---
#     if anomalies:
#         dt = np.dtype([
#             ('entity', 'U32'),
#             ('attribute', 'U20'),
#             ('timestamp', 'i8'),
#             ('z_score', 'f4'),
#             ('count', 'i4')
#         ])
#         arr = np.array([
#             (a['entity'], a['attribute'], a['timestamp'], a['z_score'], a['count'])
#             for a in anomalies
#         ], dtype=dt)
#     else:
#         dt = np.dtype([('entity', 'U32'), ('attribute', 'U20'), ('timestamp', 'i8'), ('z_score', 'f4'), ('count', 'i4')])
#         arr = np.empty(0, dtype=dt)

#     np.save(output_npy_path, arr)
#     print(f"✅ Detected {len(anomalies)} log anomalies. Saved to {output_npy_path}")

#     # --- Step 4: Human-readable report (Beijing Time) ---
#     if not anomalies:
#         print("\n🔍 No anomalies detected.")
#         return

#     report = defaultdict(lambda: defaultdict(list))
#     for a in anomalies:
#         report[a['entity']][a['attribute']].append((a['timestamp'], a['z_score'], a['count']))

#     print("\n📝 Log Anomaly Report (Beijing Time = UTC+8):")
#     print("=" * 80)
#     for entity in sorted(report.keys()):
#         print(f"\nEntity: {entity}")
#         for attr in sorted(report[entity].keys()):
#             entries = sorted(report[entity][attr])  # sort by timestamp
#             print(f"  - Attribute: {attr} ({java_patterns[int(attr.split('_')[-1])] if attr.startswith('log_pattern_') and int(attr.split('_')[-1]) < len(java_patterns) else 'N/A'})")
#             for ts, z, cnt in entries:
#                 beijing_time = datetime.utcfromtimestamp(ts) + timedelta(hours=8)
#                 print(f"      TS: {ts} ({beijing_time.strftime('%Y-%m-%d %H:%M:%S')} CST) | Count={cnt}, Z={z:.2f}")
#     print("\n💡 Tip: High Z-score means unusual spike/drop in log frequency.")


# # -----------------------------
# # CLI Entry Point
# # -----------------------------
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Detect anomalies in service log count time series using Z-score.")
#     parser.add_argument("--data", type=str, default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/log/temp_data/raw_data/train_valid/2021_03_05/raw_log/service_log_patterns_count.npy")
#     parser.add_argument("--meta", type=str, default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/log/temp_data/raw_data/train_valid/2021_03_05/raw_log/service_log_patterns_count_meta.npy")
#     parser.add_argument("--patterns", type=str, default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/log/temp_data/analysis/log/log_istio_patterns.json")
#     parser.add_argument("--start", type=int, help="Start Unix timestamp (optional)")
#     parser.add_argument("--end", type=int, help="End Unix timestamp (optional)")
#     parser.add_argument("--z_thres", type=float, default=3.0)
#     parser.add_argument("--output", type=str, default="log_anomalies_zscore.npy")

#     args = parser.parse_args()

#     time_range = None
#     if args.start is not None and args.end is not None:
#         time_range = (args.start, args.end)

#     detect_log_anomalies_zscore(
#         data_npy_path=args.data,
#         meta_npy_path=args.meta,
#         log_pattern_json_path=args.patterns,
#         time_range_query=time_range,
#         z_threshold=args.z_thres,
#         output_npy_path=args.output
#     )

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


def analyze_log_anomalies_in_time_range(
    log_meta_path: str,
    log_patterns_json_path: str,
    predictions_dir: str,
    time_range_query: tuple,  # (start_unix, end_unix)
    output_npy_path: str,
    anomaly_report: str
):
    # --- Step 1: Load log metadata (.npy) ---
    meta = np.load(log_meta_path, allow_pickle=True).item()
    
    pods = meta["pods"]                      # e.g., ['IG01', 'IG02', ...]
    timestamps = np.array(meta["timestamps"])  # sorted list of bucket start times (Unix)
    max_attributes = meta["max_attributes"]   # e.g., 175
    service_lang_map = meta.get("service_language_map", {})
    
    # --- Step 2: Load log patterns (for Java) ---
    with open(log_patterns_json_path, 'r') as f:
        patterns_dict = json.load(f)
    java_patterns = patterns_dict.get("Java", [])
    
    if len(java_patterns) != max_attributes:
        print(f"⚠️ Warning: Expected {max_attributes} Java patterns, but loaded {len(java_patterns)}.")
        # Pad or truncate to match
        if len(java_patterns) < max_attributes:
            java_patterns += [f"<UNKNOWN_PATTERN_{i}>" for i in range(len(java_patterns), max_attributes)]
        else:
            java_patterns = java_patterns[:max_attributes]

    anomalies = []  # list of dicts: {pod, pattern_id, pattern_str, timestamp}

    # --- Step 3: Process each pod ---
    for idx, pod in enumerate(pods):
        pred_file = os.path.join(predictions_dir, f"pred_per_feature_{idx}.npy")
        if not os.path.exists(pred_file):
            print(f"Warning: {pred_file} not found. Skipping pod {pod}.")
            continue

        pred = np.load(pred_file)  # shape (L, max_attributes)
        L = pred.shape[0]
        num_features = pred.shape[1]

        if num_features != max_attributes:
            print(f"Warning: pred_per_feature_{idx} has {num_features} features, expected {max_attributes}. "
                  f"Will only process first {min(num_features, max_attributes)}.")
            actual_features = min(num_features, max_attributes)
        else:
            actual_features = max_attributes

        # Assume predictions align with the last L timestamps in meta["timestamps"]
        if L > len(timestamps):
            print(f"Warning: Prediction length ({L}) exceeds available timestamps ({len(timestamps)}) for {pod}. Skipping.")
            continue

        # Align: predictions correspond to timestamps[-L:]
        pred_timestamps = timestamps[-L:]

        # Filter by query time range
        mask_time = (pred_timestamps >= time_range_query[0]) & (pred_timestamps <= time_range_query[1])

        # For each log pattern (feature)
        for feat_idx in range(actual_features):
            anomaly_mask = (pred[:, feat_idx] == 1) & mask_time
            for t in np.where(anomaly_mask)[0]:
                ts = int(pred_timestamps[t])
                anomalies.append({
                    "pod": pod,
                    "pattern_id": feat_idx,
                    "pattern_str": java_patterns[feat_idx],
                    "timestamp": ts
                })

        # --- Step 4: Save structured array ---
        dt = np.dtype([
            ('pod', 'U32'),
            ('pattern_id', 'i4'),
            ('pattern_str', 'U512'),  # may need longer if patterns are very long
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

        # --- Step 5: Human-readable report (Beijing Time) ---
        if not anomalies:
            print("\n🔍 No log anomalies found in the specified time range.")
            return        

        # Group by pod -> pattern_id
        report = defaultdict(lambda: defaultdict(list))
        for a in anomalies:
            report[a['pod']][a['pattern_id']].append(a['timestamp'])
            
        # Define output report file path
        report_output_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.pred_dir}"
        os.makedirs(report_output_dir, exist_ok=True)
        report_file_path = os.path.join(report_output_dir, f"Bank_log_anomaly_report_{anomaly_report}.txt")

        # Prepare full report as a list of lines
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
    parser = argparse.ArgumentParser(description="Extract LOG anomalies in a given time range (Beijing Time).")
    parser.add_argument("--log_meta", type=str,
                        default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/log/temp_data/raw_data/train_valid/2021_03_06/raw_log/service_log_patterns_count_meta.npy",
                        help="Path to log metadata .npy file")
    parser.add_argument("--log_patterns", type=str,
                        default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/log/temp_data/analysis/log/log_istio_patterns.json",
                        help="Path to log patterns JSON")
    parser.add_argument("--pred_dir", type=str,
                        default="1116",
                        help="Directory with pred_per_feature_*.npy for logs")
    parser.add_argument("--start", type=int,
                        help="Query start time (Unix timestamp)",
                        default=1614960000)  # example: 2021-03-06 00:00:00 UTC
    parser.add_argument("--end", type=int,
                        help="Query end time (Unix timestamp)",
                        default=1615046400)   # example: 2021-03-07 00:00:00 UTC
    parser.add_argument("--output", type=str, default="log_anomalies_in_range.npy",
                        help="Output .npy file path")
    
    parser.add_argument("--anomaly_report", type=str, default="2021_03_06_14_to_15",
                        help="Output .txt file name")

    args = parser.parse_args()
    
    predictions_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.pred_dir}/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"

    analyze_log_anomalies_in_time_range(
        log_meta_path=args.log_meta,
        log_patterns_json_path=args.log_patterns,
        predictions_dir=predictions_dir,
        time_range_query=(args.start, args.end),
        output_npy_path=args.output,
        anomaly_report=args.anomaly_report
    )
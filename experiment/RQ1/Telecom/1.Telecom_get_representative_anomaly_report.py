import os
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import argparse

# 北京时区（UTC+8）
BEIJING_TZ = timezone(timedelta(hours=8))

def ts_to_beijing_str(ts):
    """将 Unix 时间戳转换为北京时间字符串，格式：2021-03-04 00:41:00 CST"""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(BEIJING_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " CST"

def load_and_filter_npy(file_path, start_ts, end_ts):
    """加载 .npy 文件并过滤在 [start_ts, end_ts] 范围内的记录"""
    try:
        data = np.load(file_path, allow_pickle=True)
        # 过滤时间戳在区间内的记录
        filtered = [item for item in data if start_ts <= item[-1] <= end_ts]
        return filtered
    except Exception as e:
        print(f"⚠️  Warning: Failed to load {file_path}: {e}")
        return []

def group_log_anomalies(log_entries):
    """对 log 异常按 (pod, pattern_id, template) 分组"""
    groups = defaultdict(list)
    for entry in log_entries:
        pod, pattern_id, template, ts = entry
        key = (pod, pattern_id, template)
        groups[key].append(ts)
    return groups

def group_metric_A_anomalies(metric_entries):
    """对 metric_A 按 (entity, attribute) 分组"""
    groups = defaultdict(list)
    for entry in metric_entries:
        entity, attr, ts = entry
        key = (entity, attr)
        groups[key].append(ts)
    return groups

def group_metric_B_anomalies(metric_entries):
    """对 metric_B 按 (entity, attribute) 分组"""
    groups = defaultdict(list)
    for entry in metric_entries:
        entity, attr, ts = entry
        key = (entity, attr)
        groups[key].append(ts)
    return groups

def group_trace_anomalies(trace_entries):
    """对 trace 按 (edge, attribute) 分组"""
    groups = defaultdict(list)
    for entry in trace_entries:
        edge, attr, ts = entry
        # 标准化边格式：UNKNOWN_PARENT->IG01 → IG01（若需要可保留原样）
        key = (edge, attr)
        groups[key].append(ts)
    return groups


def write_metric_A_section(f, groups):
    f.write("📝 Detailed metric app Anomaly Report (Beijing Time = UTC+8):\n")
    f.write("=" * 80 + "\n\n")
    if not groups:
        f.write("No metric app anomalies found in the time window.\n\n")
        return

    entities = sorted(set(k[0] for k in groups.keys()))
    for ent in entities:
        f.write(f"Entity: {ent}\n")
        ent_groups = [(k, v) for k, v in groups.items() if k[0] == ent]
        ent_groups.sort(key=lambda x: x[0][1])  # sort by attribute
        for (entity, attr), timestamps in ent_groups:
            timestamps_sorted = sorted(timestamps)
            count = len(timestamps_sorted)
            f.write(f"  - Attribute '{attr}': {count} anomalies at timestamps:\n")
            time_strs = [f"{ts} ({ts_to_beijing_str(ts)})" for ts in timestamps_sorted]
            f.write("      " + ", ".join(time_strs) + "\n")
        f.write("\n")

def write_metric_B_section(f, groups):
    f.write("📝 Detailed metric container Anomaly Report (Beijing Time = UTC+8):\n")
    f.write("=" * 80 + "\n\n")
    if not groups:
        f.write("No metric container anomalies found in the time window.\n\n")
        return

    entities = sorted(set(k[0] for k in groups.keys()))
    for ent in entities:
        f.write(f"Entity: {ent}\n")
        ent_groups = [(k, v) for k, v in groups.items() if k[0] == ent]
        ent_groups.sort(key=lambda x: x[0][1])  # sort by attribute
        for (entity, attr), timestamps in ent_groups:
            timestamps_sorted = sorted(timestamps)
            count = len(timestamps_sorted)
            f.write(f"  - Attribute '{attr}': {count} anomalies at timestamps:\n")
            time_strs = [f"{ts} ({ts_to_beijing_str(ts)})" for ts in timestamps_sorted]
            f.write("      " + ", ".join(time_strs) + "\n")
        f.write("\n")

def write_trace_section(f, groups):
    f.write("📝 Detailed Trace Anomaly Report (Beijing Time = UTC+8):\n")
    f.write("=" * 80 + "\n\n")
    if not groups:
        f.write("No trace anomalies found in the time window.\n\n")
        return

    edges = sorted(set(k[0] for k in groups.keys()))
    for edge in edges:
        f.write(f"Edge: {edge}\n")
        edge_groups = [(k, v) for k, v in groups.items() if k[0] == edge]
        edge_groups.sort(key=lambda x: x[0][1])  # sort by attribute
        for (e, attr), timestamps in edge_groups:
            timestamps_sorted = sorted(timestamps)
            count = len(timestamps_sorted)
            f.write(f"  - Attribute '{attr}': {count} anomalies at timestamps:\n")
            time_strs = [f"{ts} ({ts_to_beijing_str(ts)})" for ts in timestamps_sorted]
            f.write("      " + ", ".join(time_strs) + "\n")
        f.write("\n")

def main(date_str, time_window_str, target_ts, output_file):
    
    base_dir = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/1216"
    start_ts = target_ts - 60
    end_ts = target_ts + 300

    # 构造文件名前缀
    prefix_metric_A = f"Telecom_metric_A_anomalies_{date_str}_{time_window_str}.npy"
    prefix_metric_B = f"Telecom_metric_B_anomalies_{date_str}_{time_window_str}.npy"
    prefix_trace = f"Telecom_trace_anomalies_{date_str}_{time_window_str}.npy"

    metric_A_path = os.path.join(base_dir, prefix_metric_A)
    metric_B_path = os.path.join(base_dir, prefix_metric_B)
    trace_path = os.path.join(base_dir, prefix_trace)

    # 加载并过滤数据
    metric_A_data = load_and_filter_npy(metric_A_path, start_ts, end_ts)
    metric_B_data = load_and_filter_npy(metric_B_path, start_ts, end_ts)
    trace_data = load_and_filter_npy(trace_path, start_ts, end_ts)

    # 分组
    metric_A_groups = group_metric_A_anomalies(metric_A_data)
    metric_B_groups = group_metric_B_anomalies(metric_B_data)
    trace_groups = group_trace_anomalies(trace_data)

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:        
        write_metric_A_section(f, metric_A_groups)
        write_metric_B_section(f, metric_B_groups)
        write_trace_section(f, trace_groups)

        f.write("💡 Note: 'CST' = China Standard Time (UTC+8).\n")

    print(f"✅ Report written to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anomaly report for Telecom dataset.")
    parser.add_argument("--date", required=True, help="Date string like 2021_03_04")
    parser.add_argument("--window", required=True, help="Time window like 0030_0100")
    parser.add_argument("--ts", type=int, required=True, help="Target Unix timestamp")
    parser.add_argument("--output", default="anomaly_report.txt", help="Output file path")

    args = parser.parse_args()
    main(args.date, args.window, args.ts, args.output)
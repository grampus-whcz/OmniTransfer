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

def group_metric_app_anomalies(metric_entries):
    """对 metric_app 按 (entity, attribute) 分组"""
    groups = defaultdict(list)
    for entry in metric_entries:
        entity, attr, ts = entry
        key = (entity, attr)
        groups[key].append(ts)
    return groups

def group_metric_container_anomalies(metric_entries):
    """对 metric_container 按 (entity, attribute) 分组"""
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

# def write_log_section(f, groups):
#     f.write("📝 Detailed Log Anomaly Report (Beijing Time = UTC+8):\n")
#     f.write("=" * 80 + "\n\n")
#     if not groups:
#         f.write("No log anomalies found in the time window.\n\n")
#         return

#     # 按 Pod 排序
#     pods = sorted(set(key[0] for key in groups.keys()))
#     for pod in pods:
#         f.write(f"Pod: {pod}\n")
#         pod_groups = [(k, v) for k, v in groups.items() if k[0] == pod]
#         # 按 pattern_id 排序
#         pod_groups.sort(key=lambda x: x[0][1])
#         for (pod_, pid, template), timestamps in pod_groups:
#             timestamps_sorted = sorted(timestamps)
#             count = len(timestamps_sorted)
#             f.write(f"  - Pattern ID {pid} ({count} anomalies):\n")
#             f.write(f"      Template: {template}\n")
#             time_strs = [f"{ts} ({ts_to_beijing_str(ts)})" for ts in timestamps_sorted]
#             for tstr in time_strs:
#                 f.write(f"      {tstr}\n")
#         f.write("\n")

def write_metric_app_section(f, groups):
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

def write_metric_container_section(f, groups):
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
        
        
import json

# 在 main 函数外或顶部定义全局模板变量（或作为参数传递）
LOG_TEMPLATES = []

def load_log_templates():
    """加载 log 模板 JSON 文件"""
    template_path = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/log/temp_data/analysis/log/log_istio_patterns.json"
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 只取 "Java" 键对应的模板列表
            templates = data.get("Java", [])
            return templates
    except Exception as e:
        print(f"⚠️ Warning: Failed to load log templates from {template_path}: {e}")
        return []

# 修改 write_log_section：接收 templates 列表作为参数
def write_log_section(f, groups, templates):
    f.write("📝 Detailed Log Anomaly Report (Beijing Time = UTC+8):\n")
    f.write("=" * 80 + "\n\n")
    if not groups:
        f.write("No log anomalies found in the time window.\n\n")
        return

    # 按 Pod 排序
    pods = sorted(set(key[0] for key in groups.keys()))
    for pod in pods:
        f.write(f"Pod: {pod}\n")
        pod_groups = [(k, v) for k, v in groups.items() if k[0] == pod]
        # 按 pattern_id 排序
        pod_groups.sort(key=lambda x: x[0][1])
        for (pod_, pid, _old_template), timestamps in pod_groups:
            timestamps_sorted = sorted(timestamps)
            count = len(timestamps_sorted)

            # 获取新模板：安全索引
            if 0 <= pid < len(templates):
                real_template = templates[pid].strip()
                if not real_template:  # 如果模板是空字符串
                    real_template = "<Empty Template>"
            else:
                real_template = f"<Invalid Pattern ID: {pid}>"

            f.write(f"  - Pattern ID {pid} ({count} anomalies):\n")
            f.write(f"      Template: {real_template}\n")
            time_strs = [f"{ts} ({ts_to_beijing_str(ts)})" for ts in timestamps_sorted]
            for tstr in time_strs:
                f.write(f"      {tstr}\n")
        f.write("\n")

def main(date_str, time_window_str, target_ts, output_file):
    # 加载日志模板（只需加载一次）
    LOG_TEMPLATES = load_log_templates()
    
    base_dir = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Bank_time_line"
    start_ts = target_ts - 60
    end_ts = target_ts + 300

    # 构造文件名前缀
    prefix_log = f"Bank_log_anomalies_{date_str}_{time_window_str}.npy"
    prefix_metric_app = f"Bank_metric_app_anomalies_{date_str}_{time_window_str}.npy"
    prefix_metric_container = f"Bank_metric_container_anomalies_{date_str}_{time_window_str}.npy"
    prefix_trace = f"Bank_trace_anomalies_{date_str}_{time_window_str}.npy"

    log_path = os.path.join(base_dir, prefix_log)
    metric_app_path = os.path.join(base_dir, prefix_metric_app)
    metric_container_path = os.path.join(base_dir, prefix_metric_container)
    trace_path = os.path.join(base_dir, prefix_trace)

    # 加载并过滤数据
    log_data = load_and_filter_npy(log_path, start_ts, end_ts)
    metric_app_data = load_and_filter_npy(metric_app_path, start_ts, end_ts)
    metric_container_data = load_and_filter_npy(metric_container_path, start_ts, end_ts)
    trace_data = load_and_filter_npy(trace_path, start_ts, end_ts)

    # 分组
    log_groups = group_log_anomalies(log_data)
    metric_app_groups = group_metric_app_anomalies(metric_app_data)
    metric_container_groups = group_metric_container_anomalies(metric_container_data)
    trace_groups = group_trace_anomalies(trace_data)

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:        
        write_metric_app_section(f, metric_app_groups)
        write_metric_container_section(f, metric_container_groups)
        write_trace_section(f, trace_groups)
        write_log_section(f, log_groups, LOG_TEMPLATES)

        f.write("💡 Note: 'CST' = China Standard Time (UTC+8).\n")

    print(f"✅ Report written to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anomaly report for Bank dataset.")
    parser.add_argument("--date", required=True, help="Date string like 2021_03_04")
    parser.add_argument("--window", required=True, help="Time window like 0030_0100")
    parser.add_argument("--ts", type=int, required=True, help="Target Unix timestamp")
    parser.add_argument("--output", default="anomaly_report.txt", help="Output file path")

    args = parser.parse_args()
    main(args.date, args.window, args.ts, args.output)
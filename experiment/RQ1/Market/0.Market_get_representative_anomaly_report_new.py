import os
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import argparse
import json  # <-- 新增：用于加载 log 模板

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
        filtered = [item for item in data if start_ts <= item[-1] <= end_ts]
        return filtered
    except Exception as e:
        print(f"⚠️ Warning: Failed to load {file_path}: {e}")
        return []

def group_anomalies(entries, is_log=False):
    """对 Market 异常按 (entity, attribute) 或 (pod, pattern_id, template) 分组"""
    groups = defaultdict(list)
    for entry in entries:
        if is_log:
            pod, pattern_id, template, ts = entry
            key = (pod, pattern_id, template)
        else:
            entity, attr, ts = entry
            key = (entity, attr)
        groups[key].append(ts)
    return groups

def write_section(f, groups, section_title, is_log=False, templates=None):
    f.write(f"{section_title}\n")
    f.write("=" * 80 + "\n\n")
    
    # 过滤出异常次数 >= 2 的分组
    frequent_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    
    if not frequent_groups:
        f.write("No frequent anomalies (≥2 occurrences) found in the time window.\n\n")
        return

    if is_log:
        # 按 Pod 分组展示
        pods = sorted(set(key[0] for key in frequent_groups.keys()))
        for pod in pods:
            f.write(f"Pod: {pod}\n")
            pod_groups = [(k, v) for k, v in frequent_groups.items() if k[0] == pod]
            pod_groups.sort(key=lambda x: x[0][1])  # 按 pattern_id 排序
            for (pod_, pid_, _old_template), timestamps in pod_groups:
                if 0 <= pid_ < len(templates):
                    real_template = templates[pid_].strip()
                    if not real_template:
                        real_template = "<Empty Template>"
                else:
                    real_template = f"<Invalid Pattern ID: {pid_}>"
                timestamps_sorted = sorted(timestamps)
                count = len(timestamps_sorted)
                f.write(f"  - Pattern ID {pid_} ({count} anomalies):\n")
                f.write(f"      Template: {real_template}\n")
                time_strs = [f"{ts} ({ts_to_beijing_str(ts)})" for ts in timestamps_sorted]
                f.write("      " + ", ".join(time_strs) + "\n")
            f.write("\n")
    else:
        # 非 log：按 (entity, attr) 排序
        sorted_items = sorted(frequent_groups.items(), key=lambda x: (x[0][0], x[0][1]))
        for (entity, attr), timestamps in sorted_items:
            timestamps_sorted = sorted(timestamps)
            count = len(timestamps_sorted)
            f.write(f"Entity: {entity}, Attribute: {attr} ({count} anomalies)\n")
            time_strs = [f"{ts} ({ts_to_beijing_str(ts)})" for ts in timestamps_sorted]
            f.write("      " + ", ".join(time_strs) + "\n")
        f.write("\n")

def load_log_templates():
    """加载 log 模板 JSON 文件"""
    template_path = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Market/log/temp_data/analysis/log/log_istio_patterns.json"
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            templates = data.get("Java", [])
            return templates
    except Exception as e:
        print(f"⚠️ Warning: Failed to load log templates from {template_path}: {e}")
        return []

def main(date_str, time_window_str, target_ts, output_file):
    base_dir = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/1215"
    start_ts = target_ts - 60
    end_ts = target_ts + 300

    full_window = "0000_2400"
    prefix_log = f"Market_log_anomalies_{date_str}_{full_window}.npy"
    prefix_metric_service = f"Market_metric_service_anomalies_{date_str}_{full_window}.npy"
    prefix_metric_runtime = f"Market_metric_runtime_anomalies_{date_str}_{full_window}.npy"
    prefix_metric_mesh = f"Market_metric_mesh_anomalies_{date_str}_{full_window}.npy"
    prefix_metric_container = f"Market_metric_container_anomalies_{date_str}_{full_window}.npy"
    prefix_metric_node = f"Market_metric_node_anomalies_{date_str}_{full_window}.npy"
    prefix_trace = f"Market_trace_anomalies_{date_str}_{full_window}.npy"

    log_path = os.path.join(base_dir, prefix_log)
    metric_service_path = os.path.join(base_dir, prefix_metric_service)
    metric_runtime_path = os.path.join(base_dir, prefix_metric_runtime)
    metric_mesh_path = os.path.join(base_dir, prefix_metric_mesh)
    metric_container_path = os.path.join(base_dir, prefix_metric_container)
    metric_node_path = os.path.join(base_dir, prefix_metric_node)
    trace_path = os.path.join(base_dir, prefix_trace)

    log_data = load_and_filter_npy(log_path, start_ts, end_ts)
    metric_service_data = load_and_filter_npy(metric_service_path, start_ts, end_ts)
    metric_runtime_data = load_and_filter_npy(metric_runtime_path, start_ts, end_ts)
    metric_mesh_data = load_and_filter_npy(metric_mesh_path, start_ts, end_ts)
    metric_container_data = load_and_filter_npy(metric_container_path, start_ts, end_ts)
    metric_node_data = load_and_filter_npy(metric_node_path, start_ts, end_ts)
    trace_data = load_and_filter_npy(trace_path, start_ts, end_ts)

    log_groups = group_anomalies(log_data, is_log=True)
    metric_service_groups = group_anomalies(metric_service_data)
    metric_runtime_groups = group_anomalies(metric_runtime_data)
    metric_mesh_groups = group_anomalies(metric_mesh_data)
    metric_container_groups = group_anomalies(metric_container_data)
    metric_node_groups = group_anomalies(metric_node_data)
    trace_groups = group_anomalies(trace_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        write_section(f, metric_service_groups, "📝 Detailed Metric Service Anomaly Report (Beijing Time = UTC+8):", is_log=False)
        write_section(f, metric_runtime_groups, "📝 Detailed Metric Runtime Anomaly Report (Beijing Time = UTC+8):", is_log=False)
        write_section(f, metric_mesh_groups, "📝 Detailed Metric Mesh Anomaly Report (Beijing Time = UTC+8):", is_log=False)
        write_section(f, metric_container_groups, "📝 Detailed Metric Container Anomaly Report (Beijing Time = UTC+8):", is_log=False)
        write_section(f, metric_node_groups, "📝 Detailed Metric Node Anomaly Report (Beijing Time = UTC+8):", is_log=False)
        write_section(f, trace_groups, "📝 Detailed Trace Anomaly Report (Beijing Time = UTC+8):", is_log=False)
        write_section(f, log_groups, "📝 Detailed Log Anomaly Report (Beijing Time = UTC+8):", is_log=True, templates=load_log_templates())

        f.write("💡 Note: 'CST' = China Standard Time (UTC+8). Only anomalies with ≥2 occurrences are reported.\n")

    print(f"✅ Report (only ≥2 anomalies) written to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anomaly report for Market dataset (only frequent anomalies ≥2).")
    parser.add_argument("--date", required=True, help="Date string like 2021_03_04")
    parser.add_argument("--window", required=True, help="Time window like 0030_0100")
    parser.add_argument("--ts", type=int, required=True, help="Target Unix timestamp")
    parser.add_argument("--output", default="market_anomaly_report.txt", help="Output file path")

    args = parser.parse_args()
    main(args.date, args.window, args.ts, args.output)
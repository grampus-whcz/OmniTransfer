import os
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import argparse

# 北京时区 (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))

def ts_to_beijing_str(ts):
    """将 Unix 时间戳转为北京时间字符串"""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(BEIJING_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " CST"

def load_anomalies_from_window(base_dir, date_str, window_str):
    """仅加载指定日期和窗口的4类异常文件，并去重"""
    anomalies = []

    file_specs = [
        ("metric_app", f"Bank_metric_app_anomalies_{date_str}_{window_str}.npy"),
        ("metric_container", f"Bank_metric_container_anomalies_{date_str}_{window_str}.npy"),
        ("trace", f"Bank_trace_anomalies_{date_str}_{window_str}.npy"),
        ("log", f"Bank_log_anomalies_{date_str}_{window_str}.npy")
    ]

    for typ, filename in file_specs:
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            continue

        try:
            data = np.load(filepath, allow_pickle=True)
            print(f"✅ Loaded {len(data)} anomalies from {filename}")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            continue

        for item in data:
            if typ == "log":
                pod, pattern_id, template, ts = item
                anomalies.append({
                    'ts': int(ts),
                    'type': 'log',
                    'entity': str(pod),
                    'attribute': f"PatternID_{pattern_id}",
                    'raw': str(template)
                })
            elif typ == "metric_app":
                service, attr, ts = item
                anomalies.append({
                    'ts': int(ts),
                    'type': 'metric_app',
                    'entity': str(service),
                    'attribute': str(attr),
                    'raw': ''
                })
            elif typ == "metric_container":
                pod, attr, ts = item
                anomalies.append({
                    'ts': int(ts),
                    'type': 'metric_container',
                    'entity': str(pod),
                    'attribute': str(attr),
                    'raw': ''
                })
            elif typ == "trace":
                edge, attr, ts = item
                anomalies.append({
                    'ts': int(ts),
                    'type': 'trace',
                    'entity': str(edge),
                    'attribute': str(attr),
                    'raw': ''
                })

    # 🔥 关键：去重！避免同一 (type, entity, attr, ts) 被重复计算
    seen = set()
    unique_anomalies = []
    for a in anomalies:
        key = (a['type'], a['entity'], a['attribute'], a['ts'])
        if key not in seen:
            seen.add(key)
            unique_anomalies.append(a)

    unique_anomalies.sort(key=lambda x: x['ts'])
    print(f"🧹 After deduplication: {len(unique_anomalies)} unique anomalies (was {len(anomalies)})")
    return unique_anomalies

def extract_keywords(template):
    """从 log 模板中提取关键故障词"""
    keywords = set()
    t_low = template.lower()
    if any(kw in t_low for kw in ['out of memory', 'oom', 'java.lang.outofmemoryerror']):
        keywords.add("OOM")
    if 'gc' in t_low and ('allocation failure' in t_low or 'full gc' in t_low):
        keywords.add("GC")
    if 'error' in t_low or 'exception' in t_low or 'fail' in t_low:
        keywords.add("Error/Failure")
    if 'timeout' in t_low:
        keywords.add("Timeout")
    return sorted(keywords)

def generate_full_report(anomalies, output_file):
    """不再DBSCAN聚类，全部异常作为一个整体输出，复用原有排版格式"""
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to generate report.")
        return

    # 全局时间范围
    ts_vals = [a['ts'] for a in anomalies]
    start_ts, end_ts = min(ts_vals), max(ts_vals)
    duration = end_ts - start_ts

    # 全局日志关键词汇总
    all_keywords = set()
    for a in anomalies:
        if a['type'] == 'log':
            all_keywords.update(extract_keywords(a['raw']))

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 写入完整报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Full Anomaly Aggregation Report for {args.date_online} {args.output_suffix}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"🚨 Global Anomaly Collection (All anomalies as one unified set)\n")
        f.write(f"   Time Span: {ts_to_beijing_str(start_ts)} → {ts_to_beijing_str(end_ts)} "
                f"(Δ = {duration} sec)\n")
        f.write(f"   Total Unique Anomalies: {len(anomalies)}\n")

        if all_keywords:
            f.write(f"   🔑 Global Log Keywords Summary: {', '.join(all_keywords)}\n")
        f.write("\n")

        # 按模态分组展示，沿用原有实体-指标聚合逻辑
        grouped = defaultdict(list)
        for a in anomalies:
            grouped[a['type']].append(a)

        type_order = ['metric_app', 'metric_container', 'trace', 'log']
        for typ in type_order:
            if typ not in grouped:
                continue
            f.write(f"   📝 {typ.replace('_', ' ').title()} Anomalies:\n")
            # 按 (entity, attribute) 聚合时间戳
            entity_attr_dict = defaultdict(list)
            for a in grouped[typ]:
                key = (a['entity'], a['attribute'])
                entity_attr_dict[key].append(a['ts'])

            for (ent, attr), timestamps in sorted(entity_attr_dict.items()):
                ts_sorted = sorted(timestamps)
                time_repr = ", ".join(f"{ts} ({ts_to_beijing_str(ts)})" for ts in ts_sorted)
                f.write(f"     • Entity: {ent} | Attribute: {attr}\n")
                f.write(f"       Timestamps: {time_repr}\n")
            f.write("\n")

        f.write("-" * 60 + "\n\n")
        f.write("💡 Note: 'CST' = China Standard Time (UTC+8).\n")
        f.write("   Processing Mode: No DBSCAN clustering, all anomalies aggregated as one global collection.\n")

    print(f"✅ Full aggregation report saved to: {output_file}")
    print(f"📊 Total unique anomalies collected: {len(anomalies)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate all multi-modal anomalies into one unified report without DBSCAN clustering.")
    parser.add_argument("--date_online", required=True, help="Date string like 2021_03_04")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0030_0100")
    # 保留eps/min_samples参数兼容原有run.sh，实际不再使用
    parser.add_argument("--eps", type=int, default=60, help="(Deprecated, no clustering used) DBSCAN eps in seconds")
    parser.add_argument("--min_samples", type=int, default=3, help="(Deprecated, no clustering used) DBSCAN min_samples")
    parser.add_argument("--output_folder_name", type=str, default="1116",
                        help="Output folder name passed to run.sh (e.g., experiment ID)")

    args = parser.parse_args()

    BASE_DIR = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"

    print(f"📁 Loading anomalies for date={args.date_online}, window={args.output_suffix}")
    anomalies = load_anomalies_from_window(BASE_DIR, args.date_online, args.output_suffix)
    
    output_file = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}/Bank_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}.txt"

    print(f"🎯 Total unique anomalies loaded: {len(anomalies)}")
    generate_full_report(anomalies, output_file)
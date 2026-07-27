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

def equal_width_segment_cluster(anomalies, output_file, seg_width_seconds=300):
    """
    Equal-Width Segmentation 等宽时序分段聚类
    :param anomalies: 去重排序后的异常列表
    :param output_file: 报告输出路径
    :param seg_width_seconds: 分段宽度，单位秒，可通过命令行调节
    """
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to segment.")
        return

    # 1. 获取全局时间边界
    ts_list = [a['ts'] for a in anomalies]
    global_min_ts = min(ts_list)
    global_max_ts = max(ts_list)
    seg_width = seg_width_seconds

    # 2. 等宽切分所有区间
    seg_clusters = defaultdict(list)
    # 计算每个异常归属的分段ID
    for anomaly in anomalies:
        ts = anomaly['ts']
        seg_id = (ts - global_min_ts) // seg_width
        seg_clusters[seg_id].append(anomaly)

    sorted_seg_ids = sorted(seg_clusters.keys())
    seg_count = len(sorted_seg_ids)

    # 3. 输出报告，格式对齐DBSCAN版本
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Equal-Width Segmentation Anomaly Report for {args.date_online} {args.output_suffix}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"🔍 Equal-Width Segment Width: {seg_width} seconds (theoretical fixed window length)\n")
        f.write(f"🔍 Total Segments Generated: {seg_count}\n")
        f.write("=" * 40 + "\n\n")

        for idx, seg_id in enumerate(sorted_seg_ids):
            seg_anoms = seg_clusters[seg_id]
            seg_ts = [a['ts'] for a in seg_anoms]
            seg_start_ts = min(seg_ts)
            seg_end_ts = max(seg_ts)
            seg_duration = seg_end_ts - seg_start_ts

            # 新增：计算当前分段理论左右边界
            seg_theo_start = global_min_ts + seg_id * seg_width
            seg_theo_end = seg_theo_start + seg_width

            f.write(f"🚨 Segment #{idx + 1} (Seg ID = {seg_id})\n")
            f.write(f"   Theoretical Fixed Window Boundary: {ts_to_beijing_str(seg_theo_start)} → {ts_to_beijing_str(seg_theo_end)} (fixed {seg_width}s)\n")
            f.write(f"   Actual Anomaly Time Span in This Segment: {ts_to_beijing_str(seg_start_ts)} → {ts_to_beijing_str(seg_end_ts)} "
                    f"(Δ = {seg_duration} sec, only covers part of theoretical window)\n")
            f.write(f"   Total Anomalies in Segment: {len(seg_anoms)}\n")

            # 汇总当前分段日志关键词
            seg_keywords = set()
            for a in seg_anoms:
                if a['type'] == 'log':
                    seg_keywords.update(extract_keywords(a['raw']))
            if seg_keywords:
                f.write(f"   🔑 Segment Log Keywords: {', '.join(seg_keywords)}\n")
            f.write("\n")

            # 按模态分组展示异常
            grouped = defaultdict(list)
            for a in seg_anoms:
                grouped[a['type']].append(a)

            type_order = ['metric_app', 'metric_container', 'trace', 'log']
            for typ in type_order:
                if typ not in grouped:
                    continue
                f.write(f"   📝 {typ.replace('_', ' ').title()} Anomalies:\n")
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
        f.write(f"   Segmentation Method: Equal-Width Uniform Time Segmentation\n")
        f.write(f"   Fixed segment width parameter: seg_width_seconds = {seg_width_seconds}\n")
        f.write("   Two time ranges are displayed for each segment:\n")
        f.write("     1. Theoretical fixed window: the full 300s interval assigned by segmentation algorithm;\n")
        f.write("     2. Actual anomaly span: only the time range covered by real anomalies inside this window.\n")

    print(f"✅ Equal-Width segmentation report saved to: {output_file}")
    print(f"📊 Generated {seg_count} equal-width time segments.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Equal-Width Uniform Time Segmentation for multi-modal anomaly grouping.")
    parser.add_argument("--date_online", required=True, help="Date string like 2021_03_04")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0030_0100")
    # 等宽分段宽度，单位秒，消融时可调
    parser.add_argument("--seg_width", type=int, default=300, help="Equal-width segment width in seconds (default: 300s = 5min)")
    # 兼容原有run.sh传参，废弃参数
    parser.add_argument("--eps", type=int, default=60, help="(Deprecated for equal-width) DBSCAN eps seconds")
    parser.add_argument("--min_samples", type=int, default=3, help="(Deprecated for equal-width) DBSCAN min_samples")
    parser.add_argument("--output_folder_name", type=str, default="1116",
                        help="Output folder name passed to run.sh (e.g., experiment ID)")

    args = parser.parse_args()

    BASE_DIR = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"

    print(f"📁 Loading anomalies for date={args.date_online}, window={args.output_suffix}")
    anomalies = load_anomalies_from_window(BASE_DIR, args.date_online, args.output_suffix)
    
    output_file = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}/Bank_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}.txt"

    print(f"🎯 Total unique anomalies loaded: {len(anomalies)}")
    equal_width_segment_cluster(
        anomalies,
        output_file=output_file,
        seg_width_seconds=args.seg_width
    )
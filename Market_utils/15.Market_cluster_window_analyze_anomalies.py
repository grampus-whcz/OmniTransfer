import os
import numpy as np
from sklearn.cluster import DBSCAN
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
    """加载 Market 的所有7类异常文件，并去重，且仅保留 window_str 指定时间段内的异常"""
    anomalies = []

    # === 新增：解析 window_str 为时间范围（格式: HHMM_HHMM，例如 0930_1000）===
    try:
        parts = window_str.split('_')
        if len(parts) != 2:
            raise ValueError(f"Expected format HHMM_HHMM, got '{window_str}'")

        start_hm, end_hm = parts[0], parts[1]

        # 验证是否为4位数字
        if not (len(start_hm) == 4 and start_hm.isdigit() and len(end_hm) == 4 and end_hm.isdigit()):
            raise ValueError(f"Time parts must be 4-digit numbers (HHMM), got '{start_hm}' and '{end_hm}'")

        # 构造北京时间 datetime
        date_part = datetime.strptime(date_str, "%Y_%m_%d")  # naive datetime
        start_hour, start_min = int(start_hm[:2]), int(start_hm[2:])
        end_hour, end_min = int(end_hm[:2]), int(end_hm[2:])

        start_dt = date_part.replace(hour=start_hour, minute=start_min, second=0, microsecond=0, tzinfo=BEIJING_TZ)
        end_dt = date_part.replace(hour=end_hour, minute=end_min, second=0, microsecond=0, tzinfo=BEIJING_TZ)

        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())  # [start_ts, end_ts)
    except Exception as e:
        print(f"⚠️ Failed to parse window '{window_str}': {e}. Skipping time filtering.")
        start_ts, end_ts = None, None

    # Market 特有的异常类型
    file_specs = [
        ("metric_service", f"Market_metric_service_anomalies_{date_str}_{window_str}.npy"),
        ("metric_runtime", f"Market_metric_runtime_anomalies_{date_str}_{window_str}.npy"),
        ("metric_container", f"Market_metric_container_anomalies_{date_str}_{window_str}.npy"),
        ("metric_mesh", f"Market_metric_mesh_anomalies_{date_str}_{window_str}.npy"),
        ("metric_node", f"Market_metric_node_anomalies_{date_str}_{window_str}.npy"),
        ("trace", f"Market_trace_anomalies_{date_str}_{window_str}.npy"),
        ("log", f"Market_log_anomalies_{date_str}_{window_str}.npy")
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
                ts = int(ts)
            else:
                entity, attr, ts = item
                ts = int(ts)

            # === 新增：时间范围过滤 ===
            if start_ts is not None and (ts < start_ts or ts >= end_ts):
                continue  # 跳过不在 [start_ts, end_ts) 区间内的异常

            if typ == "log":
                anomalies.append({
                    'ts': ts,
                    'type': 'log',
                    'entity': str(pod),
                    'attribute': f"PatternID_{pattern_id}",
                    'raw': str(template)
                })
            else:
                anomalies.append({
                    'ts': ts,
                    'type': typ,
                    'entity': str(entity),
                    'attribute': str(attr),
                    'raw': ''
                })

    # 去重
    seen = set()
    unique_anomalies = []
    for a in anomalies:
        key = (a['type'], a['entity'], a['attribute'], a['ts'])
        if key not in seen:
            seen.add(key)
            unique_anomalies.append(a)

    unique_anomalies.sort(key=lambda x: x['ts'])
    print(f"🧹 After deduplication and time filtering: {len(unique_anomalies)} unique anomalies "
          f"(original loaded: {len(anomalies)})")
    return unique_anomalies

def extract_keywords(template):
    """从 log 模板中提取关键故障词（与 Bank 一致）"""
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

def cluster_and_report(anomalies, output_file, eps_seconds=300, min_samples=2):
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to cluster.")
        return

    X = np.array([[a['ts']] for a in anomalies])
    labels = DBSCAN(eps=eps_seconds, min_samples=min_samples, metric='euclidean').fit_predict(X)

    clusters = defaultdict(list)
    noise = []
    for anomaly, label in zip(anomalies, labels):
        if label == -1:
            noise.append(anomaly)
        else:
            clusters[label].append(anomaly)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Market Anomaly Clustering Report for {args.date_online} {args.output_suffix}\n")
        f.write("=" * 80 + "\n\n")

        cluster_ids = sorted(clusters.keys())
        f.write(f"🔍 The number of clusters are {len(cluster_ids)}\n")
        f.write("=" * 40 + "\n\n")

        for idx, cid in enumerate(cluster_ids):
            cluster = clusters[cid]
            ts_vals = [a['ts'] for a in cluster]
            start_ts, end_ts = min(ts_vals), max(ts_vals)
            duration = end_ts - start_ts

            f.write(f"🚨 Cluster #{idx + 1}\n")
            f.write(f"   Time Span: {ts_to_beijing_str(start_ts)} → {ts_to_beijing_str(end_ts)} "
                    f"(Δ = {duration} sec)\n")
            f.write(f"   Total Anomalies: {len(cluster)}\n")

            all_keywords = set()
            for a in cluster:
                if a['type'] == 'log':
                    all_keywords.update(extract_keywords(a['raw']))
            if all_keywords:
                f.write(f"   🔑 Keywords: {', '.join(all_keywords)}\n")
            f.write("\n")

            grouped = defaultdict(list)
            for a in cluster:
                grouped[a['type']].append(a)

            # 按重要性或逻辑顺序排列
            type_order = [
                'metric_service', 'metric_runtime', 'metric_container',
                'metric_mesh', 'metric_node', 'trace', 'log'
            ]
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

        if noise:
            f.write("🔕 Isolated Anomalies (Noise / Single Events):\n")
            for a in sorted(noise, key=lambda x: x['ts']):
                f.write(f"   {a['type']} | {a['entity']} | {a['attribute']} | "
                        f"{a['ts']} ({ts_to_beijing_str(a['ts'])})\n")
            f.write("\n")

        f.write("💡 Note: 'CST' = China Standard Time (UTC+8).\n")
        f.write(f"   Clustering: DBSCAN(eps={eps_seconds}s, min_samples={min_samples})\n")

    print(f"✅ Report saved to: {output_file}")
    print(f"📊 Found {len(cluster_ids)} clusters and {len(noise)} isolated anomalies.")
    
def cluster_and_report_condensed(anomalies, output_file, eps_seconds=300, min_samples=2):
    """生成精简版报告：仅包含同一 (entity, attribute) 在窗口内出现多次的异常"""
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to generate condensed report.")
        return

    # 先按 DBSCAN 聚类（保持与原报告一致的聚类结果）
    X = np.array([[a['ts']] for a in anomalies])
    labels = DBSCAN(eps=eps_seconds, min_samples=min_samples, metric='euclidean').fit_predict(X)

    clusters = defaultdict(list)
    for anomaly, label in zip(anomalies, labels):
        if label != -1:  # 只考虑聚类内的点，忽略 noise（可选，也可保留 noise 中的重复项）
            clusters[label].append(anomaly)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Market Condensed Anomaly Report (Repeated Entity-Attribute Only)\n")
        f.write(f"   For {args.date_online} {args.output_suffix}\n")
        f.write("=" * 80 + "\n\n")

        total_condensed_entries = 0
        cluster_ids = sorted(clusters.keys())

        for idx, cid in enumerate(cluster_ids):
            cluster = clusters[cid]
            ts_vals = [a['ts'] for a in cluster]
            start_ts, end_ts = min(ts_vals), max(ts_vals)
            duration = end_ts - start_ts

            # 按 (type, entity, attribute) 分组，统计时间戳数量
            grouped = defaultdict(list)
            for a in cluster:
                key = (a['type'], a['entity'], a['attribute'])
                grouped[key].append(a['ts'])

            # 筛选出时间戳 ≥2 的项
            repeated_items = {
                k: sorted(set(v)) for k, v in grouped.items() if len(set(v)) >= 2
            }

            if not repeated_items:
                continue

            f.write(f"🚨 Cluster #{idx + 1}\n")
            f.write(f"   Time Span: {ts_to_beijing_str(start_ts)} → {ts_to_beijing_str(end_ts)} "
                    f"(Δ = {duration} sec)\n")
            f.write(f"   Repeated Entity-Attribute Pairs: {len(repeated_items)}\n\n")

            # 按类型分组输出
            type_groups = defaultdict(list)
            for (typ, ent, attr), tss in repeated_items.items():
                type_groups[typ].append((ent, attr, tss))

            type_order = [
                'metric_service', 'metric_runtime', 'metric_container',
                'metric_mesh', 'metric_node', 'trace', 'log'
            ]
            for typ in type_order:
                if typ not in type_groups:
                    continue
                f.write(f"   📝 {typ.replace('_', ' ').title()} Anomalies:\n")
                for ent, attr, tss in sorted(type_groups[typ]):
                    time_repr = ", ".join(f"{ts_to_beijing_str(ts)}" for ts in tss)
                    f.write(f"     • Entity: {ent} | Attribute: {attr}\n")
                    f.write(f"       Times ({len(tss)}): {time_repr}\n")
                    total_condensed_entries += 1
                f.write("\n")

            f.write("-" * 60 + "\n\n")

        if total_condensed_entries == 0:
            f.write("✅ No repeated (entity, attribute) anomalies found.\n")
        else:
            f.write(f"💡 Total repeated entity-attribute entries: {total_condensed_entries}\n")

        f.write("💡 Note: Only entries with ≥2 distinct timestamps are shown.\n")
        f.write(f"   Clustering: DBSCAN(eps={eps_seconds}s, min_samples={min_samples})\n")

    print(f"✅ Condensed report saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster anomalies in a specific half-hour window of Market dataset.")
    parser.add_argument("--date_online", required=True, help="Date string like 2022_03_20")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0830_to_0900")
    parser.add_argument("--eps", type=int, default=60, help="DBSCAN eps in seconds (default: 60)")
    parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples (default: 2)")
    parser.add_argument("--output_folder_name", type=str, default="1215",
                        help="Output folder name (e.g., experiment ID)")

    args = parser.parse_args()

    BASE_DIR = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"
    print(f"📁 Loading Market anomalies for date={args.date_online}, window={args.output_suffix}")

    anomalies = load_anomalies_from_window(BASE_DIR, args.date_online, args.output_suffix)
    output_file = f"{BASE_DIR}/Market_cluster_window_anomaly_full_report_{args.date_online}_{args.output_suffix}.txt"

    print(f"🎯 Total anomalies loaded: {len(anomalies)}")
    cluster_and_report(
        anomalies,
        output_file=output_file,
        eps_seconds=args.eps,
        min_samples=args.min_samples
    )
    
        # 生成精简版报告
    condensed_output_file = f"{BASE_DIR}/Market_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}.txt"
    cluster_and_report_condensed(
        anomalies,
        output_file=condensed_output_file,
        eps_seconds=args.eps,
        min_samples=args.min_samples
    )
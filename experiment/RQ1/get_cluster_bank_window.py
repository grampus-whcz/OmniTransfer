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
    """仅加载指定日期和窗口的4类异常文件，并去重"""
    anomalies = []

    file_specs = [
        ("log", f"Bank_log_anomalies_{date_str}_{window_str}.npy"),
        ("metric_app", f"Bank_metric_app_anomalies_{date_str}_{window_str}.npy"),
        ("metric_container", f"Bank_metric_container_anomalies_{date_str}_{window_str}.npy"),
        ("trace", f"Bank_trace_anomalies_{date_str}_{window_str}.npy")
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

def cluster_and_report(anomalies, output_file, eps_seconds=300, min_samples=2):
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to cluster.")
        return

    # 提取时间戳用于聚类（一维）
    X = np.array([[a['ts']] for a in anomalies])
    
    # ✅ 正确使用 DBSCAN：调用 fit_predict
    labels = DBSCAN(
        eps=eps_seconds,
        min_samples=min_samples,
        metric='euclidean'
    ).fit_predict(X)

    # 分组
    clusters = defaultdict(list)
    noise = []
    for anomaly, label in zip(anomalies, labels):
        if label == -1:
            noise.append(anomaly)
        else:
            clusters[label].append(anomaly)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 写入报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Anomaly Clustering Report for {args.date} {args.window}\n")
        f.write(f"🔍 The number of clusters are {args.date} {args.window}\n")
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

            # 关键词（仅 log）
            all_keywords = set()
            for a in cluster:
                if a['type'] == 'log':
                    all_keywords.update(extract_keywords(a['raw']))
            if all_keywords:
                f.write(f"   🔑 Keywords: {', '.join(all_keywords)}\n")
            f.write("\n")

            # 按类型分组展示
            grouped = defaultdict(list)
            for a in cluster:
                grouped[a['type']].append(a)

            type_order = ['log', 'metric_app', 'metric_container', 'trace']
            for typ in type_order:
                if typ not in grouped:
                    continue
                f.write(f"   📝 {typ.replace('_', ' ').title()} Anomalies:\n")
                # 按 (entity, attribute) 聚合
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster anomalies in a specific half-hour window of Bank dataset.")
    parser.add_argument("--date", required=True, help="Date string like 2021_03_04")
    parser.add_argument("--window", required=True, help="Time window like 0030_0100")
    parser.add_argument("--output", default="cluster_report.txt", help="Output report file")
    parser.add_argument("--eps", type=int, default=300, help="DBSCAN eps in seconds (default: 300 = 5 min)")
    parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN min_samples (default: 2)")

    args = parser.parse_args()

    BASE_DIR = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Bank_time_line"

    print(f"📁 Loading anomalies for date={args.date}, window={args.window}")
    anomalies = load_anomalies_from_window(BASE_DIR, args.date, args.window)

    print(f"🎯 Total anomalies loaded: {len(anomalies)}")
    cluster_and_report(
        anomalies,
        output_file=args.output,
        eps_seconds=args.eps,
        min_samples=args.min_samples
    )
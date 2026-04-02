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

# load 所有异常
def load_anomalies_from_window(base_dir, date_str, window_str):
    """加载 Telecom 的三类异常文件（metric_A, metric_B, trace），并去重"""
    anomalies = []

    file_specs = [
        ("metric_A", f"Telecom_metric_A_anomalies_{date_str}_{window_str}.npy"),
        ("metric_B", f"Telecom_metric_B_anomalies_{date_str}_{window_str}.npy"),
        ("trace", f"Telecom_trace_anomalies_{date_str}_{window_str}.npy")
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
            if typ == "metric_A":
                # 假设格式: (entity, attribute, timestamp)
                entity, attr, ts = item
                anomalies.append({
                    'ts': int(ts),
                    'type': 'metric_A',
                    'entity': str(entity),
                    'attribute': str(attr),
                    'raw': ''
                })
            elif typ == "metric_B":
                entity, attr, ts = item
                anomalies.append({
                    'ts': int(ts),
                    'type': 'metric_B',
                    'entity': str(entity),
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

    # 🔥 去重：避免同一 (type, entity, attr, ts) 重复
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

import os
import numpy as np
from collections import defaultdict

# 只 load metric_A中3个以上的异常
def load_anomalies_from_window_new(base_dir, date_str, window_str):
    """加载 Telecom 的三类异常文件（metric_A, metric_B, trace），并去重；
       对 metric_A 只保留 (entity, attribute) 出现 >=3 次的记录。
    """
    raw_anomalies = []  # 存储所有原始加载的异常（含重复）
    file_specs = [
        ("metric_A", f"Telecom_metric_A_anomalies_{date_str}_{window_str}.npy"),
        ("metric_B", f"Telecom_metric_B_anomalies_{date_str}_{window_str}.npy"),
        ("trace", f"Telecom_trace_anomalies_{date_str}_{window_str}.npy")
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
            if typ == "metric_A":
                entity, attr, ts = item
                raw_anomalies.append({
                    'ts': int(ts),
                    'type': 'metric_A',
                    'entity': str(entity),
                    'attribute': str(attr),
                    'raw': ''
                })
            elif typ == "metric_B":
                entity, attr, ts = item
                raw_anomalies.append({
                    'ts': int(ts),
                    'type': 'metric_B',
                    'entity': str(entity),
                    'attribute': str(attr),
                    'raw': ''
                })
            elif typ == "trace":
                edge, attr, ts = item
                raw_anomalies.append({
                    'ts': int(ts),
                    'type': 'trace',
                    'entity': str(edge),
                    'attribute': str(attr),
                    'raw': ''
                })

    # 第一步：对 metric_A 按 (entity, attribute) 分组计数
    metric_a_groups = defaultdict(list)
    other_anomalies = []

    for a in raw_anomalies:
        if a['type'] == 'metric_A':
            key = (a['entity'], a['attribute'])
            metric_a_groups[key].append(a)
        else:
            other_anomalies.append(a)

    # 第二步：只保留出现 >=3 次的 metric_A 异常
    filtered_metric_a = []
    for key, items in metric_a_groups.items():
        if len(items) >= 4:
            filtered_metric_a.extend(items)

    # 第三步：合并 filtered_metric_a + 其他类型（metric_B, trace）
    anomalies = filtered_metric_a + other_anomalies

    # 🔥 去重：避免同一 (type, entity, attr, ts) 重复
    seen = set()
    unique_anomalies = []
    for a in anomalies:
        key = (a['type'], a['entity'], a['attribute'], a['ts'])
        if key not in seen:
            seen.add(key)
            unique_anomalies.append(a)

    unique_anomalies.sort(key=lambda x: x['ts'])
    print(f"🧹 After deduplication and metric_A filtering: {len(unique_anomalies)} unique anomalies "
          f"(original total: {len(raw_anomalies)})")
    return unique_anomalies

def cluster_and_report(anomalies, output_file, eps_seconds=300, min_samples=2):
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to cluster.")
        return

    # 提取时间戳用于聚类（一维）
    X = np.array([[a['ts']] for a in anomalies])
    
    labels = DBSCAN(
        eps=eps_seconds,
        min_samples=min_samples,
        metric='euclidean'
    ).fit_predict(X)

    clusters = defaultdict(list)
    noise = []
    for anomaly, label in zip(anomalies, labels):
        if label == -1:
            noise.append(anomaly)
        else:
            clusters[label].append(anomaly)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Anomaly Clustering Report for Telecom on {args.date_online} {args.output_suffix}\n")
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
            f.write("\n")

            # 按类型分组展示
            grouped = defaultdict(list)
            for a in cluster:
                grouped[a['type']].append(a)

            type_order = ['metric_A', 'metric_B', 'trace']
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster anomalies in a specific half-hour window of Telecom dataset.")
    parser.add_argument("--date_online", required=True, help="Date string like 2020_04_11")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0000_0030")
    parser.add_argument("--eps", type=int, default=60, help="DBSCAN eps in seconds (default: 60)")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min_samples (default: 3)")
    parser.add_argument("--output_folder_name", type=str, default="1216",
                        help="Output folder name (e.g., experiment ID)")

    args = parser.parse_args()

    BASE_DIR = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"

    print(f"📁 Loading anomalies for date={args.date_online}, window={args.output_suffix}")
    anomalies = load_anomalies_from_window(BASE_DIR, args.date_online, args.output_suffix)
    
    output_file = f"{BASE_DIR}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}.txt"

    print(f"🎯 Total anomalies loaded: {len(anomalies)}")
    cluster_and_report(
        anomalies,
        output_file=output_file,
        eps_seconds=args.eps,
        min_samples=args.min_samples
    )
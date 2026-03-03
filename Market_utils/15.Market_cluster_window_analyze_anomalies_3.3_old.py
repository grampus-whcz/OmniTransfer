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

# 确保 BEIJING_TZ 已定义（根据你的环境选择一种方式）
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    BEIJING_TZ = ZoneInfo("Asia/Shanghai")
except ImportError:
    import pytz  # 需要 pip install pytz（适用于旧版本 Python）
    BEIJING_TZ = pytz.timezone('Asia/Shanghai')


def load_anomalies_from_window_big_new(base_dir, date_str, window_str):
    """
    从全量 24 小时异常文件（_0000_2400.npy）中加载数据，
    并根据 window_str 指定的时间段进行过滤。

    处理逻辑：
      - 所有类型：先按时间窗口过滤
      - 所有类型：保留 (entity, attribute) 出现 >=4 次的组合
      - 特别地，对 metric_mesh 和 log：
          * 在满足 >=4 的组合中，跨 entity 选择频次最高的最多 3 个组合
          * 要求这 3 个组合来自 3 个不同的 entity（每个 entity 最多贡献 1 个）. 最新策略：只选一个
          * 保留选中组合的所有原始记录
      - 其他类型：保留所有 >=4 的组合（无 top-k 限制）
    """
    anomalies = []

    # === 解析 window_str 为目标时间窗口（格式: HHMM_HHMM，例如 0930_1000）===
    try:
        parts = window_str.split('_')
        if len(parts) != 2:
            raise ValueError(f"Expected format HHMM_HHMM, got '{window_str}'")

        start_hm, end_hm = parts[0], parts[1]
        if not (len(start_hm) == 4 and start_hm.isdigit() and len(end_hm) == 4 and end_hm.isdigit()):
            raise ValueError(f"Time parts must be 4-digit numbers (HHMM), got '{start_hm}' and '{end_hm}'")

        date_part = datetime.strptime(date_str, "%Y_%m_%d")
        start_hour, start_min = int(start_hm[:2]), int(start_hm[2:])
        end_hour, end_min = int(end_hm[:2]), int(end_hm[2:])

        start_dt = date_part.replace(hour=start_hour, minute=start_min, second=0, microsecond=0, tzinfo=BEIJING_TZ)
        end_dt = date_part.replace(hour=end_hour, minute=end_min, second=0, microsecond=0, tzinfo=BEIJING_TZ)

        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())  # [start_ts, end_ts)
    except Exception as e:
        print(f"⚠️ Failed to parse window '{window_str}': {e}. Skipping time filtering.")
        start_ts, end_ts = None, None

    # ❗ 关键：始终加载 _0000_2400.npy 的全量文件
    full_window = "0000_2400"
    file_specs = [
        ("metric_service", f"Market_metric_service_anomalies_{date_str}_{full_window}.npy"),
        ("metric_runtime", f"Market_metric_runtime_anomalies_{date_str}_{full_window}.npy"),
        ("metric_container", f"Market_metric_container_anomalies_{date_str}_{full_window}.npy"),
        ("metric_mesh", f"Market_metric_mesh_anomalies_{date_str}_{full_window}.npy"),
        ("metric_node", f"Market_metric_node_anomalies_{date_str}_{full_window}.npy"),
        ("trace", f"Market_trace_anomalies_{date_str}_{full_window}.npy"),
        ("log", f"Market_log_anomalies_{date_str}_{full_window}.npy")
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
                entity_key = str(pod)
                attr_key = template  # 使用 template 作为 attribute
            else:
                entity, attr, ts = item
                ts = int(ts)
                entity_key = str(entity)
                attr_key = str(attr)

            anomalies.append({
                'ts': ts,
                'type': typ,
                'entity': entity_key,
                'attribute': attr_key,
                'raw': template if typ == "log" else ''
            })

    print(f"📊 Total anomalies loaded (before window filter): {len(anomalies)}")

    # === 时间范围过滤：只保留 window_str 指定区间内的异常 ===
    if start_ts is not None and end_ts is not None:
        anomalies = [a for a in anomalies if start_ts <= a['ts'] < end_ts]
        print(f"🕒 After time window [{window_str}] filtering: {len(anomalies)} anomalies")

    # === 按类型分组处理 ===
    final_anomalies = []
    all_types = set(a['type'] for a in anomalies)

    for typ in all_types:
        type_anomalies = [a for a in anomalies if a['type'] == typ]

        if typ in ['metric_service', 'metric_runtime', 'metric_container', 'metric_mesh', 'metric_node', 'trace', 'log']:
            # 构建 (entity, attribute) -> list of anomalies 映射，并统计频次
            counter = defaultdict(int)
            group_map = defaultdict(list)
            for a in type_anomalies:
                key = (a['entity'], a['attribute'])
                counter[key] += 1
                group_map[key].append(a)

            # 只考虑频次 >=4 的组合
            frequent_items = [(key, count) for key, count in counter.items() if count >= 4]
            # 按频次降序排序（频次相同则顺序不确定，但不影响 correctness）
            frequent_items.sort(key=lambda x: x[1], reverse=True)

            # 贪心选择：每个 entity 最多选 1 个，最多选 1 个
            selected_keys = set()
            seen_entities = set()
            for (entity, attr), freq in frequent_items:
                if len(selected_keys) >= 1:
                    break
                if entity not in seen_entities:
                    selected_keys.add((entity, attr))
                    seen_entities.add(entity)

            # 收集选中组合的所有原始记录
            for key in selected_keys:
                final_anomalies.extend(group_map[key])

            print(f"🔝 For {typ}: selected {len(selected_keys)} (entity, attribute) groups "
                  f"from {len(seen_entities)} distinct entities (top by frequency, entity-unique)")

        else:
            # 其他类型：仅保留频次 >=4 的组合，不做 top-k 限制
            counter = defaultdict(int)
            for a in type_anomalies:
                key = (a['entity'], a['attribute'])
                counter[key] += 1

            frequent_keys = {k for k, v in counter.items() if v >= 4}
            filtered = [a for a in type_anomalies if (a['entity'], a['attribute']) in frequent_keys]
            final_anomalies.extend(filtered)
            print(f"📈 For {typ}: kept {len(frequent_keys)} (entity, attribute) groups (all >=4)")

    # === 去重（按 type + entity + attribute + ts）===
    seen = set()
    unique_anomalies = []
    for a in final_anomalies:
        key = (a['type'], a['entity'], a['attribute'], a['ts'])
        if key not in seen:
            seen.add(key)
            unique_anomalies.append(a)

    unique_anomalies.sort(key=lambda x: x['ts'])
    print(f"🧹 Final unique anomalies: {len(unique_anomalies)}")
    return unique_anomalies

import re

def extract_keywords(template):
    """从 log 模板中提取关键故障词（与 Bank 一致）"""
    if template is None:
        return []
    if not isinstance(template, str):
        return []
    
    keywords = set()
    t_low = template.lower()
    
    # 定义关键字及其对应的正则模式（使用单词边界）
    patterns = {
        "OOM": r'\b(out of memory|oom|java\.lang\.outofmemoryerror)\b',
        "GC": r'\b(gc\s*(overhead\s*limit|allocation\s*failure|full\s*gc))\b',
        "Error/Failure": r'\b(error|exception)\b',
        "Timeout": r'\btimeout\b'
    }
    
    for keyword, pattern in patterns.items():
        if re.search(pattern, t_low):
            keywords.add(keyword)
    
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
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min_samples (default: 3)")
    parser.add_argument("--output_folder_name", type=str, default="1215",
                        help="Output folder name (e.g., experiment ID)")

    args = parser.parse_args()

    BASE_DIR = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"
    print(f"📁 Loading Market anomalies for date={args.date_online}, window={args.output_suffix}")

    anomalies = load_anomalies_from_window_big_new(BASE_DIR, args.date_online, args.output_suffix)
    output_file = f"{BASE_DIR}/Market_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}.txt"

    print(f"🎯 Total anomalies loaded: {len(anomalies)}")
    cluster_and_report(
        anomalies,
        output_file=output_file,
        eps_seconds=args.eps,
        min_samples=args.min_samples
    )
    
    # 生成精简版报告
    condensed_output_file = f"{BASE_DIR}/Market_cluster_window_anomaly_short_report_{args.date_online}_{args.output_suffix}.txt"
    cluster_and_report_condensed(
        anomalies,
        output_file=condensed_output_file,
        eps_seconds=args.eps,
        min_samples=args.min_samples
    )
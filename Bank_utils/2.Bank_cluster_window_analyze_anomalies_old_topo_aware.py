import os
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
import argparse

# ===================== 固定Bank业务调用拓扑边 =====================
BASE_EDGES = [
    ("apache01", "IG01"), ("apache01", "IG02"),
    ("apache02", "IG01"), ("apache02", "IG02"),
    ("IG01", "Tomcat02"), ("IG02", "Tomcat02"),
    ("Tomcat02", "MG01"), ("Tomcat02", "MG02"),
    ("MG01", "dockerA2"), ("MG02", "dockerA2"),
    ("dockerA2", "Mysql02"),
    ("Tomcat02", "Redis02"), ("Redis02", "Tomcat02"),
    ("MG01", "Redis02"), ("MG02", "Redis02"), ("Redis02", "MG01"), ("Redis02", "MG02"),
]
# ====================================================================

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

def build_topology_graph(edges):
    """构建无向连通图（服务双向连通）"""
    adj = defaultdict(set)
    all_nodes = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        all_nodes.add(u)
        all_nodes.add(v)
    return adj, all_nodes

def get_connected_component(entity, adj, visited):
    """BFS获取当前实体所在拓扑连通分量"""
    if entity in visited or entity not in adj:
        return None
    q = deque([entity])
    visited.add(entity)
    component = set([entity])
    while q:
        node = q.popleft()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                component.add(neighbor)
                q.append(neighbor)
    return component

def pure_topology_aware_cluster(anomalies, output_file):
    """
    纯拓扑感知聚类：不做任何时序分段，直接按服务拓扑连通性分组
    :param anomalies: 全局已排序、去重后的全部异常
    :param output_file: 报告输出路径
    """
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to cluster.")
        return

    # 1. 构建全局服务拓扑邻接图
    topo_adj, all_topo_nodes = build_topology_graph(BASE_EDGES)

    # 2. 直接全局划分拓扑连通簇，无前置时序分段
    entity_visited = set()
    topo_cluster_map = defaultdict(list)
    isolated_anomalies = []

    for ano in anomalies:
        ent = ano['entity']
        # 实体不在业务拓扑中，作为孤立异常单独一簇
        if ent not in all_topo_nodes:
            isolated_anomalies.append(ano)
            continue
        # 当前实体已归属某拓扑连通分量，直接追加
        if ent in entity_visited:
            for comp_key, ano_list in topo_cluster_map.items():
                if any(x['entity'] == ent for x in ano_list):
                    topo_cluster_map[comp_key].append(ano)
                    break
            continue
        # 全新拓扑连通域，BFS提取连通服务集合
        comp_nodes = get_connected_component(ent, topo_adj, entity_visited)
        comp_key = frozenset(comp_nodes)
        topo_cluster_map[comp_key].append(ano)

    # 整合全部拓扑簇 + 孤立异常单簇
    final_clusters = []
    for comp_key, ano_list in topo_cluster_map.items():
        final_clusters.append({
            "topo_component": set(comp_key),
            "anomalies": ano_list
        })
    for single_ano in isolated_anomalies:
        final_clusters.append({
            "topo_component": {single_ano['entity']},
            "anomalies": [single_ano]
        })

    total_clusters = len(final_clusters)

    # 3. 生成标准统一格式报告
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Pure Topology-Aware Clustering Report for {args.date_online} {args.output_suffix}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"🔍 Total Topology Clusters Generated: {total_clusters}\n")
        f.write(f"🔍 Built-in Service Topology Edge Count: {len(BASE_EDGES)}\n")
        f.write("🔍 Clustering Pipeline: No temporal segmentation, group anomalies directly by service topology connected components\n")
        f.write("=" * 40 + "\n\n")

        for idx, cluster_info in enumerate(final_clusters):
            cluster_anos = cluster_info["anomalies"]
            topo_comp = cluster_info["topo_component"]

            ts_list = [a['ts'] for a in cluster_anos]
            start_ts = min(ts_list)
            end_ts = max(ts_list)
            duration = end_ts - start_ts

            f.write(f"🚨 Topology Cluster #{idx + 1}\n")
            f.write(f"   Connected Service Topology Component: {sorted(topo_comp)}\n")
            f.write(f"   Global Time Span of All Anomalies in This Cluster: {ts_to_beijing_str(start_ts)} → {ts_to_beijing_str(end_ts)} "
                    f"(Δ = {duration} sec)\n")
            f.write(f"   Total Anomalies in This Topology Cluster: {len(cluster_anos)}\n")

            # 汇总当前簇日志故障关键词
            cluster_keywords = set()
            for a in cluster_anos:
                if a['type'] == 'log':
                    cluster_keywords.update(extract_keywords(a['raw']))
            if cluster_keywords:
                f.write(f"   🔑 Cluster Log Keywords: {', '.join(cluster_keywords)}\n")
            f.write("\n")

            # 按模态分组输出异常详情
            grouped = defaultdict(list)
            for a in cluster_anos:
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
        f.write("   Clustering Mechanism: Pure topology-based grouping without any temporal window segmentation.\n")
        f.write("   Anomalies sharing connected service call graph are grouped into one cluster regardless of their absolute timestamps.\n")

    print(f"✅ Pure Topology-Aware clustering report saved to: {output_file}")
    print(f"📊 Generated {total_clusters} pure topology clusters (no pre-segmentation).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Topology-Aware Clustering (No temporal segmentation) for multi-modal bank anomalies.")
    parser.add_argument("--date_online", required=True, help="Date string like 2021_03_04")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0030_0100")
    # 占位参数，兼容原有run.sh脚本，本基线不使用
    parser.add_argument("--seg_width", type=int, default=300, help="(Unused for pure topology clustering)")
    parser.add_argument("--eps", type=int, default=60, help="(Deprecated placeholder) DBSCAN eps seconds")
    parser.add_argument("--min_samples", type=int, default=3, help="(Deprecated placeholder) DBSCAN min_samples")
    parser.add_argument("--output_folder_name", type=str, default="1116",
                        help="Output folder name passed to run.sh (e.g., experiment ID)")

    args = parser.parse_args()

    BASE_DIR = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"

    print(f"📁 Loading anomalies for date={args.date_online}, window={args.output_suffix}")
    anomalies = load_anomalies_from_window(BASE_DIR, args.date_online, args.output_suffix)
    
    output_file = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}/Bank_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}.txt"

    print(f"🎯 Total unique anomalies loaded: {len(anomalies)}")
    pure_topology_aware_cluster(anomalies, output_file=output_file)
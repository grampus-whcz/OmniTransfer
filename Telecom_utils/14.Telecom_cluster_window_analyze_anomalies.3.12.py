import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
import argparse

# === 全局配置 ===
# tune5
CONCENTRATION_WINDOW_MINUTES = 5  # 分析时间窗口
ANOMALY_THRESHOLD = 1             # 异常数阈值
FALLBACK_THRESHOLD = 1            # 兜底阈值
WEIGHT_TIME = 0.025                 # 时间权重（越早异常权重越高）
WEIGHT_TOPOLOGY = 0.025             # 拓扑权重（依赖影响范围）
WEIGHT_COUNT = 0.95                # 异常数权重
ANALYSIS_START_TIMESTAMP_INDEX = 2  # 分析起点时间戳索引

# 北京时区 (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))

# Telecom 故障类型映射（属性→故障原因）
FAULT_TYPE_MAPPING = {
    # CPU 故障相关属性
    'cpu_usage': 'CPU fault',
    'cpu_utilization': 'CPU fault',
    'cpu_load': 'CPU fault',
    'cpu_saturation': 'CPU fault',
    # 网络延迟相关属性
    'network_latency': 'network delay',
    'latency': 'network delay',
    'response_time': 'network delay',
    'rt': 'network delay',
    # 网络丢包相关属性
    'packet_loss': 'network loss',
    'network_loss': 'network loss',
    'drop_rate': 'network loss',
    # 数据库连接限制相关属性
    'db_connection': 'db connection limit',
    'connection_count': 'db connection limit',
    'max_connections': 'db connection limit',
    'conn_limit': 'db connection limit',
    # 数据库关闭相关属性
    'db_status': 'db close',
    'db_down': 'db close',
    'db_shutdown': 'db close',
    'connection_closed': 'db close'
}

# Telecom 组件分层权重
COMPONENT_WEIGHTS = {
    'db': 1.0,       # 数据库服务（核心）
    'os': 0.9,       # 操作系统节点
    'docker': 0.85   # Pod/容器
}

def ts_to_beijing_str(ts):
    """将 Unix 时间戳转为北京时间字符串"""
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(BEIJING_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + " CST"
    except:
        return "N/A"

def ts_to_graph_node_format(ts):
    """将时间戳转为图谱节点命名格式（Time_年月日时分）"""
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(BEIJING_TZ)
        return f"Time_{dt.strftime('%Y%m%d%H%M')}"
    except:
        return f"Time_unknown_{ts}"

def extract_main_entity(entity_str):
    """从子实体中提取主实体（如 docker_001:LOCAL,db_009 → docker_001）"""
    if ':' in entity_str:
        return entity_str.split(':')[0]
    return entity_str

# === Telecom 专用拓扑 RCA 分析器 ===
class TelecomTopologyRCAAnalyzer:
    def __init__(self, dependency_graphs_path):
        # 加载调用图依赖关系
        self.dependency_graphs = self._load_dependency_graphs(dependency_graphs_path)
        # 构建全局拓扑图
        self.global_topology = self._build_global_topology()
        # 组件类型映射
        self.component_types = self._init_component_types()

    def _load_dependency_graphs(self, file_path):
        """加载调用图依赖关系文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                graphs = json.load(f)
            print(f"✅ Loaded {len(graphs)} dependency graphs from {file_path}")
            return graphs
        except Exception as e:
            print(f"❌ Failed to load dependency graphs: {e}")
            return []

    def _build_global_topology(self):
        """构建全局拓扑图（合并所有依赖图）"""
        G = nx.DiGraph()
        
        # 合并所有依赖边（双向，因为依赖是相互的）
        for graph in self.dependency_graphs:
            for edge in graph:
                if len(edge) == 2:
                    source, target = edge[0], edge[1]
                    G.add_edge(source, target)
                    G.add_edge(target, source)  # 双向边
        
        # 添加 OS → Pod 的默认关联（OS 托管 Pod）
        os_list = [f"os_{i:03d}" for i in range(1, 23)]
        docker_list = [f"docker_{i:03d}" for i in range(1, 9)]
        # 简单映射：os_001-004 → docker_001-002，os_005-008 → docker_003-004，以此类推
        for i, docker in enumerate(docker_list):
            os_idx = (i // 2) + 1
            os_node = f"os_{os_idx:03d}" if os_idx <= 22 else f"os_22"
            if os_node in os_list:
                G.add_edge(os_node, docker)
                G.add_edge(docker, os_node)
        
        # 添加 DB 与 OS 的直接关联（DB 部署在 OS 上）
        db_list = [f"db_{i:03d}" for i in range(1, 14)]
        for i, db in enumerate(db_list):
            os_idx = (i // 2) + 1
            os_node = f"os_{os_idx:03d}" if os_idx <= 22 else f"os_22"
            if os_node in os_list:
                G.add_edge(os_node, db)
                G.add_edge(db, os_node)
        
        print(f"✅ Built global topology graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def _init_component_types(self):
        """初始化组件类型映射"""
        component_types = {}
        
        # OS 节点
        for i in range(1, 23):
            component_types[f"os_{i:03d}"] = 'os'
        
        # Docker/Pod 节点
        for i in range(1, 9):
            component_types[f"docker_{i:03d}"] = 'docker'
        
        # DB 服务节点
        for i in range(1, 14):
            component_types[f"db_{i:03d}"] = 'db'
        
        return component_types

    def _get_component_type(self, entity):
        """获取组件类型（os/docker/db）"""
        main_entity = extract_main_entity(entity)
        return self.component_types.get(main_entity, 'unknown')

    def _map_fault_type(self, attribute):
        """从属性映射到故障类型"""
        attr_lower = attribute.lower()
        for keyword, fault_type in FAULT_TYPE_MAPPING.items():
            if keyword in attr_lower:
                return fault_type
        return "unknown"

    def prepare_anomalies_for_rca(self, anomalies):
        """增强异常数据（添加组件类型、故障类型）"""
        enriched = []
        for a in anomalies:
            entity = a['entity']
            component_type = self._get_component_type(entity)
            fault_type = self._map_fault_type(a['attribute'])
            
            enriched.append({
                'entity': entity,
                'type': a['type'],
                'attribute': a['attribute'],
                'ts': a['ts'],
                'timestamp': a['ts'],
                'component_type': component_type,
                'fault_type': fault_type,
                'component_weight': COMPONENT_WEIGHTS.get(component_type, 0.5),
                'main_entity': extract_main_entity(entity)
            })
        return enriched

    def calculate_rca_scores(self, anomalies):
        """核心RCA评分计算（时间+拓扑+异常数）"""
        try:
            # 增强异常数据
            enriched_anomalies = self.prepare_anomalies_for_rca(anomalies)
            if not enriched_anomalies:
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(enriched_anomalies)
            
            # Step 1: 筛选时间窗口
            all_ts = df['timestamp'].dropna().astype(int).tolist()
            if not all_ts:
                return None
            
            sorted_ts = sorted(list(set(all_ts)))
            if ANALYSIS_START_TIMESTAMP_INDEX >= len(sorted_ts):
                T_start = sorted_ts[-1] if sorted_ts else 0
            else:
                T_start = sorted_ts[ANALYSIS_START_TIMESTAMP_INDEX]
            
            T_window = T_start + CONCENTRATION_WINDOW_MINUTES * 60
            df_window = df[df['timestamp'] <= T_window].copy()
            
            # Step 2: 统计每个实体的异常数量
            # 先统计总数
            entity_counts = df_window.groupby('entity').size().reset_index(name='total_anomaly_count')
            # 统计最早时间
            entity_earliest = df_window.groupby('entity')['timestamp'].min().reset_index(name='earliest_timestamp')
            # 统计主要故障类型
            entity_fault = df_window.groupby('entity')['fault_type'].agg(
                lambda x: Counter(x).most_common(1)[0][0] if x.any() else 'unknown'
            ).reset_index(name='fault_type')
            # 统计组件类型
            entity_component = df_window.groupby('entity')['component_type'].first().reset_index(name='component_type')
            
            # 合并所有统计信息
            entity_stats = entity_counts.merge(entity_earliest, on='entity')
            entity_stats = entity_stats.merge(entity_fault, on='entity')
            entity_stats = entity_stats.merge(entity_component, on='entity')
            
            # Step 3: 拓扑特征计算
            def get_topology_features(entity):
                """获取拓扑特征（入度/出度/可达节点数）"""
                main_entity = extract_main_entity(entity)
                if main_entity not in self.global_topology.nodes:
                    return 0, 0, 0
                
                # 入度/出度
                in_degree = self.global_topology.in_degree(main_entity)
                out_degree = self.global_topology.out_degree(main_entity)
                
                # 可达节点数（影响范围）
                try:
                    reachable_nodes = nx.descendants(self.global_topology, main_entity)
                    reachable_count = len(reachable_nodes)
                except:
                    reachable_count = 0
                
                return in_degree, out_degree, reachable_count
            
            # 应用拓扑特征计算
            topology_features = entity_stats['entity'].apply(get_topology_features)
            entity_stats[['in_degree', 'out_degree', 'reachable_count']] = pd.DataFrame(
                topology_features.tolist(), index=entity_stats.index
            )
            
            # Step 4: 筛选候选实体
            df_candidate = entity_stats[entity_stats['total_anomaly_count'] >= ANOMALY_THRESHOLD].copy()
            if df_candidate.empty:
                df_candidate = entity_stats[entity_stats['total_anomaly_count'] >= FALLBACK_THRESHOLD].copy()
            
            if df_candidate.empty:
                return None
            
            # Step 5: 计算各项得分
            df_scored = df_candidate.copy()
            
            # 时间得分（越早越高）
            t_min = df_scored['earliest_timestamp'].min()
            t_max = df_scored['earliest_timestamp'].max()
            if t_min == t_max:
                df_scored['time_score'] = 1.0
            else:
                df_scored['time_score'] = (t_max - df_scored['earliest_timestamp']) / (t_max - t_min)
            
            # 拓扑得分
            max_in = df_scored['in_degree'].max() if df_scored['in_degree'].max() > 0 else 1
            max_out = df_scored['out_degree'].max() if df_scored['out_degree'].max() > 0 else 1
            max_reach = df_scored['reachable_count'].max() if df_scored['reachable_count'].max() > 0 else 1
            
            df_scored['in_degree_score'] = df_scored['in_degree'] / max_in
            df_scored['out_degree_score'] = df_scored['out_degree'] / max_out
            df_scored['reachable_score'] = df_scored['reachable_count'] / max_reach
            df_scored['topology_score'] = (
                df_scored['in_degree_score'] + 
                df_scored['out_degree_score'] + 
                df_scored['reachable_score']
            ) / 3
            
            # 数量得分
            max_count = df_scored['total_anomaly_count'].max() if df_scored['total_anomaly_count'].max() > 0 else 1
            df_scored['count_score'] = df_scored['total_anomaly_count'] / max_count
            
            # 组件权重修正
            df_scored['component_weight'] = df_scored['component_type'].map(COMPONENT_WEIGHTS).fillna(0.5)
            
            # 最终加权得分
            df_scored['final_score'] = (
                WEIGHT_TIME * df_scored['time_score'] +
                WEIGHT_TOPOLOGY * df_scored['topology_score'] +
                WEIGHT_COUNT * df_scored['count_score']
            ) * df_scored['component_weight']
            
            # 排序并保留两位小数
            df_scored = df_scored.sort_values('final_score', ascending=False).reset_index(drop=True)
            score_cols = ['time_score', 'topology_score', 'count_score', 'final_score']
            df_scored[score_cols] = df_scored[score_cols].round(2)
            
            return {
                'scored_candidates': df_scored,
                'topology_graph': self.global_topology,
                'time_window': (T_start, T_window),
                'total_anomalies': len(anomalies),
                'analysis_start_ts': T_start
            }
        except Exception as e:
            print(f"❌ Error in calculate_rca_scores: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_rca_report(self, cluster_id, anomalies):
        """生成RCA分析报告"""
        try:
            # 运行RCA分析
            rca_results = self.calculate_rca_scores(anomalies)
            if not rca_results:
                return self._generate_fallback_report(cluster_id, anomalies)
            
            df_scored = rca_results['scored_candidates']
            G = rca_results['topology_graph']
            T_start, T_window = rca_results['time_window']
            total_anomalies = rca_results['total_anomalies']  # 修复：使用正确的键名
            
            # 格式化时间
            t_start_str = ts_to_beijing_str(T_start)
            t_window_str = ts_to_beijing_str(T_window)
            
            # 构建报告
            report = []
            report.append(f"### Root Cause Analysis Report - Cluster #{cluster_id} (Telecom Dataset)")
            report.append(f"**Analysis Window**: {t_start_str} to {t_window_str} ({CONCENTRATION_WINDOW_MINUTES} minutes)")
            report.append(f"**Total Anomalies**: {len(anomalies)}")
            report.append("")
            
            # 主根因
            root_cause = df_scored.iloc[0]['entity']
            root_score = df_scored.iloc[0]['final_score']
            root_component = df_scored.iloc[0]['component_type']
            root_fault = df_scored.iloc[0]['fault_type']
            root_anomaly_count = int(df_scored.iloc[0]['total_anomaly_count'])
            
            report.append(f"#### Primary Root Cause Entity")
            report.append(f"- **Entity**: {root_cause} (Type: {root_component.upper()})")
            report.append(f"- **Confidence Score**: {root_score:.2f} (1.0 = highest)")
            report.append(f"- **Anomaly Count**: {root_anomaly_count}")
            report.append(f"- **Predicted Fault Type**: {root_fault}")
            
            earliest_ts = df_scored.iloc[0]['earliest_timestamp']
            if earliest_ts:
                report.append(f"- **Earliest Anomaly Time**: {ts_to_beijing_str(earliest_ts)}")
            
            # 拓扑影响
            reachable_count = df_scored.iloc[0]['reachable_count']
            report.append(f"- **Topology Impact**: {reachable_count} downstream entities affected")
            
            # 依赖路径
            main_root_entity = extract_main_entity(root_cause)
            if main_root_entity in G.nodes:
                try:
                    neighbors = list(G.neighbors(main_root_entity))[:5]
                    if neighbors:
                        report.append(f"- **Direct Dependencies**: {', '.join(neighbors)}")
                except:
                    pass
            report.append("")
            
            # 得分拆解
            report.append(f"#### Score Breakdown for {root_cause}")
            report.append(f"- **Time Score**: {df_scored.iloc[0]['time_score']:.2f} (earlier anomaly = higher score)")
            report.append(f"- **Topology Score**: {df_scored.iloc[0]['topology_score']:.2f} (based on dependency impact)")
            report.append(f"- **Anomaly Count Score**: {df_scored.iloc[0]['count_score']:.2f} (more anomalies = higher score)")
            report.append(f"- **Component Weight**: {COMPONENT_WEIGHTS.get(root_component, 0.5)} ({root_component.upper()} layer weight)")
            report.append("")
            
            # 次要根因
            if len(df_scored) > 1:
                report.append(f"#### Secondary Root Cause Candidates (Top 3)")
                for idx in range(1, min(4, len(df_scored))):
                    row = df_scored.iloc[idx]
                    report.append(f"- **{row['entity']}** (Type: {row['component_type'].upper()}): "
                                 f"Score = {row['final_score']:.2f}, Fault = {row['fault_type']}, "
                                 f"Anomalies = {int(row['total_anomaly_count'])}")
                report.append("")
            
            # 故障类型分析
            fault_counter = Counter([a['fault_type'] for a in self.prepare_anomalies_for_rca(anomalies)])
            if fault_counter:
                report.append(f"#### Fault Type Distribution")
                for fault_type, count in fault_counter.most_common(3):
                    if fault_type != 'unknown':
                        report.append(f"- **{fault_type}**: {count} anomalies ({count/len(anomalies)*100:.1f}%)")
                report.append("")
            
            # 建议
            report.append(f"#### RCA Recommendations")
            report.append(f"1. Prioritize investigation of {root_cause} (highest confidence score)")
            report.append(f"2. Verify {root_fault} issues in {root_component.upper()} layer")
            report.append(f"3. Check dependent entities of {root_cause} for cascading failures")
            report.append(f"4. Monitor resource utilization (CPU/network/database) for {root_cause}")
            
            return "\n".join(report)
        except Exception as e:
            print(f"❌ Error generating RCA report: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_report(cluster_id, anomalies)

    def _generate_fallback_report(self, cluster_id, anomalies):
        """兜底报告（无RCA结果时）"""
        report = []
        report.append(f"### Root Cause Analysis Report - Cluster #{cluster_id} (Fallback Analysis)")
        report.append(f"**Total Anomalies**: {len(anomalies)}")
        report.append("")
        
        # 基础统计
        entity_counter = Counter([a['entity'] for a in anomalies])
        fault_counter = Counter([self._map_fault_type(a['attribute']) for a in anomalies])
        
        # 主要实体
        top_entity = entity_counter.most_common(1)[0][0] if entity_counter else "unknown"
        top_fault = fault_counter.most_common(1)[0][0] if fault_counter else "unknown"
        
        report.append(f"#### Primary Root Cause Hypothesis")
        report.append(f"- **Top Anomaly Entity**: {top_entity} (Anomalies: {entity_counter.get(top_entity, 0)})")
        report.append(f"- **Predicted Fault Type**: {top_fault} (Occurrences: {fault_counter.get(top_fault, 0)})")
        report.append("")
        
        # 实体分布
        report.append(f"#### Anomaly Distribution (Top Entities)")
        for entity, count in entity_counter.most_common(5):
            component_type = self._get_component_type(entity)
            report.append(f"- **{entity}** (Type: {component_type.upper()}): {count} anomalies")
        report.append("")
        
        # 建议
        report.append(f"#### Recommendations")
        report.append(f"1. Investigate {top_entity} for {top_fault} issues")
        report.append(f"2. Check {self._get_component_type(top_entity).upper()} layer infrastructure")
        report.append(f"3. Verify network/database connectivity for affected entities")
        
        return "\n".join(report)

# === 原有数据加载逻辑（保留并优化）===
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
                # 解析trace edge（如 "docker_001->db_009"）
                if '->' in str(edge):
                    source, target = str(edge).split('->')
                    # 添加源和目标实体的异常记录
                    raw_anomalies.append({
                        'ts': int(ts),
                        'type': 'trace',
                        'entity': source,
                        'attribute': f"trace_{attr}",
                        'raw': str(edge)
                    })
                    raw_anomalies.append({
                        'ts': int(ts),
                        'type': 'trace',
                        'entity': target,
                        'attribute': f"trace_{attr}",
                        'raw': str(edge)
                    })
                else:
                    raw_anomalies.append({
                        'ts': int(ts),
                        'type': 'trace',
                        'entity': str(edge),
                        'attribute': str(attr),
                        'raw': str(edge)
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

    # 第二步：只保留出现 >=4 次的 metric_A 异常
    filtered_metric_a = []
    for key, items in metric_a_groups.items():
        if len(items) >= 4:
            filtered_metric_a.extend(items)

    # 第三步：合并 filtered_metric_a + 其他类型（metric_B, trace）
    anomalies = filtered_metric_a + other_anomalies

    # 去重：避免同一 (type, entity, attr, ts) 重复
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

# === 新增：知识图谱构建函数 ===
def build_cluster_knowledge_graph(cluster_id, cluster_anomalies, rca_analyzer, output_dir, date_str, window_str):
    """
    为单个聚类构建知识图谱，并输出多种格式的图谱文件：
    1. JSON格式的图谱结构（节点+关系）
    2. Neo4j Cypher导入脚本
    3. NetworkX可视化图（GEXF格式）
    
    Args:
        cluster_id: 聚类ID（如 1, 2, 3）
        cluster_anomalies: 该聚类的异常列表
        rca_analyzer: RCA分析器实例
        output_dir: 输出目录
        date_str: 日期字符串（如 2020_04_11）
        window_str: 时间窗口（如 0000_0030）
    """
    # 创建输出子目录
    kg_output_dir = os.path.join(output_dir, f"knowledge_graphs", f"{date_str}_{window_str}", f"cluster_{cluster_id}")
    os.makedirs(kg_output_dir, exist_ok=True)
    
    # 1. 初始化图谱节点和关系
    nodes = []          # 节点列表：[{"id": "Time_202005260500", "label": "Time", "properties": {...}}]
    relationships = []  # 关系列表：[{"source": "Time_202005260500", "target": "os_001", "type": "HAS", "properties": {...}}]
    node_id_set = set() # 避免重复节点
    
    # 2. 增强异常数据
    enriched_anomalies = rca_analyzer.prepare_anomalies_for_rca(cluster_anomalies)
    
    # 3. 构建基础节点
    # 3.1 时间节点
    ts_list = sorted(list(set([a['ts'] for a in cluster_anomalies])))
    time_nodes = {}
    for ts in ts_list:
        time_node_id = ts_to_graph_node_format(ts)
        time_nodes[ts] = time_node_id
        if time_node_id not in node_id_set:
            nodes.append({
                "id": time_node_id,
                "label": "Time",
                "properties": {
                    "timestamp": ts,
                    "time_str": ts_to_beijing_str(ts),
                    "cluster_id": cluster_id
                }
            })
            node_id_set.add(time_node_id)
    
    # 3.2 故障类型节点
    fault_types = list(set([a['fault_type'] for a in enriched_anomalies]))
    fault_nodes = {}
    for fault_type in fault_types:
        fault_node_id = f"Fault_{fault_type.replace(' ', '_')}"
        fault_nodes[fault_type] = fault_node_id
        if fault_node_id not in node_id_set:
            nodes.append({
                "id": fault_node_id,
                "label": "FaultType",
                "properties": {
                    "fault_type": fault_type,
                    "cluster_id": cluster_id
                }
            })
            node_id_set.add(fault_node_id)
    
    # 3.3 异常属性节点
    attributes = list(set([a['attribute'] for a in cluster_anomalies]))
    attr_nodes = {}
    for attr in attributes:
        attr_node_id = f"Attr_{attr.replace(' ', '_').replace(':', '_').replace(',', '_')}"
        attr_nodes[attr] = attr_node_id
        if attr_node_id not in node_id_set:
            nodes.append({
                "id": attr_node_id,
                "label": "AnomalyAttribute",
                "properties": {
                    "attribute": attr,
                    "fault_type": rca_analyzer._map_fault_type(attr),
                    "cluster_id": cluster_id
                }
            })
            node_id_set.add(attr_node_id)
    
    # 3.4 实体节点（主实体+子实体）
    entities = list(set([a['entity'] for a in cluster_anomalies]))
    entity_nodes = {}
    main_entity_map = {}  # 子实体→主实体映射
    
    for entity in entities:
        main_entity = extract_main_entity(entity)
        main_entity_map[entity] = main_entity
        
        # 主实体节点
        if main_entity not in node_id_set:
            component_type = rca_analyzer._get_component_type(main_entity)
            nodes.append({
                "id": main_entity,
                "label": component_type.upper(),
                "properties": {
                    "entity_type": component_type,
                    "is_main_entity": True,
                    "cluster_id": cluster_id
                }
            })
            node_id_set.add(main_entity)
            entity_nodes[main_entity] = main_entity
        
        # 子实体节点（如果不是主实体）
        if entity != main_entity and entity not in node_id_set:
            component_type = rca_analyzer._get_component_type(entity)
            nodes.append({
                "id": entity,
                "label": f"{component_type.upper()}_Sub",
                "properties": {
                    "entity_type": component_type,
                    "is_main_entity": False,
                    "main_entity": main_entity,
                    "cluster_id": cluster_id
                }
            })
            node_id_set.add(entity)
            entity_nodes[entity] = entity
    
    # 4. 构建关系
    # 4.1 时间→实体（发生异常）
    for a in cluster_anomalies:
        time_node_id = time_nodes[a['ts']]
        entity_id = entity_nodes[a['entity']]
        relationships.append({
            "source": time_node_id,
            "target": entity_id,
            "type": "HAS_ANOMALY",
            "properties": {
                "anomaly_type": a['type'],
                "timestamp": a['ts'],
                "cluster_id": cluster_id
            }
        })
    
    # 4.2 实体→异常属性
    for a in cluster_anomalies:
        entity_id = entity_nodes[a['entity']]
        attr_node_id = attr_nodes[a['attribute']]
        relationships.append({
            "source": entity_id,
            "target": attr_node_id,
            "type": "HAS_ATTRIBUTE",
            "properties": {
                "anomaly_type": a['type'],
                "timestamp": a['ts'],
                "cluster_id": cluster_id
            }
        })
    
    # 4.3 异常属性→故障类型
    for a in enriched_anomalies:
        attr_node_id = attr_nodes[a['attribute']]
        fault_node_id = fault_nodes[a['fault_type']]
        relationships.append({
            "source": attr_node_id,
            "target": fault_node_id,
            "type": "MAPS_TO_FAULT",
            "properties": {
                "confidence": 1.0 if a['fault_type'] != 'unknown' else 0.5,
                "cluster_id": cluster_id
            }
        })
    
    # 4.4 子实体→主实体
    for entity in entities:
        main_entity = main_entity_map[entity]
        if entity != main_entity:
            relationships.append({
                "source": entity,
                "target": main_entity,
                "type": "BELONGS_TO",
                "properties": {
                    "cluster_id": cluster_id
                }
            })
    
    # 4.5 添加拓扑依赖关系（主实体之间）
    topology_graph = rca_analyzer.global_topology
    main_entities_in_cluster = list(set(main_entity_map.values()))
    for main_entity in main_entities_in_cluster:
        if main_entity in topology_graph.nodes:
            neighbors = list(topology_graph.neighbors(main_entity))
            for neighbor in neighbors:
                if neighbor in main_entities_in_cluster and neighbor in node_id_set:
                    # 添加双向依赖关系
                    relationships.append({
                        "source": main_entity,
                        "target": neighbor,
                        "type": "TOPOLOGY_DEPENDS_ON",
                        "properties": {
                            "cluster_id": cluster_id,
                            "dependency_type": "bidirectional"
                        }
                    })
    
    # 5. 输出JSON格式图谱
    kg_json = {
        "cluster_id": cluster_id,
        "total_anomalies": len(cluster_anomalies),
        "time_span": {
            "start": ts_to_beijing_str(min(ts_list)),
            "end": ts_to_beijing_str(max(ts_list)),
            "duration_sec": max(ts_list) - min(ts_list)
        },
        "nodes": nodes,
        "relationships": relationships
    }
    
    json_file = os.path.join(kg_output_dir, f"cluster_{cluster_id}_kg.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(kg_json, f, ensure_ascii=False, indent=2)
    print(f"📄 JSON格式图谱已保存: {json_file}")
    
    # 6. 生成Neo4j Cypher导入脚本
    cypher_lines = [
        f"// 聚类 {cluster_id} 异常知识图谱导入脚本",
        f"// 生成时间: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')}",
        f"// 异常数量: {len(cluster_anomalies)}",
        "",
        "// 清空原有数据（可选）",
        "// MATCH (n) WHERE n.cluster_id = {cluster_id} DELETE n;",
        "",
        "// 创建节点",
    ]
    
    # 6.1 创建节点Cypher
    for node in nodes:
        props = []
        for k, v in node['properties'].items():
            if isinstance(v, str):
                props.append(f"{k}: '{v}'")
            else:
                props.append(f"{k}: {v}")
        props_str = ", ".join(props)
        cypher_lines.append(f"CREATE (:{node['label']} {{id: '{node['id']}', {props_str}}});")
    
    cypher_lines.extend(["", "// 创建关系"])
    
    # 6.2 创建关系Cypher
    rel_id = 0
    for rel in relationships:
        props = []
        for k, v in rel['properties'].items():
            if isinstance(v, str):
                props.append(f"{k}: '{v}'")
            else:
                props.append(f"{k}: {v}")
        props_str = ", ".join(props) if props else ""
        cypher_lines.append(f"""
MATCH (s {{id: '{rel['source']}'}}), (t {{id: '{rel['target']}'}})
CREATE (s)-[:{rel['type']} {{id: 'rel_{cluster_id}_{rel_id}', {props_str}}}]->(t);
""")
        rel_id += 1
    
    # 保存Cypher脚本
    cypher_file = os.path.join(kg_output_dir, f"cluster_{cluster_id}_neo4j_import.cypher")
    with open(cypher_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(cypher_lines))
    print(f"📜 Neo4j Cypher脚本已保存: {cypher_file}")
    
    # 7. 生成NetworkX可视化图（GEXF格式）
    G = nx.DiGraph()
    
    # 添加节点
    for node in nodes:
        G.add_node(node['id'], **node['properties'], label=node['label'])
    
    # 添加关系
    for rel in relationships:
        G.add_edge(rel['source'], rel['target'], **rel['properties'], type=rel['type'])
    
    # 保存GEXF文件（可用于Gephi可视化）
    gexf_file = os.path.join(kg_output_dir, f"cluster_{cluster_id}_visualization.gexf")
    nx.write_gexf(G, gexf_file)
    print(f"🎨 GEXF可视化文件已保存: {gexf_file}")
    
    return kg_json

# === 重构的聚类和报告生成逻辑 ===
def cluster_and_report(anomalies, output_file, dependency_graphs_path, date_str, window_str, eps_seconds=300, min_samples=2):
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to cluster.")
        return

    # 初始化RCA分析器
    rca_analyzer = TelecomTopologyRCAAnalyzer(dependency_graphs_path)

    # 提取时间戳用于聚类
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
        f.write(f"🔍 Telecom Anomaly Clustering & Root Cause Analysis Report\n")
        f.write(f"📅 Date: {date_str} | Window: {window_str}\n")
        f.write(f"⚙️  RCA Methodology: Time({WEIGHT_TIME*100}%) + Topology({WEIGHT_TOPOLOGY*100}%) + Count({WEIGHT_COUNT*100}%)\n")
        f.write("=" * 80 + "\n\n")

        cluster_ids = sorted(clusters.keys())
        f.write(f"📊 Found {len(cluster_ids)} anomaly clusters (DBSCAN: eps={eps_seconds}s, min_samples={min_samples})\n")
        f.write("=" * 40 + "\n\n")

        for idx, cid in enumerate(cluster_ids):
            cluster = clusters[cid]
            ts_vals = [a['ts'] for a in cluster]
            start_ts, end_ts = min(ts_vals), max(ts_vals)
            duration = end_ts - start_ts

            # 基础聚类信息
            f.write(f"# Cluster #{idx + 1}\n")
            f.write(f"**Time Span**: {ts_to_beijing_str(start_ts)} → {ts_to_beijing_str(end_ts)} (Δ = {duration} sec)\n")
            f.write(f"**Total Anomalies**: {len(cluster)}\n")
            f.write("\n")

            # 按类型分组展示
            f.write("## Anomaly Breakdown\n")
            grouped = defaultdict(list)
            for a in cluster:
                grouped[a['type']].append(a)

            type_order = ['metric_A', 'metric_B', 'trace']
            for typ in type_order:
                if typ not in grouped:
                    continue
                f.write(f"### {typ.replace('_', ' ').title()} Anomalies:\n")
                entity_attr_dict = defaultdict(list)
                for a in grouped[typ]:
                    key = (a['entity'], a['attribute'])
                    entity_attr_dict[key].append(a['ts'])

                for (ent, attr), timestamps in sorted(entity_attr_dict.items()):
                    ts_sorted = sorted(timestamps)
                    time_repr = ", ".join([ts_to_beijing_str(ts) for ts in ts_sorted[:3]])  # 只显示前3个时间戳
                    if len(ts_sorted) > 3:
                        time_repr += f" (+{len(ts_sorted)-3} more)"
                    fault_type = rca_analyzer._map_fault_type(attr)
                    f.write(f"- **Entity**: {ent} | **Attribute**: {attr} | **Fault Type**: {fault_type}\n")
                    f.write(f"  Timestamps: {time_repr}\n")
                f.write("\n")

            # RCA分析
            f.write("## Root Cause Analysis (Topology-Based)\n")
            rca_report = rca_analyzer.generate_rca_report(idx + 1, cluster)
            f.write(rca_report)
            f.write("\n")
            f.write("-" * 80 + "\n\n")
            
            # 为该聚类构建知识图谱
            build_cluster_knowledge_graph(
                cluster_id=idx + 1,
                cluster_anomalies=cluster,
                rca_analyzer=rca_analyzer,
                output_dir=os.path.dirname(output_file),
                date_str=date_str,
                window_str=window_str
            )

        # 孤立异常
        if noise:
            f.write("# Isolated Anomalies (Noise / Single Events)\n")
            f.write(f"**Total Isolated Anomalies**: {len(noise)}\n")
            for a in sorted(noise, key=lambda x: x['ts']):
                fault_type = rca_analyzer._map_fault_type(a['attribute'])
                f.write(f"- {a['type']} | {a['entity']} | {a['attribute']} | Fault: {fault_type} | {ts_to_beijing_str(a['ts'])}\n")
            f.write("\n")
            
            # 为孤立异常构建单独的知识图谱
            build_cluster_knowledge_graph(
                cluster_id="noise",
                cluster_anomalies=noise,
                rca_analyzer=rca_analyzer,
                output_dir=os.path.dirname(output_file),
                date_str=date_str,
                window_str=window_str
            )

        # 元数据
        f.write("### Analysis Metadata\n")
        f.write(f"- Time Zone: CST (UTC+8)\n")
        f.write(f"- Clustering Algorithm: DBSCAN (eps={eps_seconds}s, min_samples={min_samples})\n")
        f.write(f"- RCA Weighting: Time({WEIGHT_TIME*100}%), Topology({WEIGHT_TOPOLOGY*100}%), Anomaly Count({WEIGHT_COUNT*100}%)\n")
        f.write(f"- Component Weights: DB=1.0, OS=0.9, Docker/Pod=0.85\n")
        f.write(f"- Fault Types: CPU fault, network delay, network loss, db connection limit, db close\n")
        f.write(f"- Dependency Graph: Loaded from {dependency_graphs_path} ({rca_analyzer.global_topology.number_of_nodes()} nodes, {rca_analyzer.global_topology.number_of_edges()} edges)\n")
        f.write(f"- Knowledge Graphs: Generated in {os.path.join(os.path.dirname(output_file), 'knowledge_graphs', f'{date_str}_{window_str}')}\n")

    print(f"✅ Report saved to: {output_file}")
    print(f"📊 Found {len(cluster_ids)} clusters and {len(noise)} isolated anomalies.")
    print(f"📈 Knowledge graphs saved to: {os.path.join(os.path.dirname(output_file), 'knowledge_graphs', f'{date_str}_{window_str}')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster anomalies with topology-based RCA for Telecom dataset.")
    parser.add_argument("--date_online", required=True, help="Date string like 2020_04_11")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0000_0030")
    parser.add_argument("--eps", type=int, default=60, help="DBSCAN eps in seconds (default: 60)")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min_samples (default: 3)")
    parser.add_argument("--output_folder_name", type=str, default="1216",
                        help="Output folder name (e.g., experiment ID)")
    parser.add_argument("--dependency_graphs", default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ2/all_telecom_unique_dependency_graphs.json",
                        help="Path to Telecom dependency graphs JSON file")

    args = parser.parse_args()

    BASE_DIR = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"

    print(f"📁 Loading anomalies for date={args.date_online}, window={args.output_suffix}")
    anomalies = load_anomalies_from_window_new(BASE_DIR, args.date_online, args.output_suffix)
    
    output_file = f"{BASE_DIR}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}.txt"

    print(f"🎯 Total anomalies loaded: {len(anomalies)}")
    cluster_and_report(
        anomalies,
        output_file=output_file,
        dependency_graphs_path=args.dependency_graphs,
        date_str=args.date_online,
        window_str=args.output_suffix,
        eps_seconds=args.eps,
        min_samples=args.min_samples
    )
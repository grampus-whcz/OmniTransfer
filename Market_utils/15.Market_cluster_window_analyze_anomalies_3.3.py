import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
import argparse
import re

# # === 全局配置（复用Bank的RCA权重体系，适配Market调整）===
# CONCENTRATION_WINDOW_MINUTES = 4
# ANOMALY_THRESHOLD = 2
# FALLBACK_THRESHOLD = 1
# WEIGHT_TIME = 0.3
# WEIGHT_TOPOLOGY = 0.4
# WEIGHT_COUNT = 0.3
# # 新增超参数：分析起点的时间戳索引（0=T0(最早),1=T1(第二早),2=T2(第三早)）
# ANALYSIS_START_TIMESTAMP_INDEX = 2

# # config 1: 3
# # === 全局配置（复用Bank的RCA权重体系，适配Market调整）===
# CONCENTRATION_WINDOW_MINUTES = 5
# ANOMALY_THRESHOLD = 1
# FALLBACK_THRESHOLD = 1
# WEIGHT_TIME = 0.3
# WEIGHT_TOPOLOGY = 0.4
# WEIGHT_COUNT = 0.3
# # 新增超参数：分析起点的时间戳索引（0=T0(最早),1=T1(第二早),2=T2(第三早)）
# ANALYSIS_START_TIMESTAMP_INDEX = 1

# config 2: 8
# === 全局配置（复用Bank的RCA权重体系，适配Market调整）===
CONCENTRATION_WINDOW_MINUTES = 5
ANOMALY_THRESHOLD = 3
FALLBACK_THRESHOLD = 1
WEIGHT_TIME = 0.1
WEIGHT_TOPOLOGY = 0.8
WEIGHT_COUNT = 0.1
# 新增超参数：分析起点的时间戳索引（0=T0(最早),1=T1(第二早),2=T2(第三早)）
ANALYSIS_START_TIMESTAMP_INDEX = 0

# # config 3: 6
# # === 全局配置（复用Bank的RCA权重体系，适配Market调整）===
# CONCENTRATION_WINDOW_MINUTES = 6
# ANOMALY_THRESHOLD = 3
# FALLBACK_THRESHOLD = 2
# WEIGHT_TIME = 0.1
# WEIGHT_TOPOLOGY = 0.8
# WEIGHT_COUNT = 0.1
# # 新增超参数：分析起点的时间戳索引（0=T0(最早),1=T1(第二早),2=T2(第三早)）
# ANALYSIS_START_TIMESTAMP_INDEX = 0

# # config 4: 5
# # === 全局配置（复用Bank的RCA权重体系，适配Market调整）===
# CONCENTRATION_WINDOW_MINUTES = 5
# ANOMALY_THRESHOLD = 3
# FALLBACK_THRESHOLD = 1
# WEIGHT_TIME = 0.1
# WEIGHT_TOPOLOGY = 0.8
# WEIGHT_COUNT = 0.1
# # 新增超参数：分析起点的时间戳索引（0=T0(最早),1=T1(第二早),2=T2(第三早)）
# ANALYSIS_START_TIMESTAMP_INDEX = 2

# # config 5: 6
# # === 全局配置（复用Bank的RCA权重体系，适配Market调整）===
# CONCENTRATION_WINDOW_MINUTES = 5
# ANOMALY_THRESHOLD = 3
# FALLBACK_THRESHOLD = 1
# WEIGHT_TIME = 0.05
# WEIGHT_TOPOLOGY = 0.9
# WEIGHT_COUNT = 0.05
# # 新增超参数：分析起点的时间戳索引（0=T0(最早),1=T1(第二早),2=T2(第三早)）
# ANALYSIS_START_TIMESTAMP_INDEX = 0

# # config 6: 6
# # === 全局配置（复用Bank的RCA权重体系，适配Market调整）===
# CONCENTRATION_WINDOW_MINUTES = 5
# ANOMALY_THRESHOLD = 3
# FALLBACK_THRESHOLD = 2
# WEIGHT_TIME = 0.15
# WEIGHT_TOPOLOGY = 0.7
# WEIGHT_COUNT = 0.15
# # 新增超参数：分析起点的时间戳索引（0=T0(最早),1=T1(第二早),2=T2(第三早)）
# ANALYSIS_START_TIMESTAMP_INDEX = 0

# # config 7: 6
# # === 全局配置（复用Bank的RCA权重体系，适配Market调整）===
# CONCENTRATION_WINDOW_MINUTES = 5
# ANOMALY_THRESHOLD = 3
# FALLBACK_THRESHOLD = 1
# WEIGHT_TIME = 0.15
# WEIGHT_TOPOLOGY = 0.7
# WEIGHT_COUNT = 0.15
# # 新增超参数：分析起点的时间戳索引（0=T0(最早),1=T1(第二早),2=T2(第三早)）
# ANALYSIS_START_TIMESTAMP_INDEX = 0

# # config 8:
# # === 全局配置（复用Bank的RCA权重体系，适配Market调整）===
# CONCENTRATION_WINDOW_MINUTES = 5
# ANOMALY_THRESHOLD = 3
# FALLBACK_THRESHOLD = 2
# WEIGHT_TIME = 0.1
# WEIGHT_TOPOLOGY = 0.8
# WEIGHT_COUNT = 0.1
# # 新增超参数：分析起点的时间戳索引（0=T0(最早),1=T1(第二早),2=T2(第三早)）
# ANALYSIS_START_TIMESTAMP_INDEX = 0

# 北京时区 (UTC+8)
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    BEIJING_TZ = ZoneInfo("Asia/Shanghai")
except ImportError:
    import pytz  # 需要 pip install pytz（适用于旧版本 Python）
    BEIJING_TZ = pytz.timezone('Asia/Shanghai')

def ts_to_beijing_str(ts):
    """将 Unix 时间戳转为北京时间字符串"""
    if isinstance(ts, (int, float)):
        try:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(BEIJING_TZ)
            return dt.strftime("%Y-%m-%d %H:%M:%S") + " CST"
        except:
            return "N/A"
    return "N/A"

# === Market专用RCA分析器（复用Bank逻辑，适配Market拓扑）===
class ClusterBasedMarketRCAAnalyzer:
    def __init__(self):
        # Market微服务分层（适配其拓扑结构）
        self.call_chain_layers = {
            'frontend': ['frontend', 'frontend-0', 'frontend-1', 'frontend-2', 'frontend2-0'],
            'adservice': ['adservice', 'adservice-0', 'adservice-1', 'adservice-2', 'adservice2-0'],
            'cartservice': ['cartservice', 'cartservice-0', 'cartservice-1', 'cartservice-2', 'cartservice2-0'],
            'checkoutservice': ['checkoutservice', 'checkoutservice-0', 'checkoutservice-1', 'checkoutservice-2', 'checkoutservice2-0'],
            'currencyservice': ['currencyservice', 'currencyservice-0', 'currencyservice-1', 'currencyservice-2', 'currencyservice2-0'],
            'recommendationservice': ['recommendationservice', 'recommendationservice-0', 'recommendationservice-1', 'recommendationservice-2', 'recommendationservice2-0'],
            'productcatalogservice': ['productcatalogservice', 'productcatalogservice-0', 'productcatalogservice-1', 'productcatalogservice-2', 'productcatalogservice2-0'],
            'shippingservice': ['shippingservice', 'shippingservice-0', 'shippingservice-1', 'shippingservice-2', 'shippingservice2-0'],
            'emailservice': ['emailservice', 'emailservice-0', 'emailservice-1', 'emailservice-2', 'emailservice2-0'],
            'paymentservice': ['paymentservice', 'paymentservice-0', 'paymentservice-1', 'paymentservice-2', 'paymentservice2-0'],
            'node': ['node-1', 'node-2', 'node-3', 'node-4', 'node-5', 'node-6']
        }
        
        # 分层权重（入口层frontend权重最高，核心业务层次之）
        self.chain_weights = {
            'frontend': 1.0,          # 入口层，权重最高
            'checkoutservice': 0.95,  # 核心结算服务
            'cartservice': 0.9,       # 购物车服务
            'currencyservice': 0.85,  # 货币服务
            'recommendationservice': 0.85, # 推荐服务
            'productcatalogservice': 0.85, # 商品目录服务
            'shippingservice': 0.8,   # 物流服务
            'emailservice': 0.75,     # 邮件服务
            'paymentservice': 0.9,    # 支付服务
            'node': 0.7               # 节点层，权重较低
        }

        # Market基础拓扑边（service-service + service-pod）
        self.base_edges = self._build_market_base_edges()
        
        # 预编译正则表达式，提升清洗效率
        self.regex_patterns = {
            # 匹配 node-x.service-pod 格式（如 node-6.recommendationservice2-0）
            'node_pod': re.compile(r'^node-\d+\.(.*)$'),
            # 匹配 service.ts:port 格式（如 adservice.ts:8088、adservice2.ts:8088）
            'service_port': re.compile(r'^([a-zA-Z0-9]+?)(?:2)?\.ts:\d+$'),
            # 匹配 istio 网关格式（如 istio-egressgateway-7bfdcc9d86-g2d4q）
            'istio_gateway': re.compile(r'^istio-[a-zA-Z]+gateway.*$'),
            # 提取核心service名（去掉数字后缀，如 adservice2 → adservice）
            'service_suffix': re.compile(r'^([a-zA-Z]+?)(?:2)?(-\d+)?$')
        }

    def _clean_entity_name(self, entity):
        """
        清洗实体名，提取核心service/pod名称
        处理格式：
        - node-6.recommendationservice2-0 → recommendationservice2-0
        - adservice.ts:8088 → adservice
        - adservice2.ts:8088 → adservice
        - istio-egressgateway-7bfdcc9d86-g2d4q → istio-egressgateway
        - 其他格式保持不变
        """
        if not isinstance(entity, str):
            return str(entity) if entity is not None else ""
        
        # 1. 处理 node-x.pod 格式
        node_pod_match = self.regex_patterns['node_pod'].match(entity)
        if node_pod_match:
            cleaned = node_pod_match.group(1)
            # 对提取出的pod名进一步清洗（如 recommendationservice2-0 → recommendationservice）
            service_match = self.regex_patterns['service_suffix'].match(cleaned)
            if service_match:
                return service_match.group(1)
            return cleaned
        
        # 2. 处理 service.ts:port 格式
        service_port_match = self.regex_patterns['service_port'].match(entity)
        if service_port_match:
            return service_port_match.group(1)
        
        # 3. 处理 istio 网关格式
        istio_match = self.regex_patterns['istio_gateway'].match(entity)
        if istio_match:
            # 提取网关核心名（如 istio-egressgateway-7bfdcc9d86-g2d4q → istio-egressgateway）
            return entity.split('-')[0] + '-' + entity.split('-')[1]
        
        # 4. 处理普通service/pod名（如 adservice2-0 → adservice）
        service_match = self.regex_patterns['service_suffix'].match(entity)
        if service_match:
            return service_match.group(1)
        
        # 5. 其他格式直接返回
        return entity

    def _build_market_base_edges(self):
        """构建Market的基础拓扑边：service-service调用关系 + service-pod从属关系"""
        base_edges = []
        
        # 1. service-service调用关系（双向，因为依赖是相互的）
        call_relation_list = [
            'frontend -> adservice',
            'frontend -> cartservice',
            'frontend -> checkoutservice',
            'frontend -> currencyservice',
            'frontend -> recommendationservice',
            'frontend -> productcatalogservice',
            'frontend -> shippingservice',
            'checkoutservice -> cartservice',
            'checkoutservice -> currencyservice',
            'checkoutservice -> emailservice',
            'checkoutservice -> paymentservice',
            'checkoutservice -> productcatalogservice',
            'checkoutservice -> shippingservice',
            'recommendationservice -> productcatalogservice',
        ]
        for call_relation in call_relation_list:
            caller, callee = call_relation.split(' -> ')
            base_edges.append((caller, callee))
            base_edges.append((callee, caller))  # 双向边
        
        # 2. service-pod从属关系（双向）
        service_list = [
            'frontend', 'adservice', 'cartservice', 'checkoutservice',
            'currencyservice', 'recommendationservice', 'productcatalogservice',
            'shippingservice', 'emailservice', 'paymentservice'
        ]
        for service in service_list:
            for pod_suffix in ['-0', '-1', '-2', '2-0']:
                pod = f"{service}{pod_suffix}"
                base_edges.append((service, pod))
                base_edges.append((pod, service))
        
        # 3. 添加istio网关边（适配istio网关实体）
        istio_gateways = ['istio-egressgateway', 'istio-ingressgateway']
        for gateway in istio_gateways:
            # istio网关与frontend关联（Market架构中网关是入口）
            base_edges.append(('frontend', gateway))
            base_edges.append((gateway, 'frontend'))
        
        return base_edges

    def _identify_service_layer(self, entity):
        """识别实体所属的分层（先清洗实体名）"""
        # 先清洗实体名
        cleaned_entity = self._clean_entity_name(entity)
        
        for layer, instances in self.call_chain_layers.items():
            # 检查原始实例或清洗后的实例是否匹配
            if entity in instances or cleaned_entity in instances:
                return layer
        # 模糊匹配（处理可能的实体名变体）
        entity_lower = cleaned_entity.lower()
        if 'frontend' in entity_lower:
            return 'frontend'
        elif 'checkout' in entity_lower:
            return 'checkoutservice'
        elif 'cart' in entity_lower:
            return 'cartservice'
        elif 'node-' in cleaned_entity:
            return 'node'
        elif 'adservice' in entity_lower:
            return 'adservice'
        elif 'recommendationservice' in entity_lower:
            return 'recommendationservice'
        elif 'istio' in entity_lower:
            return 'frontend'  # istio网关归为frontend层
        return None

    def prepare_anomalies_for_rca(self, anomalies):
        """异常数据增强（适配Market数据格式，先清洗实体名）"""
        enriched = []
        for a in anomalies:
            # 清洗实体名
            cleaned_entity = self._clean_entity_name(a['entity'])
            layer = self._identify_service_layer(cleaned_entity)
            if layer:
                enriched.append({
                    'entity': a['entity'],
                    'cleaned_entity': cleaned_entity,  # 新增清洗后的实体名
                    'attribute': a['attribute'],
                    'layer': layer,
                    'layer_weight': self.chain_weights.get(layer, 0.5),
                    'timestamp': a['ts'],
                    'ts': a['ts']
                })
        return enriched

    def build_topology_graph(self, edge_anomalies):
        """构建Market拓扑图（适配清洗后的实体名）"""
        G = nx.DiGraph()
        G.add_edges_from(self.base_edges)

        # 初始化边属性
        for u, v in G.edges:
            G.edges[u, v]['has_anomaly'] = False
            G.edges[u, v]['anomaly_timestamp'] = None

        # 标记异常边（先清洗实体名）
        for a in edge_anomalies:
            if '->' in a['entity']:
                s, t = a['entity'].split('->')
                # 清洗源和目标实体名
                s_clean = self._clean_entity_name(s)
                t_clean = self._clean_entity_name(t)
                # 优先使用清洗后的实体名构建边
                if not G.has_edge(s_clean, t_clean):
                    G.add_edge(s_clean, t_clean)
                G.edges[s_clean, t_clean]['has_anomaly'] = True
                G.edges[s_clean, t_clean]['anomaly_timestamp'] = a['ts']
                # 同时保留原始实体名的边（兼容旧逻辑）
                if not G.has_edge(s, t):
                    G.add_edge(s, t)
                G.edges[s, t]['has_anomaly'] = True
                G.edges[s, t]['anomaly_timestamp'] = a['ts']

        return G

    def calculate_rca_scores(self, anomalies):
        """核心RCA评分计算（复用Bank逻辑，适配Market，增加实体清洗）"""
        try:
            # Step 1: 转换异常数据为DataFrame（先清洗实体名）
            node_anomalies = []
            edge_anomalies = []
            for a in anomalies:
                ts_val = int(a['ts']) if isinstance(a['ts'], (int, float)) else 0
                # 清洗实体名
                cleaned_entity = self._clean_entity_name(a['entity'])
                if '->' in a['entity']:
                    # 边异常
                    s, t = a['entity'].split('->')
                    s_clean = self._clean_entity_name(s)
                    t_clean = self._clean_entity_name(t)
                    edge_anomalies.append({
                        'source': s,
                        'source_clean': s_clean,  # 新增清洗后的源
                        'target': t,
                        'target_clean': t_clean,  # 新增清洗后的目标
                        'attr': a['attribute'],
                        'timestamp': ts_val,
                        'ts': ts_val,
                        'time_str': ts_to_beijing_str(ts_val)
                    })
                else:
                    # 节点异常
                    node_anomalies.append({
                        'entity': a['entity'],
                        'entity_clean': cleaned_entity,  # 新增清洗后的实体名
                        'attr': a['attribute'],
                        'timestamp': ts_val,
                        'ts': ts_val,
                        'time_str': ts_to_beijing_str(ts_val)
                    })

            df_node = pd.DataFrame(node_anomalies, columns=['entity', 'entity_clean', 'attr', 'timestamp', 'ts', 'time_str'])
            df_edge = pd.DataFrame(edge_anomalies, columns=['source', 'source_clean', 'target', 'target_clean', 'attr', 'timestamp', 'ts', 'time_str'])

            # Step 2: 筛选时间窗口内的异常（适配分析起点超参数）
            all_ts = []
            if not df_node.empty:
                all_ts.extend(df_node['timestamp'].dropna().astype(int).tolist())
            if not df_edge.empty:
                all_ts.extend(df_edge['timestamp'].dropna().astype(int).tolist())
            
            if not all_ts:
                return None
            
            # 选择分析起点（T0/T1/T2）
            sorted_ts = sorted(list(set(all_ts)))
            if ANALYSIS_START_TIMESTAMP_INDEX >= len(sorted_ts):
                T_start = sorted_ts[-1] if sorted_ts else 0
                print(f"⚠️  ANALYSIS_START_TIMESTAMP_INDEX={ANALYSIS_START_TIMESTAMP_INDEX} exceeds available timestamps ({len(sorted_ts)}), fallback to latest timestamp: {T_start}")
            else:
                T_start = sorted_ts[ANALYSIS_START_TIMESTAMP_INDEX] if sorted_ts else 0
            
            T_window = T_start + CONCENTRATION_WINDOW_MINUTES * 60

            df_node_window = df_node[df_node['timestamp'] <= T_window].copy() if not df_node.empty else df_node
            df_edge_window = df_edge[df_edge['timestamp'] <= T_window].copy() if not df_edge.empty else df_edge

            # Step 3: 统计每个实体的异常数量（同时统计原始和清洗后的实体）
            # 优先使用清洗后的实体名统计
            node_count = pd.DataFrame(columns=['entity', 'entity_clean', 'node_anomaly_count'])
            if not df_node_window.empty:
                node_count = df_node_window.groupby(['entity', 'entity_clean']).size().reset_index(name='node_anomaly_count')

            edge_count = pd.DataFrame(columns=['entity', 'entity_clean', 'edge_anomaly_count'])
            if not df_edge_window.empty:
                # 统计清洗后的源实体
                source_count = df_edge_window.groupby(['source', 'source_clean']).size().reset_index(name='edge_anomaly_count')
                source_count.columns = ['entity', 'entity_clean', 'edge_anomaly_count']
                # 统计清洗后的目标实体
                target_count = df_edge_window.groupby(['target', 'target_clean']).size().reset_index(name='edge_anomaly_count')
                target_count.columns = ['entity', 'entity_clean', 'edge_anomaly_count']
                edge_count = pd.concat([source_count, target_count], ignore_index=True)
                if not edge_count.empty:
                    edge_count = edge_count.groupby(['entity', 'entity_clean'])['edge_anomaly_count'].sum().reset_index()

            # 合并统计结果（优先使用清洗后的实体名）
            all_entities = set()
            all_clean_entities = set()
            if not node_count.empty:
                all_entities.update(node_count['entity'].tolist())
                all_clean_entities.update(node_count['entity_clean'].tolist())
            if not edge_count.empty:
                all_entities.update(edge_count['entity'].tolist())
                all_clean_entities.update(edge_count['entity_clean'].tolist())
            
            # 构建实体映射（原始→清洗后）
            entity_mapping = {}
            if not node_count.empty:
                for _, row in node_count.iterrows():
                    entity_mapping[row['entity']] = row['entity_clean']
            if not edge_count.empty:
                for _, row in edge_count.iterrows():
                    entity_mapping[row['entity']] = row['entity_clean']

            # 合并原始和清洗后的实体统计
            df_entity = pd.DataFrame({
                'entity': list(all_entities),
                'entity_clean': [entity_mapping.get(e, e) for e in all_entities]
            })
            # 合并节点异常统计
            df_entity = df_entity.merge(
                node_count[['entity', 'node_anomaly_count']], 
                on='entity', 
                how='left'
            ).fillna({'node_anomaly_count': 0})
            # 合并边异常统计
            df_entity = df_entity.merge(
                edge_count[['entity', 'edge_anomaly_count']], 
                on='entity', 
                how='left'
            ).fillna({'edge_anomaly_count': 0})
            # 总异常数
            df_entity['total_anomaly_window'] = df_entity['node_anomaly_count'] + df_entity['edge_anomaly_count']

            # 提取异常属性信息
            entity_attrs = defaultdict(Counter)
            if not df_node_window.empty:
                for _, row in df_node_window.iterrows():
                    entity_attrs[row['entity']][row['attr']] += 1
                    # 同时统计清洗后的实体
                    entity_attrs[row['entity_clean']][row['attr']] += 1
            if not df_edge_window.empty:
                for _, row in df_edge_window.iterrows():
                    entity_attrs[row['source']][row['attr']] += 1
                    entity_attrs[row['target']][row['attr']] += 1
                    # 同时统计清洗后的实体
                    entity_attrs[row['source_clean']][row['attr']] += 1
                    entity_attrs[row['target_clean']][row['attr']] += 1
            
            df_entity['anomaly_attrs'] = df_entity['entity'].apply(lambda x: dict(entity_attrs.get(x, {})))
            df_entity['attr_count'] = df_entity['anomaly_attrs'].apply(lambda x: len(x))

            # 筛选候选实体
            df_candidate = df_entity[df_entity['total_anomaly_window'] >= ANOMALY_THRESHOLD].copy()
            if df_candidate.empty:
                df_candidate = df_entity[df_entity['total_anomaly_window'] >= FALLBACK_THRESHOLD].copy()

            # Step 4: 构建拓扑图
            G = self.build_topology_graph(anomalies)

            # Step 5: 增强候选实体特征（适配清洗后的实体名）
            df_enriched = self.enrich_candidate_features(df_candidate, df_node_window, df_edge_window, G, T_start, T_window)
            
            # Step 6: 计算最终评分
            df_scored = self.calculate_final_scores(df_enriched)
            
            return {
                'scored_candidates': df_scored,
                'topology_graph': G,
                'time_window': (T_start, T_window),
                'total_anomalies': len(anomalies),  # 修复：使用输入的异常总数，更准确
                'analysis_start_ts': T_start,
                'entity_mapping': entity_mapping  # 新增实体映射
            }
        except Exception as e:
            print(f"❌ Error in calculate_rca_scores: {str(e)} (type: {type(e).__name__})")
            import traceback
            traceback.print_exc()
            return None

    def enrich_candidate_features(self, df_candidate, df_node_window, df_edge_window, G, T_start, T_window):
        """增强候选实体特征（时间+拓扑，适配清洗后的实体名）"""
        df_enriched = df_candidate.copy()

        # 计算最早异常时间（增加空值兜底）
        node_earliest = pd.DataFrame(columns=['entity', 'entity_clean', 'node_earliest_ts'])
        if not df_node_window.empty and 'timestamp' in df_node_window.columns:
            node_earliest = df_node_window.groupby(['entity', 'entity_clean'])['timestamp'].min().reset_index(name='node_earliest_ts')

        edge_earliest = pd.DataFrame(columns=['entity', 'entity_clean', 'edge_earliest_ts'])
        if not df_edge_window.empty and 'timestamp' in df_edge_window.columns:
            # 源实体最早时间
            source_earliest = df_edge_window.groupby(['source', 'source_clean'])['timestamp'].min().reset_index(name='edge_earliest_ts')
            source_earliest.columns = ['entity', 'entity_clean', 'edge_earliest_ts']
            # 目标实体最早时间
            target_earliest = df_edge_window.groupby(['target', 'target_clean'])['timestamp'].min().reset_index(name='edge_earliest_ts')
            target_earliest.columns = ['entity', 'entity_clean', 'edge_earliest_ts']
            edge_earliest = pd.concat([source_earliest, target_earliest], ignore_index=True)
            if not edge_earliest.empty:
                edge_earliest = edge_earliest.groupby(['entity', 'entity_clean'])['edge_earliest_ts'].min().reset_index()

        # 合并最早时间
        df_enriched = df_enriched.merge(node_earliest[['entity', 'node_earliest_ts']], on='entity', how='left').fillna({'node_earliest_ts': np.inf})
        df_enriched = df_enriched.merge(edge_earliest[['entity', 'edge_earliest_ts']], on='entity', how='left').fillna({'edge_earliest_ts': np.inf})

        # 计算综合最早时间
        def get_earliest(row):
            ts_list = [row['node_earliest_ts'], row['edge_earliest_ts']]
            ts_list = [ts for ts in ts_list if ts != np.inf and not pd.isna(ts) and isinstance(ts, (int, float))]
            return min(ts_list) if ts_list else None
        
        df_enriched['earliest_timestamp'] = df_enriched.apply(get_earliest, axis=1)

        # 拓扑特征（入度/出度/可达节点数，优先使用清洗后的实体名）
        def get_reachable(entity):
            # 先清洗实体名
            cleaned_entity = self._clean_entity_name(entity)
            # 优先检查清洗后的实体名
            if cleaned_entity in G.nodes:
                try:
                    reachable_nodes = nx.descendants(G, cleaned_entity)
                    candidate_entities = df_enriched['entity'].tolist()
                    # 同时检查原始和清洗后的候选实体
                    candidate_clean_entities = [self._clean_entity_name(e) for e in candidate_entities]
                    return len([n for n in reachable_nodes if n in candidate_entities or n in candidate_clean_entities]) + 1
                except:
                    pass
            # 再检查原始实体名
            if entity in G.nodes:
                try:
                    reachable_nodes = nx.descendants(G, entity)
                    candidate_entities = df_enriched['entity'].tolist()
                    return len([n for n in reachable_nodes if n in candidate_entities]) + 1
                except:
                    pass
            return 0

        # 计算拓扑特征（适配清洗后的实体名）
        def get_degree(entity, degree_type='in'):
            cleaned_entity = self._clean_entity_name(entity)
            if cleaned_entity in G.nodes:
                return G.in_degree(cleaned_entity) if degree_type == 'in' else G.out_degree(cleaned_entity)
            if entity in G.nodes:
                return G.in_degree(entity) if degree_type == 'in' else G.out_degree(entity)
            return 0

        df_enriched['in_degree'] = df_enriched['entity'].apply(lambda x: get_degree(x, 'in'))
        df_enriched['out_degree'] = df_enriched['entity'].apply(lambda x: get_degree(x, 'out'))
        df_enriched['reachable_count'] = df_enriched['entity'].apply(get_reachable)

        return df_enriched

    def calculate_final_scores(self, df_enriched):
        """计算加权最终评分"""
        try:
            df_scored = df_enriched.copy()
            # 过滤有效行
            valid_mask = (
                df_scored['earliest_timestamp'].notna() & 
                (df_scored['earliest_timestamp'] != np.inf) &
                df_scored['earliest_timestamp'].apply(lambda x: isinstance(x, (int, float)))
            )
            df_scored = df_scored[valid_mask].copy()
            
            if df_scored.empty:
                print("⚠️  No valid scored candidates")
                return None

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
            
            df_scored['in_degree_score'] = 1 - (df_scored['in_degree'] / max_in)
            df_scored['out_degree_score'] = df_scored['out_degree'] / max_out
            df_scored['reachable_score'] = df_scored['reachable_count'] / max_reach
            df_scored['topology_score'] = (df_scored['in_degree_score'] + df_scored['out_degree_score'] + df_scored['reachable_score']) / 3

            # 数量得分
            max_count = df_scored['total_anomaly_window'].max() if df_scored['total_anomaly_window'].max() > 0 else 1
            df_scored['count_score'] = df_scored['total_anomaly_window'] / max_count

            # 最终加权得分
            df_scored['final_score'] = (
                WEIGHT_TIME * df_scored['time_score'] +
                WEIGHT_TOPOLOGY * df_scored['topology_score'] +
                WEIGHT_COUNT * df_scored['count_score']
            )

            # 排序并保留两位小数
            df_scored = df_scored.sort_values('final_score', ascending=False).reset_index(drop=True)
            score_cols = ['time_score', 'topology_score', 'count_score', 'final_score']
            df_scored[score_cols] = df_scored[score_cols].round(2)

            return df_scored
        except Exception as e:
            print(f"❌ Error in calculate_final_scores: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_dependency_rca_report(self, cluster_id, anomalies):
        """生成Market的RCA报告（修复依赖路径分析）"""
        try:
            # 增强异常数据
            enriched_anomalies = self.prepare_anomalies_for_rca(anomalies)
            if not enriched_anomalies:
                return f"Cluster #{cluster_id}: No valid anomalies for dependency-based RCA analysis."

            # 运行RCA分析
            rca_results = self.calculate_rca_scores(anomalies)
            if not rca_results or rca_results['scored_candidates'] is None:
                return self._generate_fallback_report(cluster_id, anomalies, enriched_anomalies)

            # 提取结果（修复KeyError + 增加边界检查）
            df_scored = rca_results['scored_candidates']
            G = rca_results['topology_graph']
            T_start, T_window = rca_results['time_window']
            # 核心修复：使用正确的键名 + get方法兜底
            total_anomalies = rca_results.get('total_anomalies', len(anomalies))
            analysis_start_ts = rca_results.get('analysis_start_ts', T_start)
            entity_mapping = rca_results.get('entity_mapping', {})

            # 格式化时间
            t_start_str = ts_to_beijing_str(T_start)
            tw_str = ts_to_beijing_str(T_window)
            start_type = f"T{ANALYSIS_START_TIMESTAMP_INDEX}" if ANALYSIS_START_TIMESTAMP_INDEX <=2 else "Latest"

            # 构建报告
            report = []
            report.append(f"### Root Cause Analysis Report - Cluster #{cluster_id} (Market Dataset)")
            report.append(f"**Analysis Window**: {t_start_str} ({start_type}) to {tw_str} ({CONCENTRATION_WINDOW_MINUTES} minutes)")
            report.append(f"**Total Anomalies**: {total_anomalies}")
            report.append("")

            # 主根因
            root_cause = df_scored.iloc[0]['entity']
            root_clean = entity_mapping.get(root_cause, self._clean_entity_name(root_cause))
            root_score = df_scored.iloc[0]['final_score']
            root_layer = self._identify_service_layer(root_clean) or "unknown"
            root_attrs = df_scored.iloc[0]['anomaly_attrs']
            root_attrs_str = ', '.join([f"{k}({v} times)" for k, v in root_attrs.items()]) if root_attrs else "None"
            
            report.append(f"#### Primary Root Cause Entity")
            report.append(f"- **Entity**: {root_cause} (Cleaned: {root_clean}, Layer: {root_layer.upper()})")
            report.append(f"- **Confidence Score**: {root_score:.2f} (1.0 = highest)")
            report.append(f"- **Anomaly Count**: {int(df_scored.iloc[0]['total_anomaly_window'])}")
            report.append(f"- **Anomaly Attributes**: {root_attrs_str}")
            
            earliest_ts = df_scored.iloc[0]['earliest_timestamp']
            if earliest_ts and not np.isinf(earliest_ts):
                earliest_str = ts_to_beijing_str(earliest_ts)
                report.append(f"- **Earliest Anomaly Time**: {earliest_str}")
            else:
                report.append(f"- **Earliest Anomaly Time**: N/A")
                
            report.append(f"- **Topology Impact**: {df_scored.iloc[0]['reachable_count']} downstream entities affected")
            report.append("")

            # 得分拆解
            report.append(f"#### Score Breakdown for {root_cause} (Cleaned: {root_clean})")
            report.append(f"- **Time Score**: {df_scored.iloc[0]['time_score']:.2f} (earlier anomaly = higher score)")
            report.append(f"- **Topology Score**: {df_scored.iloc[0]['topology_score']:.2f} (based on in-degree/out-degree/reachability)")
            report.append(f"- **Anomaly Count Score**: {df_scored.iloc[0]['count_score']:.2f} (more anomalies = higher score)")
            report.append("")

            # 次要根因
            if len(df_scored) > 1:
                report.append(f"#### Secondary Root Cause Candidates (Top 3)")
                for idx in range(1, min(4, len(df_scored))):
                    row = df_scored.iloc[idx]
                    entity = row['entity']
                    entity_clean = entity_mapping.get(entity, self._clean_entity_name(entity))
                    score = row['final_score']
                    layer = self._identify_service_layer(entity_clean) or "unknown"
                    anomaly_count = int(row['total_anomaly_window'])
                    anomaly_attrs = row['anomaly_attrs']
                    attrs_str = ', '.join([f"{k}({v} times)" for k, v in anomaly_attrs.items()]) if anomaly_attrs else "None"
                    report.append(f"- **{entity}** (Cleaned: {entity_clean}, Layer: {layer.upper()}): Score = {score:.2f}, Anomalies = {anomaly_count}, Attrs = {attrs_str}")
                report.append("")

            # 依赖路径分析（修复核心问题）
            report.append(f"#### Dependency Impact Path")
            # 优先使用清洗后的根因实体名
            root_entity_to_check = root_clean if root_clean in G.nodes else root_cause
            if root_entity_to_check in G.nodes:
                try:
                    reachable_nodes = list(nx.descendants(G, root_entity_to_check))[:5]
                    if reachable_nodes:
                        # 格式化可达节点（显示原始名+清洗名）
                        reachable_str = []
                        for node in reachable_nodes:
                            original_nodes = [k for k, v in entity_mapping.items() if v == node]
                            if original_nodes:
                                reachable_str.append(f"{node} (Original: {', '.join(original_nodes[:2])})")
                            else:
                                reachable_str.append(node)
                        report.append(f"- **Downstream Entities Affected by {root_cause} (Cleaned: {root_clean})**: {', '.join(reachable_str)}")
                    else:
                        report.append(f"- **{root_clean} has no direct downstream dependencies in the topology**")
                except Exception as e:
                    report.append(f"- **Unable to calculate downstream impact for {root_clean}: {str(e)[:50]}...**")
            else:
                # 兜底提示
                report.append(f"- **{root_cause} (Cleaned: {root_clean}) not found in topology graph**")
                report.append(f"  - Suggestion: Check if {root_clean} is a valid Market microservice/pod name")
            report.append("")

            # 分层分析
            report.append(f"#### Layer-wise Impact Analysis")
            layer_counts = defaultdict(int)
            for _, row in df_scored.iterrows():
                entity_clean = entity_mapping.get(row['entity'], self._clean_entity_name(row['entity']))
                layer = self._identify_service_layer(entity_clean) or "unknown"
                layer_counts[layer] += 1
            for layer, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
                weight = self.chain_weights.get(layer, 0.5)
                report.append(f"- **{layer.upper()} Layer**: {count} entities affected (Weight: {weight})")
            report.append("")

            # 建议
            report.append(f"#### RCA Recommendations")
            report.append(f"1. Prioritize investigation of {root_cause} (Cleaned: {root_clean}, highest confidence score)")
            report.append(f"2. Check {root_cause} for {df_scored.iloc[0]['total_anomaly_window']} anomalies on attributes: {root_attrs_str}")
            if earliest_ts and not np.isinf(earliest_ts):
                report.append(f"3. Investigate anomalies starting at {earliest_str} for {root_cause} (analysis window starts at {t_start_str} ({start_type}))")
            else:
                report.append(f"3. Investigate {root_cause} within analysis window {t_start_str} ({start_type}) to {tw_str}")
            report.append(f"4. Verify downstream dependencies of {root_clean} for cascading failures")
            report.append(f"5. Check {root_layer.upper()} layer infrastructure (network, resources, configuration)")

            return "\n".join(report)
        except Exception as e:
            print(f"❌ Error in generate_dependency_rca_report: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_report(cluster_id, anomalies, self.prepare_anomalies_for_rca(anomalies))

    def _generate_fallback_report(self, cluster_id, anomalies, enriched_anomalies):
        """兜底报告（适配实体清洗）"""
        report = []
        report.append(f"### Root Cause Analysis Report - Cluster #{cluster_id} (Fallback Analysis)")
        report.append(f"**Total Anomalies**: {len(anomalies)}")
        report.append("")

        # 归集异常属性（同时统计原始和清洗后的实体）
        entity_attrs = defaultdict(Counter)
        entity_mapping = {}
        for a in anomalies:
            cleaned_entity = self._clean_entity_name(a['entity'])
            entity_mapping[a['entity']] = cleaned_entity
            entity_attrs[a['entity']][a['attribute']] += 1
            entity_attrs[cleaned_entity][a['attribute']] += 1

        # 分层分析（使用清洗后的实体名）
        layer_weights = defaultdict(float)
        for anno in enriched_anomalies:
            try:
                layer = anno['layer']
                layer_weights[layer] += anno.get('layer_weight', 0.5)
            except:
                continue
        
        top_layer = ""
        if layer_weights:
            top_layer = sorted(layer_weights.items(), key=lambda x: x[1], reverse=True)[0][0]
        else:
            top_layer = "unknown"

        # 提取顶层实体（优先使用清洗后的实体名）
        top_entity = max(entity_attrs.keys(), key=lambda x: sum(entity_attrs[x].values()), default="unknown")
        top_clean_entity = entity_mapping.get(top_entity, self._clean_entity_name(top_entity))
        top_entity_attrs = entity_attrs.get(top_entity, {})
        top_attrs_str = ', '.join([f"{k}({v} times)" for k, v in top_entity_attrs.items()]) if top_entity_attrs else "None"

        # 构建报告
        report.append(f"#### Primary Root Cause Hypothesis")
        layer_mapping = {
            'frontend': 'FRONTEND layer (entry point for all requests)',
            'checkoutservice': 'CHECKOUT layer (core payment processing)',
            'cartservice': 'CART layer (shopping cart management)',
            'node': 'NODE layer (infrastructure nodes)',
            'unknown': 'UNKNOWN layer (unclassified entity)'
        }
        report.append(f"- **Likely Root Cause Layer**: {layer_mapping.get(top_layer, top_layer.upper() + ' layer')}")
        report.append(f"- **Top Anomaly Entity**: {top_entity} (Cleaned: {top_clean_entity}, Anomaly Attrs: {top_attrs_str})")
        report.append("")

        # 异常分布
        report.append(f"#### Anomaly Distribution (Top Entities)")
        sorted_entities = sorted(entity_attrs.items(), key=lambda x: sum(x[1].values()), reverse=True)[:3]
        for entity, attrs in sorted_entities:
            cleaned_entity = entity_mapping.get(entity, self._clean_entity_name(entity))
            total = sum(attrs.values())
            attrs_str = ', '.join([f"{k}({v} times)" for k, v in attrs.items()])
            report.append(f"- **{entity}** (Cleaned: {cleaned_entity}): {total} anomalies (Attrs: {attrs_str})")
        report.append("")

        # 建议
        report.append(f"#### Recommendations")
        report.append(f"1. Investigate {top_layer.upper()} layer for anomalies")
        report.append(f"2. Focus on {top_entity} (Cleaned: {top_clean_entity}) with anomalies on attributes: {top_attrs_str}")
        report.append(f"3. Check resource utilization (CPU, memory, disk) for {top_layer.upper()} layer entities")
        report.append(f"4. Verify connectivity between {top_layer.upper()} layer and dependent services")

        return "\n".join(report)

    def analyze_with_rca_engine(self, cluster_id, anomalies):
        """主分析方法"""
        try:
            return self.generate_dependency_rca_report(cluster_id, anomalies)
        except Exception as e:
            print(f"⚠️  Dependency analysis failed for cluster {cluster_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            enriched = self.prepare_anomalies_for_rca(anomalies)
            return self._generate_fallback_report(cluster_id, anomalies, enriched)

# === 原有数据加载逻辑（保持不变）===
def load_anomalies_from_window_big_new(base_dir, date_str, window_str):
    """
    从全量 24 小时异常文件（_0000_2400.npy）中加载数据，
    并根据 window_str 指定的时间段进行过滤。
    """
    anomalies = []

    # 解析时间窗口
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

    # 加载全量文件
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

    # 时间范围过滤
    if start_ts is not None and end_ts is not None:
        anomalies = [a for a in anomalies if start_ts <= a['ts'] < end_ts]
        print(f"🕒 After time window [{window_str}] filtering: {len(anomalies)} anomalies")

    # 按类型分组处理
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

    # 去重
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

def extract_keywords(template):
    """从 log 模板中提取关键故障词"""
    if template is None or not isinstance(template, str):
        return []
    
    keywords = set()
    t_low = template.lower()
    
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

# === 重构的聚类和报告生成逻辑（集成RCA）===
def cluster_and_report(anomalies, output_file, eps_seconds=300, min_samples=2, args=None):
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

    # 初始化Market RCA分析器
    rca_analyzer = ClusterBasedMarketRCAAnalyzer()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Market Anomaly Clustering and Root Cause Analysis Report\n")
        f.write(f"📅 Date: {args.date_online} | Window: {args.output_suffix}\n")
        f.write(f"⚙️  Analysis Start Timestamp Index: T{ANALYSIS_START_TIMESTAMP_INDEX}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"📊 Found {len(clusters)} anomaly clusters (DBSCAN: eps={eps_seconds}s, min_samples={min_samples})\n")
        f.write("📈 Analysis Type: Dependency-based Root Cause Analysis (Topology + Time + Count)\n")
        f.write("=" * 80 + "\n\n")

        cluster_ids = sorted(clusters.keys())
        for idx, cid in enumerate(cluster_ids):
            cluster = clusters[cid]
            ts_vals = [a['ts'] for a in cluster]
            start_ts, end_ts = min(ts_vals), max(ts_vals)
            duration = end_ts - start_ts

            # 基础聚类信息
            f.write(f"# Cluster #{idx + 1}\n")
            f.write(f"**Time Span**: {ts_to_beijing_str(start_ts)} → {ts_to_beijing_str(end_ts)} (Duration: {duration} seconds)\n")
            f.write(f"**Total Anomalies**: {len(cluster)}\n")

            # 提取关键字
            all_keywords = set()
            for a in cluster:
                if a['type'] == 'log':
                    all_keywords.update(extract_keywords(a['raw']))
            if all_keywords:
                f.write(f"**Key Anomaly Patterns**: {', '.join(all_keywords)}\n")
            f.write("\n")

            # 异常类型拆解
            f.write("## Anomaly Breakdown\n")
            grouped = defaultdict(list)
            for a in cluster:
                grouped[a['type']].append(a)

            type_order = [
                'metric_service', 'metric_runtime', 'metric_container',
                'metric_mesh', 'metric_node', 'trace', 'log'
            ]
            for typ in type_order:
                if typ not in grouped:
                    continue
                f.write(f"- **{typ.replace('_', ' ').title()}**: {len(grouped[typ])} anomalies\n")
            f.write("\n")

            # RCA分析
            f.write("## Root Cause Analysis (Dependency-Based)\n")
            rca_report = rca_analyzer.analyze_with_rca_engine(idx + 1, cluster)
            f.write(rca_report)
            f.write("\n")
            f.write("-" * 80 + "\n\n")

        # 孤立异常
        if noise:
            f.write("# Isolated Anomalies (Noise)\n")
            f.write(f"**Total Isolated Anomalies**: {len(noise)}\n")
            for a in sorted(noise, key=lambda x: x['ts']):
                # 显示清洗后的实体名
                cleaned_entity = rca_analyzer._clean_entity_name(a['entity'])
                f.write(f"- {a['type']} | {a['entity']} (Cleaned: {cleaned_entity}) | {a['attribute']} | {ts_to_beijing_str(a['ts'])}\n")
            f.write("\n")

        # 元数据
        f.write("### Analysis Metadata\n")
        f.write(f"- Time Zone: CST (UTC+8)\n")
        f.write(f"- RCA Methodology: Weighted scoring (Time: {WEIGHT_TIME*100}%, Topology: {WEIGHT_TOPOLOGY*100}%, Anomaly Count: {WEIGHT_COUNT*100}%)\n")
        f.write(f"- Analysis Start Timestamp Index: T{ANALYSIS_START_TIMESTAMP_INDEX}\n")
        f.write(f"- Entity Cleaning: Enabled (handles node-x.pod, service.ts:port, istio-gateway formats)\n")
        f.write(f"- Topology Graph: Market microservice call chain (frontend → checkoutservice → cartservice etc.) + service-pod relationships + istio gateway edges\n")

    print(f"✅ Report saved to: {output_file}")
    print(f"📊 Found {len(cluster_ids)} clusters and {len(noise)} isolated anomalies.")

def cluster_and_report_condensed(anomalies, output_file, eps_seconds=300, min_samples=2):
    """精简版报告（保留原有逻辑，集成RCA核心信息，适配实体清洗）"""
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to generate condensed report.")
        return

    X = np.array([[a['ts']] for a in anomalies])
    labels = DBSCAN(eps=eps_seconds, min_samples=min_samples, metric='euclidean').fit_predict(X)

    clusters = defaultdict(list)
    for anomaly, label in zip(anomalies, labels):
        if label != -1:
            clusters[label].append(anomaly)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 初始化RCA分析器
    rca_analyzer = ClusterBasedMarketRCAAnalyzer()

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

            # 按 (type, entity, attribute) 分组（适配清洗后的实体名）
            grouped = defaultdict(list)
            for a in cluster:
                cleaned_entity = rca_analyzer._clean_entity_name(a['entity'])
                key = (a['type'], cleaned_entity, a['attribute'])  # 优先使用清洗后的实体名
                grouped[key].append(a['ts'])

            # 筛选重复项
            repeated_items = {
                k: sorted(set(v)) for k, v in grouped.items() if len(set(v)) >= 2
            }

            if not repeated_items:
                continue

            f.write(f"🚨 Cluster #{idx + 1}\n")
            f.write(f"   Time Span: {ts_to_beijing_str(start_ts)} → {ts_to_beijing_str(end_ts)} (Δ = {duration} sec)\n")
            f.write(f"   Repeated Entity-Attribute Pairs: {len(repeated_items)} (based on cleaned entity names)\n\n")

            # 按类型输出
            type_groups = defaultdict(list)
            for (typ, ent_clean, attr), tss in repeated_items.items():
                # 反向查找原始实体名
                original_entities = [a['entity'] for a in cluster if 
                                    rca_analyzer._clean_entity_name(a['entity']) == ent_clean and 
                                    a['type'] == typ and a['attribute'] == attr]
                original_entities = list(set(original_entities))[:2]  # 最多显示2个原始名
                type_groups[typ].append((ent_clean, original_entities, attr, tss))

            type_order = [
                'metric_service', 'metric_runtime', 'metric_container',
                'metric_mesh', 'metric_node', 'trace', 'log'
            ]
            for typ in type_order:
                if typ not in type_groups:
                    continue
                f.write(f"   📝 {typ.replace('_', ' ').title()} Anomalies:\n")
                for ent_clean, original_ents, attr, tss in sorted(type_groups[typ]):
                    time_repr = ", ".join(f"{ts_to_beijing_str(ts)}" for ts in tss)
                    original_str = f" (Original: {', '.join(original_ents)})" if original_ents else ""
                    f.write(f"     • Entity: {ent_clean}{original_str} | Attribute: {attr}\n")
                    f.write(f"       Times ({len(tss)}): {time_repr}\n")
                    total_condensed_entries += 1
                f.write("\n")

            # 精简版RCA信息（适配清洗后的实体名）
            f.write(f"   🎯 Top Root Cause Hypothesis:\n")
            rca_result = rca_analyzer.calculate_rca_scores(cluster)
            if rca_result and rca_result['scored_candidates'] is not None:
                top_entity = rca_result['scored_candidates'].iloc[0]['entity']
                top_clean = rca_result['entity_mapping'].get(top_entity, rca_analyzer._clean_entity_name(top_entity))
                top_score = rca_result['scored_candidates'].iloc[0]['final_score']
                f.write(f"     • Most Likely Root Cause: {top_entity} (Cleaned: {top_clean}, Score: {top_score:.2f})\n")
            else:
                # 兜底（使用清洗后的实体名）
                entity_attrs = defaultdict(int)
                for a in cluster:
                    cleaned_entity = rca_analyzer._clean_entity_name(a['entity'])
                    entity_attrs[cleaned_entity] += 1
                top_clean = max(entity_attrs.keys(), key=lambda x: entity_attrs[x], default="unknown")
                # 反向查找原始实体名
                original_entities = [a['entity'] for a in cluster if rca_analyzer._clean_entity_name(a['entity']) == top_clean]
                original_entity = original_entities[0] if original_entities else top_clean
                f.write(f"     • Most Likely Root Cause: {original_entity} (Cleaned: {top_clean}, Fallback Analysis)\n")
            f.write("\n")
            f.write("-" * 60 + "\n\n")

        if total_condensed_entries == 0:
            f.write("✅ No repeated (entity, attribute) anomalies found.\n")
        else:
            f.write(f"💡 Total repeated entity-attribute entries: {total_condensed_entries}\n")

        f.write("💡 Note: Only entries with ≥2 distinct timestamps are shown.\n")
        f.write(f"   Clustering: DBSCAN(eps={eps_seconds}s, min_samples={min_samples})\n")
        f.write(f"   RCA Methodology: Weighted scoring (Time: {WEIGHT_TIME*100}%, Topology: {WEIGHT_TOPOLOGY*100}%, Anomaly Count: {WEIGHT_COUNT*100}%)\n")
        f.write(f"   Entity Cleaning: Enabled (handles node-x.pod, service.ts:port, istio-gateway formats)\n")

    print(f"✅ Condensed report saved to: {output_file}")

# === 主程序入口（保持不变）===
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
        min_samples=args.min_samples,
        args=args
    )
    
    # # 生成精简版报告
    # condensed_output_file = f"{BASE_DIR}/Market_cluster_window_anomaly_short_report_{args.date_online}_{args.output_suffix}.txt"
    # cluster_and_report_condensed(
    #     anomalies,
    #     output_file=condensed_output_file,
    #     eps_seconds=args.eps,
    #     min_samples=args.min_samples
    # )
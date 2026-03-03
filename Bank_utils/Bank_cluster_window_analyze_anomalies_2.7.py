import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
import argparse
import sys
import re

# === 1. Global Configuration (From Dependency Analysis) ===
CONCENTRATION_WINDOW_MINUTES = 4
ANOMALY_THRESHOLD = 2
FALLBACK_THRESHOLD = 1
WEIGHT_TIME = 0.3
WEIGHT_TOPOLOGY = 0.4
WEIGHT_COUNT = 0.3
# 新增超参数：分析起点的时间戳索引（0=T0(最早),1=T1(第二早),2=T2(第三早)）
ANALYSIS_START_TIMESTAMP_INDEX = 2

# === 2. RCA Analyzer with Integrated Dependency Logic (Fixed Version) ===
class ClusterBasedBankRCAAnalyzer:
    def __init__(self):
        # Original domain mapping
        self.domain_mapping = {}  # Replace with actual ENHANCED_DOMAIN_MAPPING if available
        total_attrs = sum(
            len(attrs) for mapping in self.domain_mapping.values() for attrs in mapping.values()
        ) if self.domain_mapping else 0
        print(f"🎯 Loaded enhanced domain mapping: {total_attrs} attributes")

        # Call chain layers (original)
        self.call_chain_layers = {
            'entry_point': ['apache01', 'apache02'],
            'gateway': ['IG01', 'IG02'], 
            'business': ['Tomcat01', 'Tomcat02', 'Tomcat03', 'Tomcat04'],
            'governance': ['MG01', 'MG02'],
            'container': ['dockerA1', 'dockerA2', 'dockerB1', 'dockerB2'],
            'database': ['Mysql01', 'Mysql02'],
            'cache': ['Redis01', 'Redis02']
        }
        
        self.chain_weights = {
            'gateway': 1.0,
            'business': 0.9,
            'governance': 0.8,
            'container': 0.7,
            'database': 0.95,
            'cache': 0.6,
            'entry_point': 0.5
        }

        # Base topology graph (from dependency analysis)
        self.base_edges = [
            ("apache01", "IG01"), ("apache01", "IG02"),
            ("apache02", "IG01"), ("apache02", "IG02"),
            ("IG01", "Tomcat02"), ("IG02", "Tomcat02"),
            ("Tomcat02", "MG01"), ("Tomcat02", "MG02"),
            ("MG01", "dockerA2"), ("MG02", "dockerA2"),
            ("dockerA2", "Mysql02"),
            ("Tomcat02", "Redis02"), ("Redis02", "Tomcat02"),
            ("MG01", "Redis02"), ("MG02", "Redis02"), ("Redis02", "MG01"), ("Redis02", "MG02"),
        ]

    def _identify_service_layer(self, entity):
        """Original layer identification logic"""
        for layer, instances in self.call_chain_layers.items():
            for instance in instances:
                if instance in entity:
                    return layer
        if 'ServiceTest' in entity:
            return 'service_test'
        if 'Container-DOCKER_CONTAINER' in entity:
            el = entity.lower()
            if 'mysql' in el:
                return 'database'
            elif 'redis' in el:
                return 'cache'
        el = entity.lower()
        if 'apache' in el:
            return 'entry_point'
        elif 'ig' in el:
            return 'gateway'
        elif 'mg' in el:
            return 'governance'
        elif 'tomcat' in el:
            return 'business'
        elif 'docker' in el:
            return 'container'
        elif 'mysql' in el:
            return 'database'
        elif 'redis' in el:
            return 'cache'
        return None

    def _map_to_pyrca_indicators(self, layer, attribute):
        """Original indicator mapping"""
        if not self.domain_mapping or layer not in self.domain_mapping:
            return [{'indicator': 'avg_cpu', 'layer': layer, 'weight': self.chain_weights.get(layer, 0.5), 'attribute': attribute}]
        indicators = []
        layer_weight = self.chain_weights.get(layer, 0.5)
        for indicator, attrs in self.domain_mapping[layer].items():
            if attribute in attrs:
                indicators.append({
                    'indicator': indicator,
                    'layer': layer,
                    'weight': layer_weight,
                    'attribute': attribute
                })
        if not indicators:
            indicators.append({
                'indicator': 'avg_cpu',
                'layer': layer,
                'weight': layer_weight * 0.5,
                'attribute': attribute
            })
        return indicators

    def prepare_anomalies_for_rca(self, anomalies):
        """Original anomaly enrichment - FIXED: ts → timestamp mapping"""
        enriched = []
        for a in anomalies:
            layer = self._identify_service_layer(a['entity'])
            if layer:
                mapped = self._map_to_pyrca_indicators(layer, a['attribute'])
                for m in mapped:
                    enriched.append({
                        'entity': a['entity'],
                        'attribute': a['attribute'],
                        'layer': layer,
                        'pyrca_indicator': m['indicator'],
                        'layer_weight': m['weight'],
                        'mapped_attribute': m['attribute'],
                        'timestamp': a['ts'],  # FIX: Map ts → timestamp
                        'ts': a['ts']  # Keep original for backward compatibility
                    })
        return enriched

    def build_topology_graph(self, edge_anomalies):
        """Build microservice topology graph (from dependency analysis) - FIXED: ts handling"""
        G = nx.DiGraph()
        G.add_edges_from(self.base_edges)

        # Initialize edge attributes
        for u, v in G.edges:
            G.edges[u, v]['has_anomaly'] = False
            G.edges[u, v]['anomaly_timestamp'] = None

        # Mark anomalous edges - FIX: Use 'ts' instead of 'timestamp'
        for a in edge_anomalies:
            if '->' in a['entity']:
                s, t = a['entity'].split('->')
                if not G.has_edge(s, t):
                    G.add_edge(s, t)
                G.edges[s, t]['has_anomaly'] = True
                G.edges[s, t]['anomaly_timestamp'] = a['ts']  # FIX: Use ts instead of timestamp

        return G

    def calculate_rca_scores(self, anomalies):
        """Core dependency analysis logic - FULLY FIXED for ts/timestamp mapping"""
        try:
            # Step 1: Convert anomalies to DataFrames - FIXED: Use 'ts' from anomaly data
            node_anomalies = []
            edge_anomalies = []
            for a in anomalies:
                # Ensure ts is integer (data type safety)
                ts_val = int(a['ts']) if isinstance(a['ts'], (int, float)) else 0
                if '->' in a['entity']:
                    # Edge anomaly
                    s, t = a['entity'].split('->')
                    edge_anomalies.append({
                        'source': s,
                        'target': t,
                        'attr': a['attribute'],
                        'timestamp': ts_val,  # For DF consistency
                        'ts': ts_val,         # Keep original
                        'time_str': datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    # Node anomaly
                    node_anomalies.append({
                        'entity': a['entity'],
                        'attr': a['attribute'],
                        'timestamp': ts_val,  # For DF consistency
                        'ts': ts_val,         # Keep original
                        'time_str': datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M:%S")
                    })

            df_node = pd.DataFrame(node_anomalies, columns=['entity', 'attr', 'timestamp', 'ts', 'time_str'])
            df_edge = pd.DataFrame(edge_anomalies, columns=['source', 'target', 'attr', 'timestamp', 'ts', 'time_str'])

            # Step 2: Filter concentrated anomalies - FIXED: 适配自定义分析起点超参数
            all_ts = []
            if not df_node.empty:
                all_ts.extend(df_node['timestamp'].dropna().astype(int).tolist())
            if not df_edge.empty:
                all_ts.extend(df_edge['timestamp'].dropna().astype(int).tolist())
            
            if not all_ts:
                return None
            
            # 核心修改：根据超参数选择分析起点（T0/T1/T2）
            sorted_ts = sorted(list(set(all_ts)))  # 去重并排序，避免重复时间戳影响索引
            # 边界处理：若索引超过实际时间戳数量，默认用最后一个（最晚的）有效时间戳
            if ANALYSIS_START_TIMESTAMP_INDEX >= len(sorted_ts):
                T_start = sorted_ts[-1]
                print(f"⚠️  ANALYSIS_START_TIMESTAMP_INDEX={ANALYSIS_START_TIMESTAMP_INDEX} exceeds available timestamps ({len(sorted_ts)}), fallback to latest timestamp: {T_start}")
            else:
                T_start = sorted_ts[ANALYSIS_START_TIMESTAMP_INDEX]  # 0=T0,1=T1,2=T2
            
            T_window = T_start + CONCENTRATION_WINDOW_MINUTES * 60  # 分析窗口：起点 + 3分钟

            df_node_window = df_node[df_node['timestamp'] <= T_window].copy() if not df_node.empty else df_node
            df_edge_window = df_edge[df_edge['timestamp'] <= T_window].copy() if not df_edge.empty else df_edge

            # Step 3: Count anomalies per entity - FIXED: Use 'timestamp' for filtering
            node_count = pd.DataFrame(columns=['entity', 'node_anomaly_count'])
            if not df_node_window.empty:
                node_count = df_node_window.groupby('entity').size().reset_index(name='node_anomaly_count')

            edge_count = pd.DataFrame(columns=['entity', 'edge_anomaly_count'])
            if not df_edge_window.empty:
                source_count = df_edge_window.groupby('source').size().reset_index(name='edge_anomaly_count')
                source_count.columns = ['entity', 'edge_anomaly_count']
                target_count = df_edge_window.groupby('target').size().reset_index(name='edge_anomaly_count')
                target_count.columns = ['entity', 'edge_anomaly_count']
                edge_count = pd.concat([source_count, target_count], ignore_index=True)
                edge_count = edge_count.groupby('entity')['edge_anomaly_count'].sum().reset_index()

            # Merge counts
            all_entities = set()
            if not node_count.empty:
                all_entities.update(node_count['entity'].tolist())
            if not edge_count.empty:
                all_entities.update(edge_count['entity'].tolist())
            all_entities = list(all_entities)

            if not all_entities:
                return None

            df_entity = pd.DataFrame({'entity': all_entities})
            df_entity = df_entity.merge(node_count, on='entity', how='left').fillna({'node_anomaly_count': 0})
            df_entity = df_entity.merge(edge_count, on='entity', how='left').fillna({'edge_anomaly_count': 0})
            df_entity['total_anomaly_window'] = df_entity['node_anomaly_count'] + df_entity['edge_anomaly_count']

            # 新增：提取每个实体的异常属性及数量，用于后续展示
            entity_attrs = defaultdict(Counter)
            # 处理节点异常属性
            if not df_node_window.empty:
                for _, row in df_node_window.iterrows():
                    entity_attrs[row['entity']][row['attr']] += 1
            # 处理边异常属性（合并source/target的属性，按实体归集）
            if not df_edge_window.empty:
                for _, row in df_edge_window.iterrows():
                    entity_attrs[row['source']][row['attr']] += 1
                    entity_attrs[row['target']][row['attr']] += 1
            # 将属性信息转为字典，方便后续调用
            df_entity['anomaly_attrs'] = df_entity['entity'].apply(lambda x: dict(entity_attrs.get(x, {})))
            # 计算每个实体的异常属性数量
            df_entity['attr_count'] = df_entity['anomaly_attrs'].apply(lambda x: len(x))

            # Filter candidates
            df_candidate = df_entity[df_entity['total_anomaly_window'] >= ANOMALY_THRESHOLD].copy()
            if df_candidate.empty:
                df_candidate = df_entity[df_entity['total_anomaly_window'] >= FALLBACK_THRESHOLD].copy()

            # Step 4: Build topology graph
            G = self.build_topology_graph(anomalies)

            # Step 5: Enhance candidate features - FIXED version
            df_enriched = self.enrich_candidate_features(df_candidate, df_node_window, df_edge_window, G, T_start, T_window)
            
            # Step 6: Calculate final scores
            df_scored = self.calculate_final_scores(df_enriched)
            
            return {
                'scored_candidates': df_scored,
                'topology_graph': G,
                'time_window': (T_start, T_window),  # 改为T_start
                'total_anomalies': len(anomalies),
                'analysis_start_ts': T_start  # 新增：返回分析起点，用于报告展示
            }
        except Exception as e:
            print(f"❌ Error in calculate_rca_scores: {str(e)} (type: {type(e).__name__})")
            import traceback
            traceback.print_exc()
            return None

    def enrich_candidate_features(self, df_candidate, df_node_window, df_edge_window, G, T_start, T_window):
        """Enhance candidates with time/topology features - FULLY FIXED"""
        df_enriched = df_candidate.copy()

        # Earliest node anomaly time - FIXED: Use 'timestamp' column
        node_earliest = pd.DataFrame(columns=['entity', 'node_earliest_ts'])
        if not df_node_window.empty and 'timestamp' in df_node_window.columns:
            node_earliest = df_node_window.groupby('entity')['timestamp'].min().reset_index(name='node_earliest_ts')

        # Earliest edge anomaly time - FIXED: Use 'timestamp' column
        edge_earliest = pd.DataFrame(columns=['entity', 'edge_earliest_ts'])
        if not df_edge_window.empty and 'timestamp' in df_edge_window.columns:
            source_earliest = df_edge_window.groupby('source')['timestamp'].min().reset_index(name='edge_earliest_ts')
            source_earliest.columns = ['entity', 'edge_earliest_ts']
            target_earliest = df_edge_window.groupby('target')['timestamp'].min().reset_index(name='edge_earliest_ts')
            target_earliest.columns = ['entity', 'edge_earliest_ts']
            edge_earliest = pd.concat([source_earliest, target_earliest], ignore_index=True)
            edge_earliest = edge_earliest.groupby('entity')['edge_earliest_ts'].min().reset_index()

        # Merge earliest times - FIXED: Handle empty DataFrames
        df_enriched = df_enriched.merge(node_earliest, on='entity', how='left').fillna({'node_earliest_ts': np.inf})
        df_enriched = df_enriched.merge(edge_earliest, on='entity', how='left').fillna({'edge_earliest_ts': np.inf})

        # Calculate comprehensive earliest time - FIXED: Handle inf values
        def get_earliest(row):
            ts_list = [row['node_earliest_ts'], row['edge_earliest_ts']]
            ts_list = [ts for ts in ts_list if ts != np.inf and not pd.isna(ts) and isinstance(ts, (int, float))]
            return min(ts_list) if ts_list else None
        
        df_enriched['earliest_timestamp'] = df_enriched.apply(get_earliest, axis=1)

        # Topology features - FIXED: Handle entities not in graph
        def get_reachable(entity):
            if entity not in G.nodes:
                return 0
            try:
                reachable_nodes = nx.descendants(G, entity)
                candidate_entities = df_enriched['entity'].tolist()
                return len([n for n in reachable_nodes if n in candidate_entities]) + 1
            except:
                return 0

        # Calculate topology metrics with error handling
        df_enriched['in_degree'] = df_enriched['entity'].apply(lambda x: G.in_degree(x) if x in G.nodes else 0)
        df_enriched['out_degree'] = df_enriched['entity'].apply(lambda x: G.out_degree(x) if x in G.nodes else 0)
        df_enriched['reachable_count'] = df_enriched['entity'].apply(get_reachable)

        return df_enriched

    def calculate_final_scores(self, df_enriched):
        """Calculate weighted RCA scores - FIXED: Handle empty DataFrames"""
        try:
            df_scored = df_enriched.copy()
            # Filter valid rows only - FIXED: More robust filtering
            valid_mask = (
                df_scored['earliest_timestamp'].notna() & 
                (df_scored['earliest_timestamp'] != np.inf) &
                df_scored['earliest_timestamp'].apply(lambda x: isinstance(x, (int, float)))
            )
            df_scored = df_scored[valid_mask].copy()
            
            if df_scored.empty:
                print("⚠️  No valid scored candidates")
                return None

            # Time score (earlier = higher) - FIXED: Handle single value case
            t_min = df_scored['earliest_timestamp'].min()
            t_max = df_scored['earliest_timestamp'].max()
            if t_min == t_max:
                df_scored['time_score'] = 1.0
            else:
                df_scored['time_score'] = (t_max - df_scored['earliest_timestamp']) / (t_max - t_min)

            # Topology score - FIXED: Prevent division by zero
            max_in = df_scored['in_degree'].max() if df_scored['in_degree'].max() > 0 else 1
            max_out = df_scored['out_degree'].max() if df_scored['out_degree'].max() > 0 else 1
            max_reach = df_scored['reachable_count'].max() if df_scored['reachable_count'].max() > 0 else 1
            
            df_scored['in_degree_score'] = 1 - (df_scored['in_degree'] / max_in)
            df_scored['out_degree_score'] = df_scored['out_degree'] / max_out
            df_scored['reachable_score'] = df_scored['reachable_count'] / max_reach
            df_scored['topology_score'] = (df_scored['in_degree_score'] + df_scored['out_degree_score'] + df_scored['reachable_score']) / 3

            # Count score - FIXED: Prevent division by zero
            max_count = df_scored['total_anomaly_window'].max() if df_scored['total_anomaly_window'].max() > 0 else 1
            df_scored['count_score'] = df_scored['total_anomaly_window'] / max_count

            # Final weighted score
            df_scored['final_score'] = (
                WEIGHT_TIME * df_scored['time_score'] +
                WEIGHT_TOPOLOGY * df_scored['topology_score'] +
                WEIGHT_COUNT * df_scored['count_score']
            )

            # Sort by final score
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
        """Generate LLM-friendly RCA report with dependency analysis - FIXED: 新增属性列示 + 适配分析起点"""
        try:
            # Step 1: Prepare anomalies
            enriched_anomalies = self.prepare_anomalies_for_rca(anomalies)
            if not enriched_anomalies:
                return f"Cluster #{cluster_id}: No valid anomalies for dependency-based RCA analysis."

            # Step 2: Run dependency analysis
            rca_results = self.calculate_rca_scores(anomalies)
            if not rca_results or rca_results['scored_candidates'] is None:
                # Fallback to original statistical analysis
                return self._generate_fallback_report(cluster_id, anomalies, enriched_anomalies)

            # Step 3: Generate structured report
            df_scored = rca_results['scored_candidates']
            G = rca_results['topology_graph']
            T_start, T_window = rca_results['time_window']
            total_anomalies = rca_results['total_anomalies']
            analysis_start_ts = rca_results['analysis_start_ts']  # 获取分析起点

            # Format time window
            t_start_str = datetime.fromtimestamp(T_start).strftime("%Y-%m-%d %H:%M:%S")
            tw_str = datetime.fromtimestamp(T_window).strftime("%Y-%m-%d %H:%M:%S")
            # 标注分析起点类型（T0/T1/T2）
            start_type = f"T{ANALYSIS_START_TIMESTAMP_INDEX}" if ANALYSIS_START_TIMESTAMP_INDEX <=2 else "Latest"

            # Build report
            report = []
            report.append(f"### Root Cause Analysis Report - Cluster #{cluster_id}")
            report.append(f"**Analysis Window**: {t_start_str} ({start_type}) to {tw_str} (3 minutes)")
            report.append(f"**Total Anomalies**: {total_anomalies}")
            report.append("")

            # Top root cause
            root_cause = df_scored.iloc[0]['entity']
            root_score = df_scored.iloc[0]['final_score']
            root_layer = self._identify_service_layer(root_cause) or "unknown"
            # 根因实体的属性信息
            root_attrs = df_scored.iloc[0]['anomaly_attrs']
            root_attrs_str = ', '.join([f"{k}({v} times)" for k, v in root_attrs.items()]) if root_attrs else "None"
            
            report.append(f"#### Primary Root Cause Entity")
            report.append(f"- **Entity**: {root_cause} (Layer: {root_layer.upper()})")
            report.append(f"- **Confidence Score**: {root_score:.2f} (1.0 = highest)")
            report.append(f"- **Anomaly Count**: {int(df_scored.iloc[0]['total_anomaly_window'])}")
            report.append(f"- **Anomaly Attributes**: {root_attrs_str}")
            # Handle earliest timestamp formatting
            earliest_ts = df_scored.iloc[0]['earliest_timestamp']
            if earliest_ts and not np.isinf(earliest_ts):
                earliest_str = datetime.fromtimestamp(earliest_ts).strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"- **Earliest Anomaly Time**: {earliest_str}")
            else:
                report.append(f"- **Earliest Anomaly Time**: N/A")
                
            report.append(f"- **Topology Impact**: {df_scored.iloc[0]['reachable_count']} downstream entities affected")
            report.append("")

            # Score breakdown for root cause
            report.append(f"#### Score Breakdown for {root_cause}")
            report.append(f"- **Time Score**: {df_scored.iloc[0]['time_score']:.2f} (earlier anomaly = higher score)")
            report.append(f"- **Topology Score**: {df_scored.iloc[0]['topology_score']:.2f} (based on in-degree/out-degree/reachability)")
            report.append(f"- **Anomaly Count Score**: {df_scored.iloc[0]['count_score']:.2f} (more anomalies = higher score)")
            report.append("")

            # Secondary root causes - 核心修改：增加属性列示
            if len(df_scored) > 1:
                report.append(f"#### Secondary Root Cause Candidates (Top 3)")
                for idx in range(1, min(4, len(df_scored))):
                    row = df_scored.iloc[idx]
                    entity = row['entity']
                    score = row['final_score']
                    layer = self._identify_service_layer(entity) or "unknown"
                    anomaly_count = int(row['total_anomaly_window'])
                    # 提取二级根因的异常属性和数量
                    anomaly_attrs = row['anomaly_attrs']
                    # 格式化属性：属性名(异常次数)，多个用逗号分隔
                    attrs_str = ', '.join([f"{k}({v} times)" for k, v in anomaly_attrs.items()]) if anomaly_attrs else "None"
                    # 拼接展示内容
                    report.append(f"- **{entity}** (Layer: {layer.upper()}): Score = {score:.2f}, Anomalies = {anomaly_count}, Attrs = {attrs_str}")
                report.append("")

            # Dependency path analysis
            report.append(f"#### Dependency Impact Path")
            if root_cause in G.nodes:
                try:
                    reachable_nodes = list(nx.descendants(G, root_cause))[:5]  # Top 5 downstream
                    if reachable_nodes:
                        report.append(f"- **Downstream Entities Affected by {root_cause}**: {', '.join(reachable_nodes)}")
                    else:
                        report.append(f"- **{root_cause} has no direct downstream dependencies in the topology**")
                except:
                    report.append(f"- **Unable to calculate downstream impact for {root_cause}**")
            else:
                report.append(f"- **{root_cause} not found in microservice topology graph**")
            report.append("")

            # Layer analysis
            report.append(f"#### Layer-wise Impact Analysis")
            layer_counts = defaultdict(int)
            for _, row in df_scored.iterrows():
                layer = self._identify_service_layer(row['entity']) or "unknown"
                layer_counts[layer] += 1
            for layer, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
                weight = self.chain_weights.get(layer, 0.5)
                report.append(f"- **{layer.upper()} Layer**: {count} entities affected (Weight: {weight})")
            report.append("")

            # Recommendations (LLM-friendly)
            report.append(f"#### RCA Recommendations")
            report.append(f"1. Prioritize investigation of {root_cause} (highest confidence score)")
            report.append(f"2. Check {root_cause} for {df_scored.iloc[0]['total_anomaly_window']} anomalies on attributes: {root_attrs_str}")
            if earliest_ts and not np.isinf(earliest_ts):
                report.append(f"3. Investigate anomalies starting at {earliest_str} for {root_cause} (analysis window starts at {t_start_str} ({start_type}))")
            else:
                report.append(f"3. Investigate {root_cause} within analysis window {t_start_str} ({start_type}) to {tw_str}")
            report.append(f"4. Verify downstream dependencies of {root_cause} for cascading failures")
            report.append(f"5. Check {root_layer.upper()} layer infrastructure (network, resources, configuration)")

            return "\n".join(report)
        except Exception as e:
            print(f"❌ Error in generate_dependency_rca_report: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._generate_fallback_report(cluster_id, anomalies, self.prepare_anomalies_for_rca(anomalies))

    def _generate_fallback_report(self, cluster_id, anomalies, enriched_anomalies):
        """Fallback report when dependency analysis fails - IMPROVED ERROR HANDLING"""
        report = []
        report.append(f"### Root Cause Analysis Report - Cluster #{cluster_id} (Fallback Analysis)")
        report.append(f"**Total Anomalies**: {len(anomalies)}")
        report.append("")

        # 新增：归集异常属性用于兜底报告
        entity_attrs = defaultdict(Counter)
        for a in anomalies:
            entity_attrs[a['entity']][a['attribute']] += 1

        # Statistical analysis with error handling
        indicator_weights = {}
        indicator_counts = {}
        for anno in enriched_anomalies:
            try:
                ind = anno['pyrca_indicator']
                w = anno.get('layer_weight', 0.5)
                indicator_weights[ind] = indicator_weights.get(ind, 0) + w
                indicator_counts[ind] = indicator_counts.get(ind, 0) + 1
            except:
                continue

        # Top indicators
        weighted_indicators = []
        total = len(enriched_anomalies)
        if total > 0:
            for ind, weight in indicator_weights.items():
                try:
                    count = indicator_counts[ind]
                    avg_w = weight / count
                    freq = count / total
                    combined = avg_w * freq
                    weighted_indicators.append({
                        'indicator': ind,
                        'weight': combined,
                        'frequency': freq,
                        'count': count
                    })
                except:
                    continue
        
        weighted_indicators.sort(key=lambda x: x['weight'], reverse=True)
        top_indicator = weighted_indicators[0]['indicator'] if weighted_indicators else "unknown"

        # Layer analysis
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

        # 提取顶层实体的属性
        top_entity = max(entity_attrs.keys(), key=lambda x: sum(entity_attrs[x].values()), default="unknown")
        top_entity_attrs = entity_attrs.get(top_entity, {})
        top_attrs_str = ', '.join([f"{k}({v} times)" for k, v in top_entity_attrs.items()]) if top_entity_attrs else "None"

        # Build fallback report
        report.append(f"#### Primary Root Cause Hypothesis")
        layer_mapping = {
            'database': 'DATABASE layer (critical data storage)',
            'gateway': 'GATEWAY layer (entry point for all requests)',
            'business': 'BUSINESS layer (core application logic)',
            'container': 'CONTAINER layer (infrastructure resources)',
            'cache': 'CACHE layer (performance optimization)',
            'unknown': 'UNKNOWN layer (unclassified entity)'
        }
        report.append(f"- **Likely Root Cause Layer**: {layer_mapping.get(top_layer, top_layer.upper() + ' layer')}")
        report.append(f"- **Top Anomaly Entity**: {top_entity} (Anomaly Attrs: {top_attrs_str})")
        report.append(f"- **High-Impact Indicator**: {top_indicator}")
        report.append("")

        if weighted_indicators:
            report.append(f"#### Anomaly Distribution (Top 3)")
            for item in weighted_indicators[:3]:
                try:
                    report.append(f"- **{item['indicator']}**: {item['count']} occurrences (Frequency: {item['frequency']:.2f})")
                except:
                    continue
            report.append("")

        report.append(f"#### Recommendations")
        report.append(f"1. Investigate {top_layer.upper()} layer for {top_indicator} anomalies")
        report.append(f"2. Focus on {top_entity} with anomalies on attributes: {top_attrs_str}")
        report.append(f"3. Check resource utilization (CPU, memory, disk) for {top_layer.upper()} layer entities")
        report.append(f"4. Verify connectivity between {top_layer.upper()} layer and dependent services")

        return "\n".join(report)

    def analyze_with_rca_engine(self, cluster_id, anomalies):
        """Main analysis method - returns LLM-friendly report - FIXED error handling"""
        try:
            # First try dependency-based analysis
            return self.generate_dependency_rca_report(cluster_id, anomalies)
        except Exception as e:
            print(f"⚠️  Dependency analysis failed for cluster {cluster_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to original PyRCA/statistical analysis
            enriched = self.prepare_anomalies_for_rca(anomalies)
            if not enriched:
                return f"Cluster #{cluster_id}: No valid anomalies for RCA analysis."

            indicator_weights = {}
            indicator_counts = {}
            for anno in enriched:
                try:
                    ind = anno['pyrca_indicator']
                    w = anno.get('layer_weight', 0.5)
                    indicator_weights[ind] = indicator_weights.get(ind, 0) + w
                    indicator_counts[ind] = indicator_counts.get(ind, 0) + 1
                except:
                    continue
            
            weighted_indicators = []
            total = len(enriched)
            if total > 0:
                for ind, weight in indicator_weights.items():
                    try:
                        count = indicator_counts[ind]
                        avg_w = weight / count
                        freq = count / total
                        combined = avg_w * freq
                        if combined > 0.1:
                            weighted_indicators.append({
                                'indicator': ind,
                                'weight': combined,
                                'frequency': freq,
                                'count': count
                            })
                    except:
                        continue
            
            weighted_indicators.sort(key=lambda x: x['weight'], reverse=True)
            top_indicators = [item['indicator'] for item in weighted_indicators[:5]] if weighted_indicators else []
            
            # Try PyRCA engine (original logic) - with error handling
            try:
                from rca import RCAEngine
                bank_domain_knowledge_file = "/path/to/config.yaml"  # Update path
                engine = RCAEngine()
                result = engine.find_root_causes_bn(anomalies=top_indicators, domain_knowledge_file=bank_domain_knowledge_file)
                
                # Format PyRCA results for LLM
                report = []
                report.append(f"### Root Cause Analysis Report - Cluster #{cluster_id} (PyRCA Engine)")
                report.append(f"**Top Indicators**: {', '.join(top_indicators) if top_indicators else 'None'}")
                report.append("")
                report.append(f"#### Detected Root Causes")
                if isinstance(result, list) and result:
                    for cause in result:
                        if isinstance(cause, dict) and 'root_cause' in cause:
                            conf = cause.get('score', 0) * 100
                            report.append(f"- {cause['root_cause']}: {conf:.1f}% confidence")
                        else:
                            report.append(f"- {cause}")
                else:
                    report.append(f"- {result}")
                return "\n".join(report)
            except Exception as pyrca_e:
                print(f"⚠️  PyRCA engine failed: {str(pyrca_e)}")
                # Final fallback
                return self._generate_fallback_report(cluster_id, anomalies, enriched)

# === 3. Original Clustering Logic (Unchanged) ===
BEIJING_TZ = timezone(timedelta(hours=8))

def ts_to_beijing_str(ts):
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(BEIJING_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + " CST"

def load_anomalies_from_window(base_dir, date_str, window_str):
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

    # Initialize RCA analyzer with dependency logic
    rca_analyzer = ClusterBasedBankRCAAnalyzer()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Anomaly Clustering and Root Cause Analysis Report\n")
        f.write(f"📅 Date: {args.date_online} | Window: {args.output_suffix}\n")
        f.write(f"⚙️  Analysis Start Timestamp Index: T{ANALYSIS_START_TIMESTAMP_INDEX}\n")  # 新增：标注超参数
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

            # Basic cluster info
            f.write(f"# Cluster #{idx + 1}\n")
            f.write(f"**Time Span**: {ts_to_beijing_str(start_ts)} → {ts_to_beijing_str(end_ts)} (Duration: {duration} seconds)\n")
            f.write(f"**Total Anomalies**: {len(cluster)}\n")

            # Extract keywords
            all_keywords = set()
            for a in cluster:
                if a['type'] == 'log':
                    all_keywords.update(extract_keywords(a['raw']))
            if all_keywords:
                f.write(f"**Key Anomaly Patterns**: {', '.join(all_keywords)}\n")
            f.write("\n")

            # Anomaly breakdown (simplified for readability)
            f.write("## Anomaly Breakdown\n")
            grouped = defaultdict(list)
            for a in cluster:
                grouped[a['type']].append(a)

            type_order = ['metric_app', 'metric_container', 'trace', 'log']
            for typ in type_order:
                if typ not in grouped:
                    continue
                f.write(f"- **{typ.replace('_', ' ').title()}**: {len(grouped[typ])} anomalies\n")
            f.write("\n")

            # Dependency-based RCA analysis (main integration)
            f.write("## Root Cause Analysis (Dependency-Based)\n")
            rca_report = rca_analyzer.analyze_with_rca_engine(idx + 1, cluster)
            f.write(rca_report)
            f.write("\n")
            f.write("-" * 80 + "\n\n")

        # Noise/anomalous single events
        if noise:
            f.write("# Isolated Anomalies (Noise)\n")
            f.write(f"**Total Isolated Anomalies**: {len(noise)}\n")
            for a in sorted(noise, key=lambda x: x['ts']):
                f.write(f"- {a['type']} | {a['entity']} | {a['attribute']} | {ts_to_beijing_str(a['ts'])}\n")
            f.write("\n")

        # Footer
        f.write("### Analysis Metadata\n")
        f.write(f"- Time Zone: CST (UTC+8)\n")
        f.write(f"- RCA Methodology: Weighted scoring (Time: 40%, Topology: 30%, Anomaly Count: 30%)\n")
        f.write(f"- Analysis Start Timestamp Index: T{ANALYSIS_START_TIMESTAMP_INDEX} (0=T0,1=T1,2=T2)\n")  # 新增：标注超参数
        f.write(f"- Topology Graph: Bank microservice call chain (apache → IG → Tomcat → MG → docker → mysql/redis)\n")

    print(f"✅ Report saved to: {output_file}")
    print(f"📊 Found {len(cluster_ids)} clusters and {len(noise)} isolated anomalies.")

# === 4. Main Program (Unchanged) ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster anomalies and perform dependency-based RCA for Bank dataset.")
    parser.add_argument("--date_online", required=True, help="Date string like 2021_03_04")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0230_0300")
    parser.add_argument("--eps", type=int, default=60, help="DBSCAN eps in seconds (default: 60 = 1 min)")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min_samples (default: 3)")
    parser.add_argument("--output_folder_name", type=str, default="1116",
                        help="Output folder name (e.g., experiment ID)")

    args = parser.parse_args()

    BASE_DIR = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"
    print(f"📁 Loading anomalies for date={args.date_online}, window={args.output_suffix}")
    anomalies = load_anomalies_from_window(BASE_DIR, args.date_online, args.output_suffix)
    
    output_file = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}/Bank_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}.txt"

    print(f"🎯 Total anomalies loaded: {len(anomalies)}")
    cluster_and_report(
        anomalies,
        output_file=output_file,
        eps_seconds=args.eps,
        min_samples=args.min_samples,
        args=args
    )
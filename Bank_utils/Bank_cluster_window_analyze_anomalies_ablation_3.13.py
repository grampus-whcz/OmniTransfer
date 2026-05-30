import os
import json
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
ANALYSIS_START_TIMESTAMP_INDEX = 2

# === 2. RCA Analyzer with Integrated Dependency Logic (Fixed Version) ===
class ClusterBasedBankRCAAnalyzer:
    def __init__(self):
        self.domain_mapping = {}
        total_attrs = sum(
            len(attrs) for mapping in self.domain_mapping.values() for attrs in mapping.values()
        ) if self.domain_mapping else 0
        print(f"🎯 Loaded enhanced domain mapping: {total_attrs} attributes")

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
                        'timestamp': a['ts'],
                        'ts': a['ts']
                    })
        return enriched

    def build_topology_graph(self, edge_anomalies):
        G = nx.DiGraph()
        G.add_edges_from(self.base_edges)

        for u, v in G.edges:
            G.edges[u, v]['has_anomaly'] = False
            G.edges[u, v]['anomaly_timestamp'] = None

        for a in edge_anomalies:
            if '->' in a['entity']:
                s, t = a['entity'].split('->')
                if not G.has_edge(s, t):
                    G.add_edge(s, t)
                G.edges[s, t]['has_anomaly'] = True
                G.edges[s, t]['anomaly_timestamp'] = a['ts']

        return G

    def calculate_rca_scores(self, anomalies):
        try:
            node_anomalies = []
            edge_anomalies = []
            for a in anomalies:
                ts_val = int(a['ts']) if isinstance(a['ts'], (int, float)) else 0
                if '->' in a['entity']:
                    s, t = a['entity'].split('->')
                    edge_anomalies.append({
                        'source': s,
                        'target': t,
                        'attr': a['attribute'],
                        'timestamp': ts_val,
                        'ts': ts_val,
                        'time_str': datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M:%S")
                    })
                else:
                    node_anomalies.append({
                        'entity': a['entity'],
                        'attr': a['attribute'],
                        'timestamp': ts_val,
                        'ts': ts_val,
                        'time_str': datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M:%S")
                    })

            df_node = pd.DataFrame(node_anomalies, columns=['entity', 'attr', 'timestamp', 'ts', 'time_str'])
            df_edge = pd.DataFrame(edge_anomalies, columns=['source', 'target', 'attr', 'timestamp', 'ts', 'time_str'])

            all_ts = []
            if not df_node.empty:
                all_ts.extend(df_node['timestamp'].dropna().astype(int).tolist())
            if not df_edge.empty:
                all_ts.extend(df_edge['timestamp'].dropna().astype(int).tolist())
            
            if not all_ts:
                return None
            
            sorted_ts = sorted(list(set(all_ts)))
            if ANALYSIS_START_TIMESTAMP_INDEX >= len(sorted_ts):
                T_start = sorted_ts[-1]
                print(f"⚠️  ANALYSIS_START_TIMESTAMP_INDEX={ANALYSIS_START_TIMESTAMP_INDEX} exceeds available timestamps ({len(sorted_ts)}), fallback to latest timestamp: {T_start}")
            else:
                T_start = sorted_ts[ANALYSIS_START_TIMESTAMP_INDEX]
            
            T_window = T_start + CONCENTRATION_WINDOW_MINUTES * 60

            df_node_window = df_node[df_node['timestamp'] <= T_window].copy() if not df_node.empty else df_node
            df_edge_window = df_edge[df_edge['timestamp'] <= T_window].copy() if not df_edge.empty else df_edge

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

            entity_attrs = defaultdict(Counter)
            if not df_node_window.empty:
                for _, row in df_node_window.iterrows():
                    entity_attrs[row['entity']][row['attr']] += 1
            if not df_edge_window.empty:
                for _, row in df_edge_window.iterrows():
                    entity_attrs[row['source']][row['attr']] += 1
                    entity_attrs[row['target']][row['attr']] += 1
            df_entity['anomaly_attrs'] = df_entity['entity'].apply(lambda x: dict(entity_attrs.get(x, {})))
            df_entity['attr_count'] = df_entity['anomaly_attrs'].apply(lambda x: len(x))

            df_candidate = df_entity[df_entity['total_anomaly_window'] >= ANOMALY_THRESHOLD].copy()
            if df_candidate.empty:
                df_candidate = df_entity[df_entity['total_anomaly_window'] >= FALLBACK_THRESHOLD].copy()

            G = self.build_topology_graph(anomalies)
            df_enriched = self.enrich_candidate_features(df_candidate, df_node_window, df_edge_window, G, T_start, T_window)
            df_scored = self.calculate_final_scores(df_enriched)
            
            return {
                'scored_candidates': df_scored,
                'topology_graph': G,
                'time_window': (T_start, T_window),
                'total_anomalies': len(anomalies),
                'analysis_start_ts': T_start
            }
        except Exception as e:
            print(f"❌ Error in calculate_rca_scores: {str(e)} (type: {type(e).__name__})")
            import traceback
            traceback.print_exc()
            return None

    def enrich_candidate_features(self, df_candidate, df_node_window, df_edge_window, G, T_start, T_window):
        df_enriched = df_candidate.copy()

        node_earliest = pd.DataFrame(columns=['entity', 'node_earliest_ts'])
        if not df_node_window.empty and 'timestamp' in df_node_window.columns:
            node_earliest = df_node_window.groupby('entity')['timestamp'].min().reset_index(name='node_earliest_ts')

        edge_earliest = pd.DataFrame(columns=['entity', 'edge_earliest_ts'])
        if not df_edge_window.empty and 'timestamp' in df_edge_window.columns:
            source_earliest = df_edge_window.groupby('source')['timestamp'].min().reset_index(name='edge_earliest_ts')
            source_earliest.columns = ['entity', 'edge_earliest_ts']
            target_earliest = df_edge_window.groupby('target')['timestamp'].min().reset_index(name='edge_earliest_ts')
            target_earliest.columns = ['entity', 'edge_earliest_ts']
            edge_earliest = pd.concat([source_earliest, target_earliest], ignore_index=True)
            edge_earliest = edge_earliest.groupby('entity')['edge_earliest_ts'].min().reset_index()

        df_enriched = df_enriched.merge(node_earliest, on='entity', how='left').fillna({'node_earliest_ts': np.inf})
        df_enriched = df_enriched.merge(edge_earliest, on='entity', how='left').fillna({'edge_earliest_ts': np.inf})

        def get_earliest(row):
            ts_list = [row['node_earliest_ts'], row['edge_earliest_ts']]
            ts_list = [ts for ts in ts_list if ts != np.inf and not pd.isna(ts) and isinstance(ts, (int, float))]
            return min(ts_list) if ts_list else None
        
        df_enriched['earliest_timestamp'] = df_enriched.apply(get_earliest, axis=1)

        def get_reachable(entity):
            if entity not in G.nodes:
                return 0
            try:
                reachable_nodes = nx.descendants(G, entity)
                candidate_entities = df_enriched['entity'].tolist()
                return len([n for n in reachable_nodes if n in candidate_entities]) + 1
            except:
                return 0

        df_enriched['in_degree'] = df_enriched['entity'].apply(lambda x: G.in_degree(x) if x in G.nodes else 0)
        df_enriched['out_degree'] = df_enriched['entity'].apply(lambda x: G.out_degree(x) if x in G.nodes else 0)
        df_enriched['reachable_count'] = df_enriched['entity'].apply(get_reachable)

        return df_enriched

    def calculate_final_scores(self, df_enriched):
        try:
            df_scored = df_enriched.copy()
            valid_mask = (
                df_scored['earliest_timestamp'].notna() & 
                (df_scored['earliest_timestamp'] != np.inf) &
                df_scored['earliest_timestamp'].apply(lambda x: isinstance(x, (int, float)))
            )
            df_scored = df_scored[valid_mask].copy()
            
            if df_scored.empty:
                print("⚠️  No valid scored candidates")
                return None

            t_min = df_scored['earliest_timestamp'].min()
            t_max = df_scored['earliest_timestamp'].max()
            if t_min == t_max:
                df_scored['time_score'] = 1.0
            else:
                df_scored['time_score'] = (t_max - df_scored['earliest_timestamp']) / (t_max - t_min)

            max_in = df_scored['in_degree'].max() if df_scored['in_degree'].max() > 0 else 1
            max_out = df_scored['out_degree'].max() if df_scored['out_degree'].max() > 0 else 1
            max_reach = df_scored['reachable_count'].max() if df_scored['reachable_count'].max() > 0 else 1
            
            df_scored['in_degree_score'] = 1 - (df_scored['in_degree'] / max_in)
            df_scored['out_degree_score'] = df_scored['out_degree'] / max_out
            df_scored['reachable_score'] = df_scored['reachable_count'] / max_reach
            df_scored['topology_score'] = (df_scored['in_degree_score'] + df_scored['out_degree_score'] + df_scored['reachable_score']) / 3

            max_count = df_scored['total_anomaly_window'].max() if df_scored['total_anomaly_window'].max() > 0 else 1
            df_scored['count_score'] = df_scored['total_anomaly_window'] / max_count

            df_scored['final_score'] = (
                WEIGHT_TIME * df_scored['time_score'] +
                WEIGHT_TOPOLOGY * df_scored['topology_score'] +
                WEIGHT_COUNT * df_scored['count_score']
            )

            df_scored = df_scored.sort_values('final_score', ascending=False).reset_index(drop=True)
            score_cols = ['time_score', 'topology_score', 'count_score', 'final_score']
            df_scored[score_cols] = df_scored[score_cols].round(2)

            return df_scored
        except Exception as e:
            print(f"❌ Error in calculate_final_scores: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_dependency_rca_report(self, anomalies):
        try:
            enriched_anomalies = self.prepare_anomalies_for_rca(anomalies)
            if not enriched_anomalies:
                return "No valid anomalies for dependency-based RCA analysis."

            rca_results = self.calculate_rca_scores(anomalies)
            if not rca_results or rca_results['scored_candidates'] is None:
                return self._generate_fallback_report(anomalies, enriched_anomalies)

            df_scored = rca_results['scored_candidates']
            G = rca_results['topology_graph']
            T_start, T_window = rca_results['time_window']
            total_anomalies = rca_results['total_anomalies']
            analysis_start_ts = rca_results['analysis_start_ts']

            t_start_str = datetime.fromtimestamp(T_start).strftime("%Y-%m-%d %H:%M:%S")
            tw_str = datetime.fromtimestamp(T_window).strftime("%Y-%m-%d %H:%M:%S")
            start_type = f"T{ANALYSIS_START_TIMESTAMP_INDEX}" if ANALYSIS_START_TIMESTAMP_INDEX <=2 else "Latest"

            report = []
            report.append(f"### Root Cause Analysis Report - Full Time Domain Anomalies")
            report.append(f"**Analysis Window**: {t_start_str} ({start_type}) to {tw_str} (3 minutes)")
            report.append(f"**Total Anomalies**: {total_anomalies}")
            report.append("")

            root_cause = df_scored.iloc[0]['entity']
            root_score = df_scored.iloc[0]['final_score']
            root_layer = self._identify_service_layer(root_cause) or "unknown"
            root_attrs = df_scored.iloc[0]['anomaly_attrs']
            root_attrs_str = ', '.join([f"{k}({v} times)" for k, v in root_attrs.items()]) if root_attrs else "None"
            
            report.append(f"#### Primary Root Cause Entity")
            report.append(f"- **Entity**: {root_cause} (Layer: {root_layer.upper()})")
            report.append(f"- **Confidence Score**: {root_score:.2f} (1.0 = highest)")
            report.append(f"- **Anomaly Count**: {int(df_scored.iloc[0]['total_anomaly_window'])}")
            report.append(f"- **Anomaly Attributes**: {root_attrs_str}")
            earliest_ts = df_scored.iloc[0]['earliest_timestamp']
            if earliest_ts and not np.isinf(earliest_ts):
                earliest_str = datetime.fromtimestamp(earliest_ts).strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"- **Earliest Anomaly Time**: {earliest_str}")
            else:
                report.append(f"- **Earliest Anomaly Time**: N/A")
                
            report.append(f"- **Topology Impact**: {df_scored.iloc[0]['reachable_count']} downstream entities affected")
            report.append("")

            report.append(f"#### Score Breakdown for {root_cause}")
            report.append(f"- **Time Score**: {df_scored.iloc[0]['time_score']:.2f} (earlier anomaly = higher score)")
            report.append(f"- **Topology Score**: {df_scored.iloc[0]['topology_score']:.2f} (based on in-degree/out-degree/reachability)")
            report.append(f"- **Anomaly Count Score**: {df_scored.iloc[0]['count_score']:.2f} (more anomalies = higher score)")
            report.append("")

            if len(df_scored) > 1:
                report.append(f"#### Secondary Root Cause Candidates (Top 3)")
                for idx in range(1, min(4, len(df_scored))):
                    row = df_scored.iloc[idx]
                    entity = row['entity']
                    score = row['final_score']
                    layer = self._identify_service_layer(entity) or "unknown"
                    anomaly_count = int(row['total_anomaly_window'])
                    anomaly_attrs = row['anomaly_attrs']
                    attrs_str = ', '.join([f"{k}({v} times)" for k, v in anomaly_attrs.items()]) if anomaly_attrs else "None"
                    report.append(f"- **{entity}** (Layer: {layer.upper()}): Score = {score:.2f}, Anomalies = {anomaly_count}, Attrs = {attrs_str}")
                report.append("")

            report.append(f"#### Dependency Impact Path")
            if root_cause in G.nodes:
                try:
                    reachable_nodes = list(nx.descendants(G, root_cause))[:5]
                    if reachable_nodes:
                        report.append(f"- **Downstream Entities Affected by {root_cause}**: {', '.join(reachable_nodes)}")
                    else:
                        report.append(f"- **{root_cause} has no direct downstream dependencies in the topology**")
                except:
                    report.append(f"- **Unable to calculate downstream impact for {root_cause}**")
            else:
                report.append(f"- **{root_cause} not found in microservice topology graph**")
            report.append("")

            report.append(f"#### Layer-wise Impact Analysis")
            layer_counts = defaultdict(int)
            for _, row in df_scored.iterrows():
                layer = self._identify_service_layer(row['entity']) or "unknown"
                layer_counts[layer] += 1
            for layer, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
                weight = self.chain_weights.get(layer, 0.5)
                report.append(f"- **{layer.upper()} Layer**: {count} entities affected (Weight: {weight})")
            report.append("")

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
            return self._generate_fallback_report(anomalies, self.prepare_anomalies_for_rca(anomalies))

    def _generate_fallback_report(self, anomalies, enriched_anomalies):
        report = []
        report.append(f"### Root Cause Analysis Report - Full Time Domain Anomalies (Fallback Analysis)")
        report.append(f"**Total Anomalies**: {len(anomalies)}")
        report.append("")

        entity_attrs = defaultdict(Counter)
        for a in anomalies:
            entity_attrs[a['entity']][a['attribute']] += 1

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

        top_entity = max(entity_attrs.keys(), key=lambda x: sum(entity_attrs[x].values()), default="unknown")
        top_entity_attrs = entity_attrs.get(top_entity, {})
        top_attrs_str = ', '.join([f"{k}({v} times)" for k, v in top_entity_attrs.items()]) if top_entity_attrs else "None"

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

    def analyze_with_rca_engine(self, anomalies):
        try:
            return self.generate_dependency_rca_report(anomalies)
        except Exception as e:
            print(f"⚠️  Dependency analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            enriched = self.prepare_anomalies_for_rca(anomalies)
            if not enriched:
                return "No valid anomalies for RCA analysis."
            return self._generate_fallback_report(anomalies, enriched)

# === 3. Knowledge Graph Construction Logic (Enhanced with GEXF/Cypher) ===
def build_knowledge_graph_for_all(anomalies, base_dir, date_str, window_str, output_folder_name):
    """
    增强版知识图谱构建：针对全量异常数据，生成JSON、GEXF（可视化）、Cypher（Neo4j导入）三种文件
    """
    # 初始化知识图谱结构
    kg_data = {
        "cluster_id": "cluster_1",
        "date": date_str,
        "time_window": window_str,
        "total_anomalies": len(anomalies),
        "time_span": {
            "start": min([a['ts'] for a in anomalies]) if anomalies else 0,
            "end": max([a['ts'] for a in anomalies]) if anomalies else 0,
            "duration_sec": (max([a['ts'] for a in anomalies]) - min([a['ts'] for a in anomalies])) if anomalies else 0
        },
        "nodes": [],
        "relationships": []
    }

    # 节点ID生成器
    node_id_counter = 1
    node_id_mapping = {}  # 节点名称 -> 节点ID
    node_label_mapping = {}  # 节点ID -> 标签（用于GEXF/Cypher）

    # 1. 创建实体节点 (Entity Nodes)
    entity_types = {
        'metric_app': 'OS',
        'metric_container': 'DOCKER',
        'trace': 'OS_Sub',
        'log': 'DB'
    }

    for anomaly in anomalies:
        entity_name = anomaly['entity']
        anomaly_type = anomaly['type']
        
        if entity_name in node_id_mapping:
            continue
        
        entity_type = entity_types.get(anomaly_type, 'unknown').upper()
        is_main_entity = not any(sub in entity_name.lower() for sub in ['sub', 'child', 'slave'])
        
        node_id = f"node_{node_id_counter}"
        node_id_counter += 1
        node_id_mapping[entity_name] = node_id
        node_label_mapping[node_id] = entity_name
        
        node = {
            "id": node_id,
            "label": entity_type,
            "properties": {
                "entity_id": entity_name,
                "entity_type": entity_type.lower(),
                "is_main_entity": is_main_entity,
                "main_entity": entity_name if is_main_entity else None,
                "anomaly_type": anomaly_type,
                "created_at": datetime.now().isoformat()
            }
        }
        kg_data["nodes"].append(node)

    # 2. 创建异常属性节点 (Attribute Nodes)
    attr_node_mapping = {}  # (entity, attribute) -> 节点ID
    for anomaly in anomalies:
        entity_name = anomaly['entity']
        attribute = anomaly['attribute']
        key = (entity_name, attribute)
        
        if key in attr_node_mapping:
            continue
        
        node_id = f"node_{node_id_counter}"
        node_id_counter += 1
        attr_node_mapping[key] = node_id
        node_label_mapping[node_id] = f"Attr_{attribute[:20]}"  # 截断过长属性名
        
        fault_type = "unknown"
        attr_lower = attribute.lower()
        
        if anomaly['type'] == 'log':
            if any(kw in attr_lower for kw in ['oom', 'out of memory']):
                fault_type = "OOM"
            elif 'gc' in attr_lower and ('full' in attr_lower or 'allocation' in attr_lower):
                fault_type = "GC_ISSUE"
            elif any(kw in attr_lower for kw in ['error', 'exception', 'fail']):
                fault_type = "SYSTEM_ERROR"
            elif 'timeout' in attr_lower:
                fault_type = "TIMEOUT"
        elif anomaly['type'] == 'metric_app' or anomaly['type'] == 'metric_container':
            if any(kw in attr_lower for kw in ['cpu', 'usage']):
                fault_type = "HIGH_CPU_USAGE"
            elif any(kw in attr_lower for kw in ['memory', 'mem']):
                fault_type = "HIGH_MEMORY_USAGE"
            elif 'disk' in attr_lower:
                fault_type = "DISK_FULL"
            elif 'network' in attr_lower:
                fault_type = "NETWORK_LATENCY"
        elif anomaly['type'] == 'trace':
            if 'latency' in attr_lower:
                fault_type = "TRACE_LATENCY"
            elif 'error' in attr_lower:
                fault_type = "TRACE_ERROR"
        
        node = {
            "id": node_id,
            "label": "AnomalyAttribute",
            "properties": {
                "attribute_name": attribute,
                "fault_type": fault_type,
                "anomaly_type": anomaly['type'],
                "timestamp": anomaly['ts'],
                "time_str": datetime.fromtimestamp(anomaly['ts']).strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        kg_data["nodes"].append(node)

    # 3. 创建故障类型节点 (FaultType Nodes)
    fault_type_mapping = {}
    all_fault_types = set()
    
    for anomaly in anomalies:
        attribute = anomaly['attribute']
        attr_lower = attribute.lower()
        
        if anomaly['type'] == 'log':
            if any(kw in attr_lower for kw in ['oom', 'out of memory']):
                all_fault_types.add("OOM")
            elif 'gc' in attr_lower and ('full' in attr_lower or 'allocation' in attr_lower):
                all_fault_types.add("GC_ISSUE")
            elif any(kw in attr_lower for kw in ['error', 'exception', 'fail']):
                all_fault_types.add("SYSTEM_ERROR")
            elif 'timeout' in attr_lower:
                all_fault_types.add("TIMEOUT")
        elif anomaly['type'] == 'metric_app' or anomaly['type'] == 'metric_container':
            if any(kw in attr_lower for kw in ['cpu', 'usage']):
                fault_type = "HIGH_CPU_USAGE"
                all_fault_types.add(fault_type)
            elif any(kw in attr_lower for kw in ['memory', 'mem']):
                fault_type = "HIGH_MEMORY_USAGE"
                all_fault_types.add(fault_type)
            elif 'disk' in attr_lower:
                fault_type = "DISK_FULL"
                all_fault_types.add(fault_type)
            elif 'network' in attr_lower:
                fault_type = "NETWORK_LATENCY"
                all_fault_types.add(fault_type)
        elif anomaly['type'] == 'trace':
            if 'latency' in attr_lower:
                fault_type = "TRACE_LATENCY"
                all_fault_types.add(fault_type)
            elif 'error' in attr_lower:
                fault_type = "TRACE_ERROR"
                all_fault_types.add(fault_type)
    
    for fault_type in all_fault_types:
        node_id = f"node_{node_id_counter}"
        node_id_counter += 1
        fault_type_mapping[fault_type] = node_id
        node_label_mapping[node_id] = f"Fault_{fault_type}"
        
        node = {
            "id": node_id,
            "label": "FaultType",
            "properties": {
                "fault_type": fault_type,
                "description": get_fault_type_description(fault_type),
                "severity": get_fault_severity(fault_type)
            }
        }
        kg_data["nodes"].append(node)

    # 4. 创建关系 (Relationships)
    rel_id_counter = 1
    
    # 4.1 HAS_ANOMALY 关系
    for anomaly in anomalies:
        entity_name = anomaly['entity']
        attribute = anomaly['attribute']
        key = (entity_name, attribute)
        
        if entity_name not in node_id_mapping or key not in attr_node_mapping:
            continue
        
        rel = {
            "id": f"rel_{rel_id_counter}",
            "type": "HAS_ANOMALY",
            "source": node_id_mapping[entity_name],
            "target": attr_node_mapping[key],
            "properties": {
                "timestamp": anomaly['ts'],
                "anomaly_type": anomaly['type'],
                "confidence": 1.0,
                "relationship_id": f"rel_{rel_id_counter}"
            }
        }
        rel_id_counter += 1
        kg_data["relationships"].append(rel)

    # 4.2 HAS_ATTRIBUTE 关系
    for (entity_name, attribute), attr_node_id in attr_node_mapping.items():
        if entity_name not in node_id_mapping:
            continue
        
        rel = {
            "id": f"rel_{rel_id_counter}",
            "type": "HAS_ATTRIBUTE",
            "source": node_id_mapping[entity_name],
            "target": attr_node_id,
            "properties": {
                "attribute_name": attribute,
                "created_at": datetime.now().isoformat()
            }
        }
        rel_id_counter += 1
        kg_data["relationships"].append(rel)

    # 4.3 MAPS_TO_FAULT 关系
    for anomaly in anomalies:
        entity_name = anomaly['entity']
        attribute = anomaly['attribute']
        key = (entity_name, attribute)
        attr_lower = attribute.lower()
        
        fault_type = "unknown"
        if anomaly['type'] == 'log':
            if any(kw in attr_lower for kw in ['oom', 'out of memory']):
                fault_type = "OOM"
            elif 'gc' in attr_lower and ('full' in attr_lower or 'allocation' in attr_lower):
                fault_type = "GC_ISSUE"
            elif any(kw in attr_lower for kw in ['error', 'exception', 'fail']):
                fault_type = "SYSTEM_ERROR"
            elif 'timeout' in attr_lower:
                fault_type = "TIMEOUT"
        elif anomaly['type'] == 'metric_app' or anomaly['type'] == 'metric_container':
            if any(kw in attr_lower for kw in ['cpu', 'usage']):
                fault_type = "HIGH_CPU_USAGE"
            elif any(kw in attr_lower for kw in ['memory', 'mem']):
                fault_type = "HIGH_MEMORY_USAGE"
            elif 'disk' in attr_lower:
                fault_type = "DISK_FULL"
            elif 'network' in attr_lower:
                fault_type = "NETWORK_LATENCY"
        elif anomaly['type'] == 'trace':
            if 'latency' in attr_lower:
                fault_type = "TRACE_LATENCY"
            elif 'error' in attr_lower:
                fault_type = "TRACE_ERROR"
        
        if fault_type == "unknown" or fault_type not in fault_type_mapping or key not in attr_node_mapping:
            continue
        
        rel = {
            "id": f"rel_{rel_id_counter}",
            "type": "MAPS_TO_FAULT",
            "source": attr_node_mapping[key],
            "target": fault_type_mapping[fault_type],
            "properties": {
                "mapping_confidence": 0.95,
                "fault_type": fault_type
            }
        }
        rel_id_counter += 1
        kg_data["relationships"].append(rel)

    # 4.4 TOPOLOGY_DEPENDS_ON 关系
    rca_analyzer = ClusterBasedBankRCAAnalyzer()
    base_edges = rca_analyzer.base_edges
    
    for source, target in base_edges:
        if source in node_id_mapping and target in node_id_mapping:
            rel = {
                "id": f"rel_{rel_id_counter}",
                "type": "TOPOLOGY_DEPENDS_ON",
                "source": node_id_mapping[source],
                "target": node_id_mapping[target],
                "properties": {
                    "dependency_type": "call_chain",
                    "weight": 1.0
                }
            }
            rel_id_counter += 1
            kg_data["relationships"].append(rel)

    # 5. 保存文件
    kg_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new6/{output_folder_name}/knowledge_graphs/{date_str}_{window_str}"
    os.makedirs(kg_dir, exist_ok=True)
    
    # 5.1 保存JSON文件
    kg_json_path = f"{kg_dir}/cluster_1/cluster_1_kg.json"
    os.makedirs(f"{kg_dir}/cluster_1", exist_ok=True)
    with open(kg_json_path, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)
    
    # 5.2 生成并保存GEXF可视化文件
    gexf_path = f"{kg_dir}/cluster_1/cluster_1_visualization.gexf"
    generate_gexf_file(kg_data, node_label_mapping, gexf_path)
    
    # 5.3 生成并保存Neo4j Cypher导入脚本
    cypher_path = f"{kg_dir}/cluster_1/cluster_1_neo4j_import.cypher"
    generate_cypher_script(kg_data, node_label_mapping, cypher_path)
    
    print(f"✅ All knowledge graph files saved to: {kg_dir}")
    print(f"   - JSON: cluster_1_kg.json")
    print(f"   - GEXF: cluster_1_visualization.gexf")
    print(f"   - Cypher: cluster_1_neo4j_import.cypher")
    return kg_json_path

def generate_gexf_file(kg_data, node_label_mapping, output_path):
    """
    生成GEXF文件（可用于Gephi等工具可视化）
    """
    # 创建NetworkX图
    G = nx.DiGraph()
    
    # 添加节点
    for node in kg_data['nodes']:
        node_id = node['id']
        label = node_label_mapping.get(node_id, node_id)
        G.add_node(
            node_id,
            label=label,
            type=node['label'],
            **node['properties']
        )
    
    # 添加边
    for rel in kg_data['relationships']:
        G.add_edge(
            rel['source'],
            rel['target'],
            label=rel['type'],
            **rel['properties']
        )
    
    # 保存为GEXF格式
    nx.write_gexf(G, output_path, encoding='utf-8')
    print(f"✅ GEXF visualization file saved: {output_path}")

def generate_cypher_script(kg_data, node_label_mapping, output_path):
    """
    生成Neo4j Cypher导入脚本
    """
    cypher_lines = []
    
    # 添加注释头
    cypher_lines.append(f"// Neo4j Import Script for Full Time Domain Anomalies")
    cypher_lines.append(f"// Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    cypher_lines.append(f"// Total nodes: {len(kg_data['nodes'])}, Total relationships: {len(kg_data['relationships'])}")
    cypher_lines.append("")
    
    # 清除已有数据（可选）
    cypher_lines.append("// Clear existing data (comment out if needed)")
    cypher_lines.append(f"MATCH (n) WHERE n.cluster_id = 'cluster_1' DELETE n;")
    cypher_lines.append("")
    
    # 创建节点
    cypher_lines.append("// Create nodes")
    for node in kg_data['nodes']:
        node_id = node['id']
        label = node['label']
        properties = node['properties'].copy()
        properties['cluster_id'] = kg_data['cluster_id']
        properties['node_id'] = node_id
        
        # 格式化属性
        prop_str = []
        for k, v in properties.items():
            if isinstance(v, str):
                # 转义单引号
                v_escaped = v.replace("'", "\\'")
                prop_str.append(f"{k}: '{v_escaped}'")
            elif isinstance(v, bool):
                prop_str.append(f"{k}: {str(v).lower()}")
            else:
                prop_str.append(f"{k}: {v}")
        
        prop_str = ", ".join(prop_str)
        cypher_lines.append(f"CREATE (n:{label} {{{prop_str}}});")
    
    cypher_lines.append("")
    
    # 创建关系
    cypher_lines.append("// Create relationships")
    for rel in kg_data['relationships']:
        rel_type = rel['type']
        source_id = rel['source']
        target_id = rel['target']
        properties = rel['properties'].copy()
        
        # 格式化属性
        prop_str = []
        for k, v in properties.items():
            if isinstance(v, str):
                v_escaped = v.replace("'", "\\'")
                prop_str.append(f"{k}: '{v_escaped}'")
            elif isinstance(v, bool):
                prop_str.append(f"{k}: {str(v).lower()}")
            else:
                prop_str.append(f"{k}: {v}")
        
        prop_str = ", ".join(prop_str) if prop_str else ""
        if prop_str:
            prop_str = f" {{{prop_str}}}"
        
        cypher_lines.append(f"""
MATCH (source) WHERE source.node_id = '{source_id}'
MATCH (target) WHERE target.node_id = '{target_id}'
CREATE (source)-[r:{rel_type}{prop_str}]->(target);
""")
    
    # 保存Cypher文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(cypher_lines))
    print(f"✅ Neo4j Cypher script saved: {output_path}")

def get_fault_type_description(fault_type):
    descriptions = {
        "OOM": "Out of Memory error - 内存不足导致进程崩溃",
        "GC_ISSUE": "Garbage Collection issue - 垃圾回收异常导致性能下降",
        "SYSTEM_ERROR": "System error/exception - 系统运行时异常",
        "TIMEOUT": "Timeout error - 服务调用超时",
        "HIGH_CPU_USAGE": "High CPU usage - CPU利用率过高",
        "HIGH_MEMORY_USAGE": "High memory usage - 内存使用率过高",
        "DISK_FULL": "Disk full - 磁盘空间不足",
        "NETWORK_LATENCY": "Network latency - 网络延迟过高",
        "TRACE_LATENCY": "Trace latency - 调用链延迟过高",
        "TRACE_ERROR": "Trace error - 调用链执行错误"
    }
    return descriptions.get(fault_type, "Unknown fault type")

def get_fault_severity(fault_type):
    severity_map = {
        "OOM": "CRITICAL",
        "GC_ISSUE": "HIGH",
        "SYSTEM_ERROR": "CRITICAL",
        "TIMEOUT": "MEDIUM",
        "HIGH_CPU_USAGE": "HIGH",
        "HIGH_MEMORY_USAGE": "HIGH",
        "DISK_FULL": "CRITICAL",
        "NETWORK_LATENCY": "MEDIUM",
        "TRACE_LATENCY": "MEDIUM",
        "TRACE_ERROR": "HIGH"
    }
    return severity_map.get(fault_type, "LOW")

# === 4. 移除聚类逻辑，仅保留全量数据处理 ===
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

def generate_full_report(anomalies, output_file, args=None):
    """
    移除聚类逻辑，直接生成全量异常的报告和知识图谱
    """
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to process.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    rca_analyzer = ClusterBasedBankRCAAnalyzer()

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Full Time Domain Anomaly Root Cause Analysis Report\n")
        f.write(f"📅 Date: {args.date_online} | Window: {args.output_suffix}\n")
        f.write(f"⚙️  Analysis Start Timestamp Index: T{ANALYSIS_START_TIMESTAMP_INDEX}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"📊 Total Anomalies: {len(anomalies)}\n")
        f.write("📈 Analysis Type: Dependency-based Root Cause Analysis (Topology + Time + Count)\n")
        f.write("=" * 80 + "\n\n")

        # 全量异常统计
        f.write(f"# Full Time Domain Anomaly Summary\n")
        ts_vals = [a['ts'] for a in anomalies]
        start_ts, end_ts = min(ts_vals), max(ts_vals)
        duration = end_ts - start_ts
        f.write(f"**Time Span**: {ts_to_beijing_str(start_ts)} → {ts_to_beijing_str(end_ts)} (Duration: {duration} seconds)\n")
        f.write(f"**Total Unique Anomalies**: {len(anomalies)}\n")

        all_keywords = set()
        for a in anomalies:
            if a['type'] == 'log':
                all_keywords.update(extract_keywords(a['raw']))
        if all_keywords:
            f.write(f"**Key Anomaly Patterns**: {', '.join(all_keywords)}\n")
        f.write("\n")

        f.write("## Anomaly Breakdown\n")
        grouped = defaultdict(list)
        for a in anomalies:
            grouped[a['type']].append(a)

        type_order = ['metric_app', 'metric_container', 'trace', 'log']
        for typ in type_order:
            if typ not in grouped:
                continue
            f.write(f"- **{typ.replace('_', ' ').title()}**: {len(grouped[typ])} anomalies\n")
        f.write("\n")

        # 全量数据RCA分析
        f.write("## Root Cause Analysis (Dependency-Based)\n")
        rca_report = rca_analyzer.analyze_with_rca_engine(anomalies)
        f.write(rca_report)
        f.write("\n")
        f.write("-" * 80 + "\n\n")
        
        # 构建全量知识图谱
        print(f"\n🏗️  Building knowledge graph for full time domain anomalies...")
        build_knowledge_graph_for_all(
            anomalies=anomalies,
            base_dir=os.path.dirname(output_file),
            date_str=args.date_online,
            window_str=args.output_suffix,
            output_folder_name=args.output_folder_name
        )

        f.write("### Analysis Metadata\n")
        f.write(f"- Time Zone: CST (UTC+8)\n")
        f.write(f"- RCA Methodology: Weighted scoring (Time: 40%, Topology: 30%, Anomaly Count: 30%)\n")
        f.write(f"- Analysis Start Timestamp Index: T{ANALYSIS_START_TIMESTAMP_INDEX} (0=T0,1=T1,2=T2)\n")
        f.write(f"- Topology Graph: Bank microservice call chain (apache → IG → Tomcat → MG → docker → mysql/redis)\n")
        f.write(f"- Knowledge Graph: Generated for full time domain in /knowledge_graphs/{args.date_online}_{args.output_suffix}/ (JSON/GEXF/Cypher formats)\n")

    print(f"✅ Report saved to: {output_file}")
    print(f"📊 Processed {len(anomalies)} full time domain anomalies.")

# === 5. Main Program ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate full time domain knowledge graph and RCA for Bank dataset.")
    parser.add_argument("--date_online", required=True, help="Date string like 2021_03_04")
    parser.add_argument("--eps", type=int, default=60, help="DBSCAN eps in seconds (default: 60 = 1 min)")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min_samples (default: 3)")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0230_0300")
    parser.add_argument("--output_folder_name", type=str, default="1116",
                        help="Output folder name (e.g., experiment ID)")

    args = parser.parse_args()

    BASE_DIR = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new6/{args.output_folder_name}"
    print(f"📁 Loading anomalies for date={args.date_online}, window={args.output_suffix}")
    anomalies = load_anomalies_from_window(BASE_DIR, args.date_online, args.output_suffix)
    
    output_file = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new6/{args.output_folder_name}/Bank_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}.txt"

    print(f"🎯 Total anomalies loaded: {len(anomalies)}")
    # 直接处理全量数据，无聚类
    generate_full_report(
        anomalies=anomalies,
        output_file=output_file,
        args=args
    )
import json
import os
import re
import time
import numpy as np
import argparse
import requests
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from zhipuai import ZhipuAI

# virtual environment: conda faiss-env

# ====================== Global Configuration ======================
# Root cause analysis weight configuration (enhanced for Market microservice architecture)
SCORE_WEIGHTS = {
    "anomaly_count": 0.3,        # Weight for anomaly count (reduced from 0.4)
    "time_priority": 0.15,       # Weight for time priority (earlier is higher)
    "topology_impact": 0.2,      # Weight for topology impact scope
    "component_weight": 0.15,    # Weight for component hierarchy
    "severity_score": 0.1,       # NEW: Weight for anomaly severity
    "business_impact": 0.05,     # NEW: Weight for business impact
    "causal_propagation": 0.05   # NEW: Weight for causal propagation
}

# Market microservice component base weights (enhanced)
COMPONENT_BASE_WEIGHT = {
    "elasticsearch": 1.0,  # ES - Highest priority (search core)
    "mq": 0.95,            # Kafka/RabbitMQ - Second highest (message queue)
    "business": 0.9,       # SpringBoot - Business layer
    "cache": 0.85,         # Memcached - Cache layer
    "gateway": 0.8,        # Nginx - Gateway layer
    "container": 0.75,     # Kubernetes - Container layer
    "entry_point": 0.7,    # Nginx - Entry layer
    "service_test": 0.65,  # Test service
    "unknown": 0.5         # Unknown component
}

# NEW: Anomaly severity weight configuration
ANOMALY_SEVERITY_WEIGHT = {
    "critical": 1.0,   # Critical - service down
    "major": 0.8,      # Major - severe performance degradation
    "minor": 0.6,      # Minor - partial function affected
    "warning": 0.4,    # Warning - potential issue
    "info": 0.2        # Information - normal event
}

# LLM Configuration (enhanced GLM-4.7 settings)
# LLM_CONFIG = {
#     "model": "glm-4.7",
#     "api_key": "e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa",
#     "api_url": "https://open.bigmodel.cn/api/coding/paas/v4",
#     "temperature": 0.4,        # Lower for more stable results
#     "max_tokens": 8192,
#     "max_retries": 3           # NEW: retry mechanism
# }

LLM_CONFIG = {
    "model": "deepseek-r1-0528",
    "api_key": "sk-e8bbbd81c0dc42dfa73d557012d1a3dd",
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "temperature": 0.4,        # Lower temperature for more stable results
    "max_tokens": 8192,
    "max_retries": 3           # New: maximum retry attempts
}

# Market timezone configuration
BEIJING_TZ = timezone(timedelta(hours=8))

# ====================== Enhanced Programmatic RCA Analyzer ======================
class ProgrammaticRCAAnalyzer:
    def __init__(self, kg_json_path):
        """Initialize enhanced programmatic root cause analyzer for Market"""
        self.kg_json_path = kg_json_path
        self.kg_data = self._load_kg_data()
        self.cluster_id = self.kg_data["cluster_id"]
        self.total_anomalies = self.kg_data["total_anomalies"]
        
        # Core analysis results (enhanced)
        self.entity_scores = {}  # Entity root cause scores
        self.root_causes = []    # Sorted root cause results
        self.analysis_summary = {}  # Final summary
        self.propagation_paths = [] # NEW: fault propagation paths
    
    def _load_kg_data(self):
        """Load knowledge graph JSON data with enhanced error handling"""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Knowledge graph file not found: {self.kg_json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in knowledge graph file: {self.kg_json_path}")
        except Exception as e:
            raise ValueError(f"Failed to load knowledge graph: {str(e)}")
    
    def _identify_service_layer(self, entity_name):
        """Enhanced Market microservice layer identification"""
        entity_lower = entity_name.lower()
        
        if 'elasticsearch' in entity_lower or 'es' in entity_lower:
            return "elasticsearch"
        elif 'kafka' in entity_lower or 'rabbitmq' in entity_lower or 'mq' in entity_lower:
            return "mq"
        elif 'springboot' in entity_lower or 'spring' in entity_lower:
            return "business"
        elif 'memcached' in entity_lower:
            return "cache"
        elif 'nginx' in entity_lower and 'gateway' in entity_lower:
            return "gateway"
        elif 'kubernetes' in entity_lower or 'k8s' in entity_lower:
            return "container"
        elif 'nginx' in entity_lower and 'entry' in entity_lower:
            return "entry_point"
        elif 'servicetest' in entity_lower:
            return "service_test"
        else:
            return "unknown"
    
    def _extract_entity_features(self):
        """Enhanced feature extraction for Market entities"""
        # 1. Basic entity information (enhanced)
        entities = {}
        for node in self.kg_data["nodes"]:
            if node["label"] in ["OS", "K8S", "ES", "OS_Sub", "K8S_Sub", "unknown"]:
                entity_id = node["id"]
                entity_name = node["properties"].get("entity_id", entity_id)
                is_main = node["properties"].get("is_main_entity", True)
                
                # Identify Market microservice layer
                component_type = self._identify_service_layer(entity_name)
                
                entities[entity_id] = {
                    "id": entity_id,
                    "entity_name": entity_name,
                    "component_type": component_type,
                    "is_main": is_main,
                    "main_entity": node["properties"].get("main_entity", entity_name),
                    "anomaly_count": 0,
                    "first_anomaly_ts": float('inf'),
                    "last_anomaly_ts": 0,
                    "fault_types": set(),
                    "topology_neighbors": set(),
                    "component_weight": COMPONENT_BASE_WEIGHT.get(component_type, 0.5),
                    # NEW features
                    "severity_score": 0.0,
                    "total_duration": 0.0,
                    "business_impact": 0,
                    "anomaly_severities": []
                }
        
        # 2. Enhanced anomaly analysis (count, time, severity, duration)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                entity_id = rel["source"]
                if entity_id in entities:
                    # Count anomalies
                    entities[entity_id]["anomaly_count"] += 1
                    
                    # Record first/last anomaly time
                    ts = rel["properties"]["timestamp"]
                    if ts < entities[entity_id]["first_anomaly_ts"]:
                        entities[entity_id]["first_anomaly_ts"] = ts
                    if ts > entities[entity_id]["last_anomaly_ts"]:
                        entities[entity_id]["last_anomaly_ts"] = ts
                    
                    # Calculate severity score (NEW)
                    severity = rel["properties"].get("severity", "info")
                    severity_weight = ANOMALY_SEVERITY_WEIGHT.get(severity, 0.2)
                    entities[entity_id]["severity_score"] += severity_weight
                    entities[entity_id]["anomaly_severities"].append(severity)
                    
                    # Calculate anomaly duration (NEW)
                    start_ts = rel["properties"].get("start_ts", ts)
                    end_ts = rel["properties"].get("end_ts", ts)
                    entities[entity_id]["total_duration"] += max(0, end_ts - start_ts)
        
        # 3. Associate fault types (enhanced)
        attr_fault_map = {}
        # First build mapping from attribute ID to fault type
        for node in self.kg_data["nodes"]:
            if node["label"] == "AnomalyAttribute" and "fault_type" in node["properties"]:
                attr_fault_map[node["id"]] = node["properties"]["fault_type"]
        
        # Then associate to entities
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ATTRIBUTE":
                entity_id = rel["source"]
                attr_id = rel["target"]
                if entity_id in entities and attr_id in attr_fault_map:
                    entities[entity_id]["fault_types"].add(attr_fault_map[attr_id])
        
        # 4. Count topology neighbors (impact scope)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "TOPOLOGY_DEPENDS_ON":
                src = rel["source"]
                dst = rel["target"]
                if src in entities:
                    entities[src]["topology_neighbors"].add(dst)
                if dst in entities:
                    entities[dst]["topology_neighbors"].add(src)
        
        # 5. Business impact analysis (NEW)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "IMPACTS_BUSINESS":
                entity_id = rel["source"]
                if entity_id in entities:
                    entities[entity_id]["business_impact"] += 1
        
        return entities
    
    def _analyze_causal_relationships(self, entities):
        """NEW: Analyze causal relationships between entities"""
        # 1. Build anomaly timeline
        anomaly_timeline = defaultdict(list)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                entity_id = rel["source"]
                ts = rel["properties"]["timestamp"]
                anomaly_timeline[ts].append(entity_id)
        
        # 2. Analyze propagation paths by time sequence
        sorted_timestamps = sorted(anomaly_timeline.keys())
        propagation_paths = []
        prev_entities = set()
        
        for ts in sorted_timestamps:
            current_entities = set(anomaly_timeline[ts])
            # Find topological connections between current and previous entities
            for entity in current_entities:
                if entity not in prev_entities and entity in entities:
                    neighbors = entities[entity]["topology_neighbors"]
                    if neighbors & prev_entities:
                        source_entity = list(neighbors & prev_entities)[0]
                        propagation_paths.append({
                            "source": source_entity,
                            "target": entity,
                            "timestamp": ts,
                            "confidence": 0.9,
                            "time_str": datetime.fromtimestamp(ts, BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
                        })
            prev_entities = current_entities
        
        self.propagation_paths = propagation_paths
        return propagation_paths
    
    def _calculate_entity_scores(self, entities):
        """Enhanced scoring algorithm for Market entities"""
        # 1. Normalization parameters (enhanced)
        max_anomaly_count = max([e["anomaly_count"] for e in entities.values()], default=1)
        valid_ts = [e["first_anomaly_ts"] for e in entities.values() if e["first_anomaly_ts"] != float('inf')]
        min_ts = min(valid_ts, default=0)
        max_ts = max(valid_ts, default=1)
        max_neighbors = max([len(e["topology_neighbors"]) for e in entities.values()], default=1)
        max_severity = max([e["severity_score"] for e in entities.values()], default=1)
        max_duration = max([e["total_duration"] for e in entities.values()], default=1)
        max_business_impact = max([e["business_impact"] for e in entities.values()], default=1)
        
        # 2. Analyze causal relationships (NEW)
        propagation_paths = self._analyze_causal_relationships(entities)
        
        # 3. Build causal weights (NEW)
        causal_weight = defaultdict(float)
        for path in propagation_paths:
            causal_weight[path["source"]] += 0.1  # Bonus for source entities
        
        # 4. Calculate scores for each dimension (enhanced)
        for entity_id, entity in entities.items():
            # Only analyze main entities
            if not entity["is_main"]:
                continue
            
            # 4.1 Anomaly count score (0-1)
            count_score = entity["anomaly_count"] / max_anomaly_count if max_anomaly_count > 0 else 0
            
            # 4.2 Time priority score (earlier is higher, 0-1)
            if entity["first_anomaly_ts"] == float('inf'):
                time_score = 0
            elif max_ts == min_ts:
                time_score = 1.0
            else:
                time_score = (max_ts - entity["first_anomaly_ts"]) / (max_ts - min_ts)
            
            # 4.3 Topology impact score (more neighbors = higher, 0-1)
            topology_score = len(entity["topology_neighbors"]) / max_neighbors if max_neighbors > 0 else 0
            
            # 4.4 Component weight score (0-1)
            component_score = entity["component_weight"] / max(COMPONENT_BASE_WEIGHT.values())
            
            # 4.5 Severity score (NEW)
            severity_score = entity["severity_score"] / max_severity if max_severity > 0 else 0
            
            # 4.6 Business impact score (NEW)
            business_impact_score = entity["business_impact"] / max_business_impact if max_business_impact > 0 else 0
            
            # 4.7 Causal propagation score (NEW)
            causal_score = min(causal_weight.get(entity_id, 0), 0.5)  # Cap at 0.5
            
            # 5. Enhanced weighted total score
            total_score = (
                count_score * SCORE_WEIGHTS["anomaly_count"] +
                time_score * SCORE_WEIGHTS["time_priority"] +
                topology_score * SCORE_WEIGHTS["topology_impact"] +
                component_score * SCORE_WEIGHTS["component_weight"] +
                severity_score * SCORE_WEIGHTS["severity_score"] +
                business_impact_score * SCORE_WEIGHTS["business_impact"] +
                causal_score * SCORE_WEIGHTS["causal_propagation"]
            )
            
            # Convert timestamp to Beijing timezone
            if entity["first_anomaly_ts"] != float('inf'):
                first_time = datetime.fromtimestamp(entity["first_anomaly_ts"], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
            else:
                first_time = "N/A"
            
            self.entity_scores[entity_id] = {
                "entity_id": entity_id,
                "entity_name": entity["entity_name"],
                "component_type": entity["component_type"],
                "total_score": round(total_score, 4),
                # Scores for each dimension
                "count_score": round(count_score, 4),
                "time_score": round(time_score, 4),
                "topology_score": round(topology_score, 4),
                "component_score": round(component_score, 4),
                "severity_score": round(severity_score, 4),
                "business_impact_score": round(business_impact_score, 4),
                "causal_score": round(causal_score, 4),
                # Original metrics
                "anomaly_count": entity["anomaly_count"],
                "first_anomaly_time": first_time,
                "total_duration": round(entity["total_duration"], 2),
                "business_impact_count": entity["business_impact"],
                "fault_types": list(entity["fault_types"]),
                "neighbor_count": len(entity["topology_neighbors"]),
                "component_weight": entity["component_weight"],
                "severity_distribution": dict(Counter(entity["anomaly_severities"])),
                # Propagation related
                "propagation_source_count": len([p for p in propagation_paths if p["source"] == entity_id]),
                "propagation_target_count": len([p for p in propagation_paths if p["target"] == entity_id])
            }
    
    def _filter_root_causes(self):
        """Enhanced root cause filtering with Market-specific rules"""
        # 1. Sort by total score
        sorted_entities = sorted(
            self.entity_scores.values(),
            key=lambda x: x["total_score"],
            reverse=True
        )
        
        # 2. Enhanced rule filtering
        threshold = 0.1  # Minimum score threshold
        for entity in sorted_entities:
            if entity["total_score"] < threshold:
                continue
            # Exclude entities with no anomalies
            if entity["anomaly_count"] == 0 and not entity["fault_types"]:
                continue
            
            self.root_causes.append(entity)
        
        # 3. Fallback (return at least 1 root cause)
        if not self.root_causes and sorted_entities:
            self.root_causes.append(sorted_entities[0])
    
    def generate_rca_report(self):
        """Generate enhanced Market-specific programmatic RCA report"""
        # 1. Extract features and calculate scores
        entities = self._extract_entity_features()
        self._calculate_entity_scores(entities)
        self._filter_root_causes()
        
        # 2. Build enhanced report
        report = []
        report.append(f"# Programmatic Root Cause Analysis Report - Cluster {self.cluster_id} (Market Microservice)")
        report.append(f"**Analysis Time**: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S CST')}")
        report.append(f"**Total Anomalies**: {self.total_anomalies}")
        report.append(f"**Analyzed Entities**: {len([e for e in entities.values() if e['is_main']])}")
        report.append(f"**RCA Dimension Weights**: Anomaly Count({SCORE_WEIGHTS['anomaly_count']*100}%) + "
                      f"Time Priority({SCORE_WEIGHTS['time_priority']*100}%) + "
                      f"Topology Impact({SCORE_WEIGHTS['topology_impact']*100}%) + "
                      f"Component Weight({SCORE_WEIGHTS['component_weight']*100}%) + "
                      f"Severity({SCORE_WEIGHTS['severity_score']*100}%) + "
                      f"Business Impact({SCORE_WEIGHTS['business_impact']*100}%) + "
                      f"Causal Propagation({SCORE_WEIGHTS['causal_propagation']*100}%)")
        report.append(f"**Market Component Weights**: Elasticsearch(1.0) > MQ(0.95) > Business(0.9) > Cache(0.85) > Gateway(0.8) > Container(0.75) > EntryPoint(0.7)")
        report.append("")
        
        # 3. NEW: Fault propagation path analysis
        if self.propagation_paths:
            report.append("## Fault Propagation Path Analysis")
            report.append("| Source Entity | Target Entity | Propagation Time | Confidence |")
            report.append("|---------------|---------------|------------------|------------|")
            for path in self.propagation_paths[:10]:  # Show first 10
                report.append(f"| {path['source']} | {path['target']} | {path['time_str']} | {path['confidence']} |")
            report.append("")
        
        # 4. Root cause results (enhanced)
        report.append("## Fault Root Cause Ranking")
        for idx, cause in enumerate(self.root_causes[:5]):  # Show top 5 only
            report.append(f"### Root Cause #{idx+1}")
            report.append(f"- **Entity ID**: {cause['entity_id']}")
            report.append(f"- **Entity Name**: {cause['entity_name']}")
            report.append(f"- **Component Type**: {cause['component_type'].upper()}")
            report.append(f"- **Root Cause Confidence**: {cause['total_score']:.4f}")
            anomaly_pct = (cause['anomaly_count']/self.total_anomalies*100) if self.total_anomalies >0 else 0
            report.append(f"- **Anomaly Count**: {cause['anomaly_count']} ({anomaly_pct:.1f}%)")
            report.append(f"- **First Anomaly Time**: {cause['first_anomaly_time']}")
            report.append(f"- **Total Anomaly Duration**: {cause['total_duration']} seconds")
            report.append(f"- **Business Impact**: {cause['business_impact_count']} business lines affected")
            report.append(f"- **Fault Types**: {', '.join(cause['fault_types']) if cause['fault_types'] else 'unknown'}")
            report.append(f"- **Topology Impact Scope**: {cause['neighbor_count']} associated entities")
            report.append(f"- **Propagation Role**: Source of {cause['propagation_source_count']} faults, target of {cause['propagation_target_count']} faults")
            report.append(f"- **Anomaly Severity Distribution**: {cause['severity_distribution']}")
            report.append(f"- **Dimension Score Breakdown**:")
            report.append(f"  - Anomaly Count Score: {cause['count_score']:.4f}")
            report.append(f"  - Time Priority Score: {cause['time_score']:.4f}")
            report.append(f"  - Topology Impact Score: {cause['topology_score']:.4f}")
            report.append(f"  - Component Weight Score: {cause['component_score']:.4f}")
            report.append(f"  - Severity Score: {cause['severity_score']:.4f}")
            report.append(f"  - Business Impact Score: {cause['business_impact_score']:.4f}")
            report.append(f"  - Causal Propagation Score: {cause['causal_score']:.4f}")
            report.append("")
        
        # 5. Fault type analysis (enhanced)
        all_fault_types = []
        for cause in self.root_causes:
            all_fault_types.extend(cause['fault_types'])
        fault_counter = Counter(all_fault_types)
        if fault_counter:
            report.append("## Fault Type Distribution")
            for fault_type, count in fault_counter.most_common():
                if fault_type != 'unknown':
                    pct = (count/len(self.root_causes)*100) if self.root_causes else 0
                    report.append(f"- **{fault_type}**: {count} entities involved ({pct:.1f}%)")
            report.append("")
        
        # 6. Remediation recommendations (enhanced Market-specific)
        report.append("## Remediation Recommendations")
        if self.root_causes:
            primary_cause = self.root_causes[0]
            report.append(f"### Emergency Actions (Complete within 1 hour)")
            report.append(f"1. Prioritize investigation of {primary_cause['entity_name']} (Confidence: {primary_cause['total_score']:.4f})")
            report.append(f"2. Focus on {primary_cause['component_type'].upper()} layer issues with {', '.join(primary_cause['fault_types'])}")
            report.append(f"3. Verify if {primary_cause['neighbor_count']} associated entities of {primary_cause['entity_name']} have cascading faults")
            report.append(f"4. Check anomaly severity distribution and prioritize critical/major anomalies")
            
            # Market-specific recommendations (enhanced)
            if primary_cause['component_type'] == 'elasticsearch':
                report.append(f"5. Check Elasticsearch cluster health status:")
                report.append(f"   - Verify shard allocation, replica status, and cluster red/yellow/green status")
                report.append(f"   - Analyze slow query logs and optimize search performance")
                report.append(f"   - Check JVM heap usage and garbage collection status")
            elif primary_cause['component_type'] == 'mq':
                report.append(f"5. Check Kafka/RabbitMQ message queue status:")
                report.append(f"   - Monitor message backlog and consumer lag metrics")
                report.append(f"   - Verify broker health and partition distribution")
                report.append(f"   - Check producer/consumer connection stability")
            elif primary_cause['component_type'] == 'business':
                report.append(f"5. Check SpringBoot application status:")
                report.append(f"   - Analyze thread pool utilization and deadlock issues")
                report.append(f"   - Check JVM memory usage and GC performance")
                report.append(f"   - Verify third-party service dependencies and timeout settings")
            elif primary_cause['component_type'] == 'cache':
                report.append(f"5. Check Memcached cache status:")
                report.append(f"   - Monitor cache hit/miss rates and eviction policies")
                report.append(f"   - Verify connection pool settings and timeout configurations")
                report.append(f"   - Check memory usage and slab allocation")
            
            report.append(f"\n### Long-term Optimization Measures")
            report.append(f"1. Enhance monitoring for {primary_cause['component_type'].upper()} layer with severity-based alert thresholds")
            report.append(f"2. Analyze fault propagation paths and optimize Market microservice call chain dependencies")
            report.append(f"3. Establish fault emergency response and rapid recovery procedures for {primary_cause['component_type'].upper()} layer")
            report.append(f"4. Regularly conduct capacity assessment and performance stress testing for core components")
        else:
            report.append("1. No clear root cause identified, recommend comprehensive inspection of Market microservice call chain")
            report.append("2. Focus on health status of Elasticsearch/MQ core data layers")
        
        # 7. Generate enhanced analysis summary
        self.analysis_summary = {
            "cluster_id": self.cluster_id,
            "analysis_time": datetime.now(BEIJING_TZ).isoformat(),
            "total_anomalies": self.total_anomalies,
            "total_entities_analyzed": len([e for e in entities.values() if e['is_main']]),
            "primary_root_cause": self.root_causes[0] if self.root_causes else {},
            "top_5_root_causes": self.root_causes[:5],
            "fault_type_distribution": dict(fault_counter),
            "propagation_paths": self.propagation_paths[:10],  # Save first 10
            "score_weights": SCORE_WEIGHTS,
            "component_base_weights": COMPONENT_BASE_WEIGHT,
            "severity_weights": ANOMALY_SEVERITY_WEIGHT
        }
        
        return "\n".join(report), self.analysis_summary

def run_programmatic_analysis(kg_dir, summary_output_path):
    """Enhanced batch programmatic analysis with progress tracking"""
    programmatic_summary = {
        "analysis_metadata": {
            "analysis_time": datetime.now(BEIJING_TZ).isoformat(),
            "score_weights": SCORE_WEIGHTS,
            "component_base_weights": COMPONENT_BASE_WEIGHT,
            "severity_weights": ANOMALY_SEVERITY_WEIGHT,
            "analysis_type": "Market_microservice_programmatic_rca"
        },
        "clusters": {}
    }
    
    processed_count = 0
    failed_count = 0
    
    for root, dirs, files in os.walk(kg_dir):
        for file in files:
            if file.endswith("_kg.json"):
                kg_path = os.path.join(root, file)
                try:
                    analyzer = ProgrammaticRCAAnalyzer(kg_path)
                    report, cluster_summary = analyzer.generate_rca_report()
                    
                    # Extract cluster name
                    cluster_name = os.path.basename(os.path.dirname(kg_path))
                    # Save individual report
                    report_path = kg_path.replace("_kg.json", "_programmatic_rca_report.md")
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"✅ Programmatic RCA report generated: {report_path}")
                    
                    # Add to summary
                    programmatic_summary["clusters"][cluster_name] = cluster_summary
                    processed_count += 1
                    
                except Exception as e:
                    failed_count += 1
                    print(f"❌ Failed to analyze {kg_path}: {e}")
    
    # Save summary JSON
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(programmatic_summary, f, ensure_ascii=False, indent=2)
    print(f"\n📊 Programmatic RCA Summary:")
    print(f"   Successfully processed: {processed_count} files")
    print(f"   Failed to process: {failed_count} files")
    print(f"   Summary saved to: {summary_output_path}")
    
    return programmatic_summary

# ====================== Enhanced LLM-driven RCA Analyzer ======================
class LLMbasedRCAAnalyzer:
    def __init__(self, kg_json_path, llm_config=None):
        """Initialize enhanced LLM-driven RCA analyzer for Market"""
        self.kg_json_path = kg_json_path
        self.kg_data = self._load_kg_data()
        self.cluster_id = self.kg_data["cluster_id"]
        self.total_anomalies = self.kg_data["total_anomalies"]
        self.llm_config = llm_config or LLM_CONFIG
        self.llm_response_dict = None
        self.llm_raw_content = None
        self.rca_report = None
        self.analysis_summary = {}
    
    def _load_kg_data(self):
        """Load knowledge graph data with enhanced error handling"""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load knowledge graph: {e}")
    
    def _identify_service_layer(self, entity_name):
        """Enhanced Market microservice layer identification for LLM prompts"""
        entity_lower = entity_name.lower()
        if 'elasticsearch' in entity_lower or 'es' in entity_lower:
            return "Elasticsearch (ES)"
        elif 'kafka' in entity_lower or 'rabbitmq' in entity_lower or 'mq' in entity_lower:
            return "Message Queue (Kafka/RabbitMQ)"
        elif 'springboot' in entity_lower or 'spring' in entity_lower:
            return "Business Layer (SpringBoot)"
        elif 'memcached' in entity_lower:
            return "Cache Layer (Memcached)"
        elif 'nginx' in entity_lower and 'gateway' in entity_lower:
            return "Gateway Layer (Nginx)"
        elif 'kubernetes' in entity_lower or 'k8s' in entity_lower:
            return "Container Layer (K8S)"
        elif 'nginx' in entity_lower and 'entry' in entity_lower:
            return "Entry Point (Nginx)"
        else:
            return "Unknown Layer"
    
    def _convert_kg_to_prompt(self):
        """Enhanced prompt engineering for Market microservice RCA"""
        # 1. System role definition (structured and detailed)
        prompt = [
            """### System Role (Must be strictly followed)
You are a senior SRE expert with 10 years of experience in Market microservice architecture fault diagnosis, proficient in the complete call chain architecture of nginx→gateway→SpringBoot→MQ→K8S→ES/Memcached.
Your analysis must follow the 5-step RCA methodology:
Step 1: Data Validation - Verify anomaly data integrity and topological relationships;
Step 2: Anomaly Clustering - Cluster anomalies by time, entity, and fault type;
Step 3: Causal Reasoning - Identify causal relationships (e.g., ES slow query→API timeout→user complaint);
Step 4: Impact Assessment - Evaluate business impact scope and propagation paths;
Step 5: Root Cause Confirmation - Provide ranked root causes with confidence scores and quantitative evidence.

### Analysis Constraints
- Reason based on data-driven inference, avoid subjective speculation
- Prioritize high severity (critical/major) and high business impact anomalies
- Focus on anomalies in Elasticsearch and Message Queue core data layers
- Output must include quantitative evidence (anomaly count, severity, propagation path, etc.)
- Strictly follow the specified output format, no deviation"""
        ]
        
        # 2. Analysis background (structured and enhanced)
        time_span = self.kg_data["time_span"]
        start_time = datetime.fromtimestamp(time_span['start'], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
        end_time = datetime.fromtimestamp(time_span['end'], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
        
        prompt.append(f"""
### Analysis Background
- Target Cluster: {self.cluster_id}
- Time Range: {start_time} to {end_time} (Duration: {time_span['duration_sec']} seconds)
- Total Anomalies: {self.total_anomalies}
- Market Microservice Layer Priority: Elasticsearch(ES) > Message Queue(Kafka/RabbitMQ) > Business(SpringBoot) > Cache(Memcached) > Gateway(Nginx) > Container(K8S) > EntryPoint(Nginx)
- Anomaly Severity Priority: critical(1.0) > major(0.8) > minor(0.6) > warning(0.4) > info(0.2)""")
        
        # 3. Entity metrics (structured table format for better LLM understanding)
        prompt.append("\n### Entity Anomaly Metrics (Structured)")
        
        # 3.1 Collect enhanced entity statistics
        entity_stats = defaultdict(lambda: {
            "name": "",
            "layer": "",
            "count": 0, 
            "severity": 0.0, 
            "first_ts": float('inf'),
            "last_ts": 0, 
            "fault_types": set(), 
            "business_impact": 0,
            "severity_dist": defaultdict(int)
        })
        
        # Extract basic entity information
        for node in self.kg_data["nodes"]:
            if node["label"] in ["OS", "K8S", "ES", "OS_Sub", "K8S_Sub", "unknown"]:
                entity_id = node["id"]
                entity_name = node["properties"].get("entity_id", entity_id)
                entity_stats[entity_id]["name"] = entity_name
                entity_stats[entity_id]["layer"] = self._identify_service_layer(entity_name)
        
        # Extract anomaly data with severity
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                entity_id = rel["source"]
                stats = entity_stats[entity_id]
                stats["count"] += 1
                
                # Calculate severity
                severity = rel["properties"].get("severity", "info")
                severity_weight = ANOMALY_SEVERITY_WEIGHT.get(severity, 0.2)
                stats["severity"] += severity_weight
                stats["severity_dist"][severity] += 1
                
                # Timestamps
                ts = rel["properties"]["timestamp"]
                if ts < stats["first_ts"]:
                    stats["first_ts"] = ts
                if ts > stats["last_ts"]:
                    stats["last_ts"] = ts
        
        # Extract fault types
        attr_fault_map = {}
        for node in self.kg_data["nodes"]:
            if node["label"] == "AnomalyAttribute" and "fault_type" in node["properties"]:
                attr_fault_map[node["id"]] = node["properties"]["fault_type"]
        
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ATTRIBUTE":
                entity_id = rel["source"]
                attr_id = rel["target"]
                if attr_id in attr_fault_map:
                    entity_stats[entity_id]["fault_types"].add(attr_fault_map[attr_id])
        
        # Extract business impact
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "IMPACTS_BUSINESS":
                entity_id = rel["source"]
                entity_stats[entity_id]["business_impact"] += 1
        
        # 3.2 Generate markdown table for LLM
        prompt.append("| Entity ID | Entity Name | Microservice Layer | Anomaly Count | Total Severity | First Anomaly Time | Business Impact | Main Fault Types |")
        prompt.append("|-----------|-------------|--------------------|---------------|----------------|--------------------|-----------------|------------------|")
        
        for entity_id, stats in sorted(entity_stats.items(), key=lambda x: (x[1]["severity"], x[1]["count"]), reverse=True):
            if stats["count"] == 0:
                continue
                
            # Format time
            first_time = "N/A"
            if stats["first_ts"] != float('inf'):
                first_time = datetime.fromtimestamp(stats["first_ts"], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
            
            # Format fault types
            fault_types = ", ".join(stats["fault_types"]) if stats["fault_types"] else "unknown"
            
            prompt.append(
                f"| {entity_id} | {stats['name'][:20]} | {stats['layer']} | {stats['count']} | {stats['severity']:.2f} | "
                f"{first_time} | {stats['business_impact']} | {fault_types[:30]} |"
            )
        
        # 4. Topological dependency relationships (enhanced)
        prompt.append("\n### Topological Dependency Relationships")
        topology_rels = defaultdict(list)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "TOPOLOGY_DEPENDS_ON":
                src_id = rel["source"]
                dst_id = rel["target"]
                src_name = entity_stats[src_id]["name"] if src_id in entity_stats else src_id
                dst_name = entity_stats[dst_id]["name"] if dst_id in entity_stats else dst_id
                topology_rels[src_name].append(dst_name)
        
        if topology_rels:
            prompt.append("Entity dependency relationship list (Market call chain):")
            for src, dsts in list(topology_rels.items())[:20]:  # Limit to 20 entries
                prompt.append(f"- {src} depends on: {', '.join(dsts)}")
        else:
            prompt.append("No topology dependency data")
        
        # 5. Mandatory output format (structured and detailed)
        prompt.append("""
### Mandatory Output Format (No Deviation Allowed)
## 1. Root Cause Ranking (Top 3)
| Rank | Entity ID | Entity Name | Confidence (0-1) | Core Root Cause | Quantitative Evidence |
|------|-----------|-------------|------------------|-----------------|-----------------------|
| 1    | [ID]      | [Name]      | [0.0-1.0]        | [Concise root cause description] | [Quantitative evidence including anomaly count, severity, propagation role, business impact, etc.] |
| 2    | [ID]      | [Name]      | [0.0-1.0]        | [Concise root cause description] | [Quantitative evidence including anomaly count, severity, propagation role, business impact, etc.] |
| 3    | [ID]      | [Name]      | [0.0-1.0]        | [Concise root cause description] | [Quantitative evidence including anomaly count, severity, propagation role, business impact, etc.] |

## 2. Fault Propagation Path Analysis
- Main Propagation Chain: [Root Entity] → [Anomaly Type] → [Affected Entity 1] → [Affected Entity 2] → [Business Impact]
- Time Sequence: [Time 1 (Root Cause)] → [Time 2 (Propagation)] → [Time 3 (Full Impact)]
- Key Trigger Point: [Specific event/metric causing propagation]

## 3. Business Impact Analysis
- Affected Business Lines: [List of specific business lines]
- Impact Severity: [High/Medium/Low] (Based on user impact and service degradation level)
- Core Metric Changes: [Specific metrics such as response time, error rate, throughput]

## 4. Remediation Recommendations (Market Microservice Specific)
### Emergency Measures (Within 1 hour)
1. [Specific executable steps with verification methods]
2. [Specific executable steps with verification methods]
3. [Specific executable steps with verification methods]

### Long-term Optimization (1-7 days)
1. [Structured improvement measures with implementation timeline and responsible person]
2. [Monitoring enhancement measures with specific metrics]
3. [Architecture optimization recommendations adapted to Market call chain]

## 5. Core Data Layer Risk Alert
- Elasticsearch Risk: [Specific risk points for ES layer]
- Message Queue Risk: [Specific risk points for MQ layer]
- Overall Risk Level: [High/Medium/Low]""")
        
        return "\n".join(prompt)
    
    def _call_llm_api(self, prompt):
                
        # Build messages
        messages = [
            {"role": "system", "content": "You are a Bank microservice fault root cause analysis expert, proficient in fault analysis of apache→IG→Tomcat→MG→docker→mysql/redis architecture."},
            {"role": "user", "content": prompt}
        ]
        # Get configuration parameters
        temperature = self.llm_config.get('temperature', 0.4)
        max_output_tokens = self.llm_config.get('max_tokens', 8192)
        
        # API call with retry
        max_retries = self.llm_config.get("max_retries", 3)
        
        if "glm" in self.llm_config['model']:
            
            print("Calling GLM API...")
                    
            """Call GLM-4.7 API (enhanced version with retry)"""
            # Initialize GLM client
            client = ZhipuAI(
                api_key=self.llm_config['api_key'],
                base_url=self.llm_config.get('api_base', 'https://open.bigmodel.cn/api/coding/paas/v4')
            )
            
            for retry in range(max_retries):
                try:
                    # Call GLM API
                    full_response = client.chat.completions.create(
                        model=self.llm_config['model'],
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_output_tokens,
                        top_p=0.95
                    )
                    
                    # Extract response content
                    response_content = full_response.choices[0].message.content
                    self.llm_raw_content = response_content
                    
                    # Convert to serializable dictionary
                    self.llm_response_dict = {
                        "model": self.llm_config['model'],
                        "content": response_content,
                        "usage": {
                            "prompt_tokens": getattr(full_response.usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(full_response.usage, 'completion_tokens', 0),
                            "total_tokens": getattr(full_response.usage, 'total_tokens', 0)
                        },
                        "created_at": datetime.now(BEIJING_TZ).isoformat(),
                        "temperature": temperature
                    }
                    
                    # Print token usage
                    total_tokens = self.llm_response_dict['usage']['total_tokens']
                    print(f"✅ LLM API call successful - Cluster {self.cluster_id} (Tokens: {total_tokens})")
                    
                    # Token limit warning
                    if total_tokens > 120000:
                        print(f"⚠️ Warning: Token usage ({total_tokens}) approaching 128K limit")
                    
                    return response_content
                    
                except Exception as e:
                    if retry == max_retries - 1:
                        raise RuntimeError(f"GLM API call failed (after {max_retries} retries): {e}")
                    wait_time = 2 ** retry  # Exponential backoff
                    print(f"❌ LLM API call failed (retry {retry+1}/{max_retries}), waiting {wait_time} seconds: {e}")
                    time.sleep(wait_time)
                    
        elif "deepseek" in self.llm_config['model'] or "qwen" in self.llm_config['model']:
            
            print(f"Calling {self.llm_config['model']} API...")
            print(f"LLM Configuration: Model={self.llm_config['model']}, Temperature={temperature}, Max Tokens={max_output_tokens}, key={self.llm_config['api_key']}, url={self.llm_config['api_base']}")
            
            
            from openai import OpenAI
    
            client = OpenAI(
                api_key=self.llm_config['api_key'],
                base_url=self.llm_config['api_base']
            )
            
            for retry in range(max_retries):
                try:
            
                    full_response = client.chat.completions.create(
                        model = self.llm_config['model'],
                        messages = messages,
                        temperature = temperature,
                    )
                    
                    response_content = full_response.choices[0].message.content
                    
                    self.llm_raw_content = response_content
                            
                    # Convert to serializable dictionary
                    self.llm_response_dict = {
                        "model": self.llm_config['model'],
                        "content": response_content,
                        "usage": {
                            "prompt_tokens": getattr(full_response.usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(full_response.usage, 'completion_tokens', 0),
                            "total_tokens": getattr(full_response.usage, 'total_tokens', 0)
                        },
                        "created_at": datetime.now(BEIJING_TZ).isoformat(),
                        "temperature": temperature
                    }
                    
                    # Print token usage
                    total_tokens = self.llm_response_dict['usage']['total_tokens']
                    print(f"✅ LLM API call successful - Cluster {self.cluster_id} (Tokens: {total_tokens})")
                    
                    # Token limit warning
                    if total_tokens > 120000:
                        print(f"⚠️ Warning: Token usage ({total_tokens}) approaching 128K limit")
                    
                    return response_content
                
                except Exception as e:
                    if retry == max_retries - 1:
                        raise RuntimeError(f"{self.llm_config['model']} API call failed (after {max_retries} retries): {e}")
                    wait_time = 2 ** retry  # Exponential backoff
                    print(f"❌ LLM API call failed (retry {retry+1}/{max_retries}), waiting {wait_time} seconds: {e}")
                    time.sleep(wait_time)
    
    def generate_rca_report(self):
        """Generate enhanced LLM-driven Market RCA report"""
        # 1. Generate optimized prompt
        prompt = self._convert_kg_to_prompt()
        
        # 2. Call LLM with retry
        llm_output = self._call_llm_api(prompt)
        
        # 3. Build final report
        report = []
        report.append(f"# LLM-driven Root Cause Analysis Report - Cluster {self.cluster_id} (Market Microservice)")
        report.append(f"**Analysis Time**: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S CST')}")
        report.append(f"**Model Used**: {self.llm_config['model']} (Temperature: {self.llm_config['temperature']})")
        report.append(f"**Knowledge Graph Source**: {self.kg_json_path}")
        report.append(f"**Market Architecture**: nginx → Gateway → SpringBoot → MQ → K8S → ES/Memcached")
        report.append(f"**Token Usage**: {self.llm_response_dict['usage']['total_tokens'] if self.llm_response_dict else 'N/A'}")
        report.append("="*80)
        report.append("")
        report.append(llm_output)
        
        self.rca_report = "\n".join(report)
        
        # 4. Generate enhanced analysis summary
        self.analysis_summary = {
            "cluster_id": self.cluster_id,
            "analysis_time": datetime.now(BEIJING_TZ).isoformat(),
            "model_used": self.llm_config['model'],
            "temperature": self.llm_config['temperature'],
            "total_anomalies": self.total_anomalies,
            "llm_response_content": llm_output,
            "token_usage": self.llm_response_dict['usage'] if self.llm_response_dict else {},
            "prompt_char_count": len(prompt)
        }
        
        return self.rca_report, self.analysis_summary
    
    def save_report(self):
        """Enhanced report saving with exception handling"""
        if not self.rca_report:
            raise ValueError("Please generate analysis report first")
        
        # Save main report
        report_path = self.kg_json_path.replace("_kg.json", "_llm_rca_report.md")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self.rca_report)
        except Exception as e:
            raise RuntimeError(f"Failed to save LLM report: {e}")
        
        # Save LLM response
        if self.llm_response_dict:
            response_path = self.kg_json_path.replace("_kg.json", "_llm_response.json")
            try:
                with open(response_path, 'w', encoding='utf-8') as f:
                    json.dump(self.llm_response_dict, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ Failed to save LLM response JSON: {e}")
                # Fallback to save raw content
                fallback_path = self.kg_json_path.replace("_kg.json", "_llm_response_raw.txt")
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    f.write(self.llm_raw_content or "No response content")
        
        print(f"✅ LLM-driven RCA report generated: {report_path}")
        return report_path

def run_llm_analysis(kg_dir, api_key, summary_output_path):
    """Enhanced batch LLM-driven analysis with progress tracking"""
    llm_config = LLM_CONFIG.copy()
    # llm_config["api_key"] = api_key
    
    llm_summary = {
        "analysis_metadata": {
            "analysis_time": datetime.now(BEIJING_TZ).isoformat(),
            "model_used": llm_config['model'],
            "temperature": llm_config['temperature'],
            "max_tokens": llm_config['max_tokens'],
            "max_retries": llm_config['max_retries'],
            "analysis_type": "Market_microservice_llm_rca"
        },
        "clusters": {}
    }
    
    processed_count = 0
    failed_count = 0
    
    for root, dirs, files in os.walk(kg_dir):
        for file in files:
            if file.endswith("_kg.json"):
                kg_path = os.path.join(root, file)
                try:
                    analyzer = LLMbasedRCAAnalyzer(kg_path, llm_config)
                    report, cluster_summary = analyzer.generate_rca_report()
                    analyzer.save_report()
                    
                    # Extract cluster name
                    cluster_name = os.path.basename(os.path.dirname(kg_path))
                    # Add to summary
                    llm_summary["clusters"][cluster_name] = cluster_summary
                    processed_count += 1
                    
                except Exception as e:
                    failed_count += 1
                    print(f"❌ Failed to analyze {kg_path}: {e}")
    
    # Save summary JSON
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(llm_summary, f, ensure_ascii=False, indent=2)
    print(f"\n📊 LLM-driven RCA Summary:")
    print(f"   Successfully processed: {processed_count} files")
    print(f"   Failed to process: {failed_count} files")
    print(f"   Summary saved to: {summary_output_path}")
    
    return llm_summary

# ====================== NEW: Result Fusion and Evaluation Module ======================
def merge_rca_results(programmatic_summary, llm_summary):
    """Fuse programmatic and LLM analysis results for Market"""
    merged_summary = {
        "analysis_metadata": {
            "merge_time": datetime.now(BEIJING_TZ).isoformat(),
            "programmatic_weight": 0.6,  # Weight for programmatic analysis
            "llm_weight": 0.4            # Weight for LLM analysis
        },
        "clusters": {}
    }
    
    # Process each cluster
    for cluster_id in programmatic_summary["clusters"].keys():
        if cluster_id not in llm_summary["clusters"]:
            print(f"⚠️ Cluster {cluster_id} not found in LLM results, skipping fusion")
            continue
        
        prog_result = programmatic_summary["clusters"][cluster_id]
        llm_result = llm_summary["clusters"][cluster_id]
        
        # 1. Extract programmatic scores
        prog_root_causes = {}
        for rc in prog_result.get("top_5_root_causes", []):
            prog_root_causes[rc["entity_id"]] = rc["total_score"]
        
        # 2. Parse scores from LLM output
        llm_root_causes = {}
        llm_content = llm_result.get("llm_response_content", "")
        
        # Regex to extract root cause table
        table_pattern = r"\| \d+ \| ([^|]+) \| ([^|]+) \| ([0-9.]+) \|.*?\|"
        matches = re.findall(table_pattern, llm_content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                entity_id = match[0].strip()
                confidence = float(match[2].strip())
                llm_root_causes[entity_id] = confidence
            except (ValueError, IndexError):
                continue
        
        # 3. Weighted fusion
        merged_root_causes = {}
        all_entities = set(prog_root_causes.keys()).union(set(llm_root_causes.keys()))
        
        prog_weight = merged_summary["analysis_metadata"]["programmatic_weight"]
        llm_weight = merged_summary["analysis_metadata"]["llm_weight"]
        
        for entity_id in all_entities:
            prog_score = prog_root_causes.get(entity_id, 0.0)
            llm_score = llm_root_causes.get(entity_id, 0.0)
            
            # Weighted average
            merged_score = (prog_score * prog_weight) + (llm_score * llm_weight)
            merged_root_causes[entity_id] = round(merged_score, 4)
        
        # 4. Sort fusion results
        sorted_merged = sorted(
            merged_root_causes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 5. Build fusion results
        merged_summary["clusters"][cluster_id] = {
            "primary_root_cause": sorted_merged[0][0] if sorted_merged else "",
            "primary_confidence_score": sorted_merged[0][1] if sorted_merged else 0.0,
            "top_5_merged_root_causes": sorted_merged[:5],
            "programmatic_details": prog_result,
            "llm_details": llm_result,
            "fusion_weights": {
                "programmatic": prog_weight,
                "llm": llm_weight
            }
        }
    
    return merged_summary

def evaluate_rca_results(merged_summary, ground_truth_path=None):
    """Evaluate RCA result accuracy (NEW)"""
    evaluation = {
        "evaluation_time": datetime.now(BEIJING_TZ).isoformat(),
        "metrics": {
            "top_1_accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "total_clusters_evaluated": 0,
            "top_1_correct": 0,
            "top_3_correct": 0
        },
        "cluster_evaluation": {}
    }
    
    # Skip evaluation if no ground truth
    if not ground_truth_path or not os.path.exists(ground_truth_path):
        print("⚠️ Ground truth file not provided, skipping evaluation")
        return evaluation
    
    # Load ground truth
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load ground truth: {e}")
        return evaluation
    
    # Evaluate each cluster
    for cluster_id, merged_result in merged_summary["clusters"].items():
        if cluster_id not in ground_truth:
            continue
        
        evaluation["metrics"]["total_clusters_evaluated"] += 1
        true_root_cause = ground_truth[cluster_id].get("root_cause_entity", "")
        top_5_merged = [e[0] for e in merged_result.get("top_5_merged_root_causes", [])]
        
        # Top-1 accuracy
        if top_5_merged and top_5_merged[0] == true_root_cause:
            evaluation["metrics"]["top_1_correct"] += 1
        
        # Top-3 accuracy
        if true_root_cause in top_5_merged[:3]:
            evaluation["metrics"]["top_3_correct"] += 1
        
        # Cluster-level evaluation
        evaluation["cluster_evaluation"][cluster_id] = {
            "true_root_cause": true_root_cause,
            "predicted_top_1": top_5_merged[0] if top_5_merged else "",
            "predicted_top_3": top_5_merged[:3],
            "top_1_correct": top_5_merged[0] == true_root_cause if top_5_merged else False,
            "top_3_correct": true_root_cause in top_5_merged[:3]
        }
    
    # Calculate overall accuracy
    total = evaluation["metrics"]["total_clusters_evaluated"]
    if total > 0:
        evaluation["metrics"]["top_1_accuracy"] = round(
            evaluation["metrics"]["top_1_correct"] / total, 4
        )
        evaluation["metrics"]["top_3_accuracy"] = round(
            evaluation["metrics"]["top_3_correct"] / total, 4
        )
    
    print(f"\n📊 RCA Evaluation Results:")
    print(f"   Clusters evaluated: {total}")
    print(f"   Top-1 accuracy: {evaluation['metrics']['top_1_accuracy']:.2%}")
    print(f"   Top-3 accuracy: {evaluation['metrics']['top_3_accuracy']:.2%}")
    
    return evaluation

# ====================== Main Program ======================
def main():
    # Parse command line arguments (enhanced)
    parser = argparse.ArgumentParser(description="Market Dataset Root Cause Analysis Program (Enhanced Version)")
    parser.add_argument("--date_online", required=True, help="Date string, e.g., 2021_03_04")
    parser.add_argument("--output_suffix", required=True, help="Time window, e.g., 0230_0300")
    parser.add_argument("--output_folder_name", type=str, default="1116",
                        help="Output folder name (e.g., experiment ID)")
    parser.add_argument("--api_key", type=str, default="e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa",
                        help="GLM API key")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Ground truth JSON file path (optional for evaluation)")
    args = parser.parse_args()
    
    # Base paths
    base_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"
    kg_root_dir = f"{base_dir}/knowledge_graphs/{args.date_online}_{args.output_suffix}"
    
    # Validate input directory
    if not os.path.exists(kg_root_dir):
        print(f"❌ Error: Knowledge graph directory does not exist - {kg_root_dir}")
        return
    
    # Create output directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Define output paths (enhanced)
    programmatic_summary_path = f"{base_dir}/Market_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_programmatic_rca_summary.json"
    llm_summary_path = f"{base_dir}/Market_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_llm_rca_summary.json"
    merged_summary_path = f"{base_dir}/Market_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_merged_rca_summary.json"
    evaluation_path = f"{base_dir}/Market_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_rca_evaluation.json"
    
    # 1. Run enhanced programmatic analysis
    print("\n=== Starting Enhanced Programmatic RCA Analysis (Market Microservice) ===")
    programmatic_summary = run_programmatic_analysis(kg_root_dir, programmatic_summary_path)
    
    # 2. Run enhanced LLM-driven analysis
    print("\n=== Starting Enhanced LLM-driven RCA Analysis (Market Microservice) ===")
    llm_summary = run_llm_analysis(kg_root_dir, args.api_key, llm_summary_path)
    
    # 3. NEW: Fuse analysis results
    print("\n=== Fusing RCA Analysis Results ===")
    merged_summary = merge_rca_results(programmatic_summary, llm_summary)
    
    # Save fusion results
    with open(merged_summary_path, 'w', encoding='utf-8') as f:
        json.dump(merged_summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Fusion results saved to: {merged_summary_path}")
    
    # 4. NEW: Evaluate results (optional)
    if args.ground_truth:
        print("\n=== Evaluating RCA Analysis Results ===")
        evaluation = evaluate_rca_results(merged_summary, args.ground_truth)
        
        # Save evaluation results
        with open(evaluation_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        print(f"✅ Evaluation results saved to: {evaluation_path}")
    
    # Final summary
    print("\n✅ All enhanced analysis completed!")
    print(f"📊 Programmatic analysis summary: {programmatic_summary_path}")
    print(f"📊 LLM-driven analysis summary: {llm_summary_path}")
    print(f"📊 Fusion results summary: {merged_summary_path}")
    if args.ground_truth:
        print(f"📊 Evaluation report: {evaluation_path}")

if __name__ == "__main__":
    main()
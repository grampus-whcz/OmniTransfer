import json
import os
import numpy as np
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

# virtual environment: conda faiss-env

# ====================== Global Configuration ======================
# Root cause analysis weight configuration (adapted to Market microservice architecture)
SCORE_WEIGHTS = {
    "anomaly_count": 0.4,      # Weight for anomaly count
    "time_priority": 0.2,      # Weight for time priority (earlier is higher)
    "topology_impact": 0.25,    # Weight for topology impact scope
    "component_weight": 0.15    # Weight for component hierarchy (ES>Message Queue>Business>Cache>Gateway>Container>Entry)
}

# Market microservice component base weights (adapted to Market layered architecture)
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

# LLM配置（GLM-4.7）
LLM_CONFIG = {
    "model": "glm-4.7",
    "api_key": "xxx",
    "api_url": "https://xxx",
    "temperature": 0.7,
    "max_tokens": 8192
}

# Market timezone configuration
BEIJING_TZ = timezone(timedelta(hours=8))

# ====================== Programmatic RCA Analyzer ======================
class ProgrammaticRCAAnalyzer:
    def __init__(self, kg_json_path):
        """Initialize programmatic root cause analyzer"""
        self.kg_json_path = kg_json_path
        self.kg_data = self._load_kg_data()
        self.cluster_id = self.kg_data["cluster_id"]
        self.total_anomalies = self.kg_data["total_anomalies"]
        
        # Core analysis results
        self.entity_scores = {}  # Entity root cause scores
        self.root_causes = []    # Sorted root cause results
        self.analysis_summary = {}  # Final summary
    
    def _load_kg_data(self):
        """Load knowledge graph JSON data"""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load knowledge graph: {e}")
    
    def _identify_service_layer(self, entity_name):
        """Identify Market microservice layer for entity (adapted to Market architecture)"""
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
        """Extract core features of Market entities"""
        # 1. Basic entity information
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
                    "fault_types": set(),
                    "topology_neighbors": set(),
                    "component_weight": COMPONENT_BASE_WEIGHT.get(component_type, 0.5)
                }
        
        # 2. Count anomaly number, first anomaly time, fault types
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                # Source of HAS_ANOMALY is entity, target is attribute
                entity_id = rel["source"]
                if entity_id in entities:
                    # Count anomalies
                    entities[entity_id]["anomaly_count"] += 1
                    # Record first anomaly time
                    ts = rel["properties"]["timestamp"]
                    if ts < entities[entity_id]["first_anomaly_ts"]:
                        entities[entity_id]["first_anomaly_ts"] = ts
        
        # 3. Associate fault types
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
        
        return entities
    
    def _calculate_entity_scores(self, entities):
        """Calculate entity root cause scores (adapted to Market architecture)"""
        # 1. Normalization parameters
        max_anomaly_count = max([e["anomaly_count"] for e in entities.values()], default=1)
        valid_ts = [e["first_anomaly_ts"] for e in entities.values() if e["first_anomaly_ts"] != float('inf')]
        min_ts = min(valid_ts, default=0)
        max_ts = max(valid_ts, default=1)
        max_neighbors = max([len(e["topology_neighbors"]) for e in entities.values()], default=1)
        
        # 2. Calculate scores for each dimension
        for entity_id, entity in entities.items():
            # Only analyze main entities
            if not entity["is_main"]:
                continue
            
            # 2.1 Anomaly count score (0-1)
            count_score = entity["anomaly_count"] / max_anomaly_count if max_anomaly_count > 0 else 0
            
            # 2.2 Time priority score (earlier is higher, 0-1)
            if entity["first_anomaly_ts"] == float('inf'):
                time_score = 0
            elif max_ts == min_ts:
                time_score = 1.0
            else:
                time_score = (max_ts - entity["first_anomaly_ts"]) / (max_ts - min_ts)
            
            # 2.3 Topology impact score (more neighbors = higher, 0-1)
            topology_score = len(entity["topology_neighbors"]) / max_neighbors if max_neighbors > 0 else 0
            
            # 2.4 Component weight score (0-1)
            component_score = entity["component_weight"] / max(COMPONENT_BASE_WEIGHT.values())
            
            # 3. Weighted total score
            total_score = (
                count_score * SCORE_WEIGHTS["anomaly_count"] +
                time_score * SCORE_WEIGHTS["time_priority"] +
                topology_score * SCORE_WEIGHTS["topology_impact"] +
                component_score * SCORE_WEIGHTS["component_weight"]
            )
            
            # Convert timestamp to Beijing timezone
            if entity["first_anomaly_ts"] != float('inf'):
                first_time = datetime.fromtimestamp(entity["first_anomaly_ts"], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
            else:
                first_time = "N/A"
            
            self.entity_scores[entity_id] = {
                "entity_id": entity_id,
                "entity_name": entity["entity_name"],
                "component_type": component_type,
                "total_score": round(total_score, 4),
                "count_score": round(count_score, 4),
                "time_score": round(time_score, 4),
                "topology_score": round(topology_score, 4),
                "component_score": round(component_score, 4),
                "anomaly_count": entity["anomaly_count"],
                "first_anomaly_time": first_time,
                "fault_types": list(entity["fault_types"]),
                "neighbor_count": len(entity["topology_neighbors"]),
                "component_weight": entity["component_weight"]
            }
    
    def _filter_root_causes(self):
        """Filter root causes based on rules (adapted to Market microservices)"""
        # 1. Sort by total score
        sorted_entities = sorted(
            self.entity_scores.values(),
            key=lambda x: x["total_score"],
            reverse=True
        )
        
        # 2. Rule filtering
        threshold = 0.1  # Minimum score threshold
        for entity in sorted_entities:
            if entity["total_score"] < threshold:
                continue
            # Exclude entities with no anomalies
            if entity["anomaly_count"] == 0:
                continue
            
            self.root_causes.append(entity)
        
        # 3. Fallback (return at least 1 root cause)
        if not self.root_causes and sorted_entities:
            self.root_causes.append(sorted_entities[0])
    
    def generate_rca_report(self):
        """Generate Market-specific programmatic root cause analysis report"""
        # 1. Extract features and calculate scores
        entities = self._extract_entity_features()
        self._calculate_entity_scores(entities)
        self._filter_root_causes()
        
        # 2. Build report
        report = []
        report.append(f"# Programmatic Root Cause Analysis Report - Cluster {self.cluster_id} (Market Microservice)")
        report.append(f"**Analysis Time**: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S CST')}")
        report.append(f"**Total Anomalies**: {self.total_anomalies}")
        report.append(f"**RCA Dimension Weights**: Anomaly Count({SCORE_WEIGHTS['anomaly_count']*100}%) + "
                      f"Time Priority({SCORE_WEIGHTS['time_priority']*100}%) + "
                      f"Topology Impact({SCORE_WEIGHTS['topology_impact']*100}%) + "
                      f"Component Weight({SCORE_WEIGHTS['component_weight']*100}%)")
        report.append(f"**Market Component Weights**: Elasticsearch(1.0) > MQ(0.95) > Business(0.9) > Cache(0.85) > Gateway(0.8) > Container(0.75) > EntryPoint(0.7)")
        report.append("")
        
        # 3. Root cause results
        report.append("## Fault Root Cause Ranking")
        for idx, cause in enumerate(self.root_causes[:5]):  # Show only top 5
            report.append(f"### Root Cause #{idx+1}")
            report.append(f"- **Entity ID**: {cause['entity_id']}")
            report.append(f"- **Entity Name**: {cause['entity_name']}")
            report.append(f"- **Component Type**: {cause['component_type'].upper()}")
            report.append(f"- **Root Cause Confidence**: {cause['total_score']:.4f}")
            anomaly_pct = (cause['anomaly_count']/self.total_anomalies*100) if self.total_anomalies >0 else 0
            report.append(f"- **Anomaly Count**: {cause['anomaly_count']} ({anomaly_pct:.1f}%)")
            report.append(f"- **First Anomaly Time**: {cause['first_anomaly_time']}")
            report.append(f"- **Fault Types**: {', '.join(cause['fault_types']) if cause['fault_types'] else 'unknown'}")
            report.append(f"- **Topology Impact Scope**: {cause['neighbor_count']} associated entities")
            report.append(f"- **Dimension Score Breakdown**:")
            report.append(f"  - Anomaly Count Score: {cause['count_score']:.4f}")
            report.append(f"  - Time Priority Score: {cause['time_score']:.4f}")
            report.append(f"  - Topology Impact Score: {cause['topology_score']:.4f}")
            report.append(f"  - Component Weight Score: {cause['component_score']:.4f}")
            report.append("")
        
        # 4. Fault type analysis
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
        
        # 5. Remediation recommendations (adapted to Market architecture)
        report.append("## Remediation Recommendations")
        if self.root_causes:
            primary_cause = self.root_causes[0]
            report.append(f"1. Prioritize troubleshooting {primary_cause['entity_name']} (Confidence: {primary_cause['total_score']:.4f})")
            report.append(f"2. Focus on {primary_cause['component_type'].upper()} layer {', '.join(primary_cause['fault_types'])} issues")
            report.append(f"3. Verify whether {primary_cause['neighbor_count']} associated entities of {primary_cause['entity_name']} have cascading faults")
            # Market-specific recommendations
            if primary_cause['component_type'] == 'elasticsearch':
                report.append(f"4. Check Elasticsearch cluster health status, shard allocation and query performance")
            elif primary_cause['component_type'] == 'mq':
                report.append(f"4. Check Kafka/RabbitMQ message backlog, consumption rate and broker health status")
            elif primary_cause['component_type'] == 'business':
                report.append(f"4. Check SpringBoot application interface response time, thread pool and business logic exceptions")
            elif primary_cause['component_type'] == 'cache':
                report.append(f"4. Check Memcached cache hit rate, connection count and memory usage")
            elif primary_cause['component_type'] == 'gateway':
                report.append(f"4. Check Nginx gateway reverse proxy configuration, rate limiting policies and request forwarding performance")
            report.append(f"5. Continuously monitor anomaly frequency and recovery status of {primary_cause['entity_name']}")
        else:
            report.append("1. No clear root cause identified, recommend comprehensive inspection of Market microservice call chain")
        
        # 6. Generate analysis summary
        self.analysis_summary = {
            "cluster_id": self.cluster_id,
            "analysis_time": datetime.now(BEIJING_TZ).isoformat(),
            "total_anomalies": self.total_anomalies,
            "primary_root_cause": self.root_causes[0] if self.root_causes else {},
            "top_5_root_causes": self.root_causes[:5],
            "fault_type_distribution": dict(fault_counter),
            "score_weights": SCORE_WEIGHTS,
            "component_base_weights": COMPONENT_BASE_WEIGHT
        }
        
        return "\n".join(report), self.analysis_summary

def run_programmatic_analysis(kg_dir, summary_output_path):
    """Batch run programmatic root cause analysis and generate summary"""
    programmatic_summary = {
        "analysis_metadata": {
            "analysis_time": datetime.now(BEIJING_TZ).isoformat(),
            "score_weights": SCORE_WEIGHTS,
            "component_base_weights": COMPONENT_BASE_WEIGHT,
            "analysis_type": "Market_microservice_programmatic_rca"
        },
        "clusters": {}
    }
    
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
                    
                except Exception as e:
                    print(f"❌ Failed to analyze {kg_path}: {e}")
    
    # Save summary JSON
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(programmatic_summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Programmatic RCA summary saved to: {summary_output_path}")
    
    return programmatic_summary

# ====================== LLM-driven RCA Analyzer ======================
class LLMbasedRCAAnalyzer:
    def __init__(self, kg_json_path, llm_config=None):
        """Initialize LLM-driven root cause analyzer (Market-specific)"""
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
        """Load knowledge graph data"""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load knowledge graph: {e}")
    
    def _identify_service_layer(self, entity_name):
        """Identify Market microservice layer"""
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
        """Convert Market knowledge graph to LLM prompt"""
        # 1. Basic information
        time_span = self.kg_data["time_span"]
        start_time = datetime.fromtimestamp(time_span['start'], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
        end_time = datetime.fromtimestamp(time_span['end'], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
        
        prompt = []
        prompt.append("### Analysis Background")
        prompt.append("You are a senior fault root cause analysis expert for Market microservice architecture, familiar with the call chain architecture of nginx→gateway→SpringBoot→MQ→K8S→ES/Memcached.")
        prompt.append("Please analyze fault root causes based on the following Market microservice anomaly knowledge graph data.")
        prompt.append(f"Analysis Target: Anomaly Cluster {self.cluster_id}")
        prompt.append(f"Time Range: {start_time} to {end_time} (Duration: {time_span['duration_sec']} seconds)")
        prompt.append(f"Total Anomalies: {self.total_anomalies}")
        prompt.append("Market Microservice Layer Priority: Elasticsearch(ES) > Message Queue(Kafka/RabbitMQ) > Business(SpringBoot) > Cache(Memcached) > Gateway(Nginx) > Container(K8S) > EntryPoint(Nginx)")
        prompt.append("")
        
        # 2. Entity information
        prompt.append("### Entity Information")
        entities = {}
        for node in self.kg_data["nodes"]:
            if node["label"] in ["OS", "K8S", "ES", "OS_Sub", "K8S_Sub", "unknown"]:
                entity_id = node["id"]
                entity_name = node["properties"].get("entity_id", entity_id)
                layer = self._identify_service_layer(entity_name)
                entities[entity_id] = {
                    "id": entity_id,
                    "name": entity_name,
                    "layer": layer,
                    "is_main": node["properties"].get("is_main_entity", True)
                }
        
        prompt.append(f"Total Involved Entities: {len(entities)}")
        prompt.append("Entity List (including Market microservice layers):")
        for entity_id, info in entities.items():
            prompt.append(f"- {entity_id} (Name: {info['name']}, Layer: {info['layer']}, Main Entity: {info['is_main']})")
        prompt.append("")
        
        # 3. Anomaly distribution
        prompt.append("### Anomaly Distribution")
        entity_anomaly_count = defaultdict(int)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                entity_anomaly_count[rel["source"]] += 1
        
        prompt.append("Anomaly count per entity:")
        sorted_counts = sorted(entity_anomaly_count.items(), key=lambda x: x[1], reverse=True)
        for entity_id, count in sorted_counts:
            if count > 0 and entity_id in entities:
                prompt.append(f"- {entities[entity_id]['name']} ({entity_id}): {count} anomalies")
        prompt.append("")
        
        # 4. Fault types
        prompt.append("### Fault Types")
        fault_types = defaultdict(int)
        attr_fault_map = {}
        
        # Build attribute-fault type mapping
        for node in self.kg_data["nodes"]:
            if node["label"] == "AnomalyAttribute" and "fault_type" in node["properties"]:
                attr_fault_map[node["id"]] = node["properties"]["fault_type"]
        
        # Count fault types
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "MAPS_TO_FAULT" and rel["source"] in attr_fault_map:
                fault_type = attr_fault_map[rel["source"]]
                fault_types[fault_type] += 1
        
        prompt.append("Fault type distribution:")
        for fault_type, count in fault_types.items():
            prompt.append(f"- {fault_type}: Involves {count} anomaly attributes")
        prompt.append("")
        
        # 5. Topology relationships
        prompt.append("### Topology Dependencies")
        topology_rels = defaultdict(list)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "TOPOLOGY_DEPENDS_ON":
                src_id = rel["source"]
                dst_id = rel["target"]
                src_name = entities.get(src_id, {}).get("name", src_id)
                dst_name = entities.get(dst_id, {}).get("name", dst_id)
                topology_rels[src_name].append(dst_name)
        
        prompt.append("Entity topology dependencies (Market call chain):")
        for src, dsts in topology_rels.items():
            prompt.append(f"- {src} depends on: {', '.join(dsts)}")
        prompt.append("")
        
        # 6. Analysis requirements (Market-specific)
        prompt.append("### Analysis Requirements")
        prompt.append("1. Identify the most likely fault root cause entities (ranked by confidence, at least 3), explain the reasoning process combined with Market microservice layer priority;")
        prompt.append("2. Analyze fault propagation path (time dimension + Market call chain topology dimension);")
        prompt.append("3. Identify main fault types and impact scope;")
        prompt.append("4. Provide specific, actionable remediation recommendations and fault troubleshooting steps for Market microservice architecture;")
        prompt.append("5. Focus on anomalies in search engine (ES) and message queue (Kafka/RabbitMQ) layers, which are core business layers of Market;")
        prompt.append("6. Output format: Use Markdown format, clear sections, rigorous logic, and sufficient supporting evidence.")
        
        return "\n".join(prompt)
    
    def _call_llm_api(self, prompt):
        """Call GLM-4.7 API (adapted to Market analysis)"""
        from zhipuai import ZhipuAI
        
        # Initialize GLM client
        client = ZhipuAI(
            api_key=self.llm_config['api_key'],
            base_url=self.llm_config.get('api_base', 'https://open.bigmodel.cn/api/coding/paas/v4')
        )
        
        # Build messages
        messages = [
            {"role": "system", "content": "You are a Market microservice fault root cause analysis expert, proficient in fault analysis of nginx→gateway→SpringBoot→MQ→K8S→ES/Memcached architecture."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Get configuration parameters
            temperature = self.llm_config.get('temperature', 0.7)
            max_output_tokens = self.llm_config.get('max_tokens', 8192)
            
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
                "created_at": datetime.now(BEIJING_TZ).isoformat()
            }
            
            # Print token usage
            prompt_tokens = self.llm_response_dict['usage']['prompt_tokens']
            completion_tokens = self.llm_response_dict['usage']['completion_tokens']
            total_tokens = self.llm_response_dict['usage']['total_tokens']
            print(f"=={self.llm_config['model']}== Input tokens: {prompt_tokens}, Output tokens: {completion_tokens}, Total: {total_tokens}")
            
            # Token limit warning
            if total_tokens > 120000:
                print(f"Warning: Token usage ({total_tokens}) is close to 128K limit")
            
            return response_content
            
        except Exception as e:
            raise RuntimeError(f"GLM API call failed: {e}")
    
    def generate_rca_report(self):
        """Generate LLM-driven Market root cause analysis report"""
        # 1. Generate prompt
        prompt = self._convert_kg_to_prompt()
        
        # 2. Call LLM
        llm_output = self._call_llm_api(prompt)
        
        # 3. Build final report
        report = []
        report.append(f"# LLM-driven Root Cause Analysis Report - Cluster {self.cluster_id} (Market Microservice)")
        report.append(f"**Analysis Time**: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S CST')}")
        report.append(f"**Model Used**: {self.llm_config['model']}")
        report.append(f"**Knowledge Graph Source**: {self.kg_json_path}")
        report.append(f"**Market Architecture**: nginx → Gateway → SpringBoot → MQ → K8S → ES/Memcached")
        report.append("="*80)
        report.append("")
        report.append(llm_output)
        
        self.rca_report = "\n".join(report)
        
        # 4. Generate analysis summary
        self.analysis_summary = {
            "cluster_id": self.cluster_id,
            "analysis_time": datetime.now(BEIJING_TZ).isoformat(),
            "model_used": self.llm_config['model'],
            "total_anomalies": self.total_anomalies,
            "llm_response_content": llm_output,
            "token_usage": self.llm_response_dict['usage'] if self.llm_response_dict else {}
        }
        
        return self.rca_report, self.analysis_summary
    
    def save_report(self):
        """Save analysis report"""
        if not self.rca_report:
            raise ValueError("Please generate analysis report first")
        
        # Save main report
        report_path = self.kg_json_path.replace("_kg.json", "_llm_rca_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.rca_report)
        
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
    """Batch run LLM-driven root cause analysis"""
    llm_config = LLM_CONFIG.copy()
    llm_config["api_key"] = api_key
    
    llm_summary = {
        "analysis_metadata": {
            "analysis_time": datetime.now(BEIJING_TZ).isoformat(),
            "model_used": llm_config['model'],
            "temperature": llm_config['temperature'],
            "max_tokens": llm_config['max_tokens'],
            "analysis_type": "Market_microservice_llm_rca"
        },
        "clusters": {}
    }
    
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
                    
                except Exception as e:
                    print(f"❌ Failed to analyze {kg_path}: {e}")
    
    # Save summary JSON
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(llm_summary, f, ensure_ascii=False, indent=2)
    print(f"✅ LLM-driven RCA summary saved to: {summary_output_path}")
    
    return llm_summary

# ====================== Main Program ======================
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Market dataset root cause analysis program (Programmatic + LLM-driven)")
    parser.add_argument("--date_online", required=True, help="Date string, e.g. 2021_03_04")
    parser.add_argument("--output_suffix", required=True, help="Time window, e.g. 0230_0300")
    parser.add_argument("--output_folder_name", type=str, default="1116",
                        help="Output folder name (e.g. experiment ID)")
    args = parser.parse_args()
    
    # Basic paths
    base_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"
    kg_root_dir = f"{base_dir}/knowledge_graphs/{args.date_online}_{args.output_suffix}"
    
    # Summary output paths
    programmatic_summary_path = f"{base_dir}/Market_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_programmatic_rca_summary.json"
    llm_summary_path = f"{base_dir}/Market_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_llm_rca_summary.json"
    
    # Validate input directory
    if not os.path.exists(kg_root_dir):
        print(f"❌ Error: Knowledge graph directory does not exist - {kg_root_dir}")
        return
    
    # 1. Run programmatic analysis
    print("\n=== Starting Programmatic RCA Analysis (Market Microservice) ===")
    run_programmatic_analysis(kg_root_dir, programmatic_summary_path)
    
    # 2. Run LLM-driven analysis
    print("\n=== Starting LLM-driven RCA Analysis (Market Microservice) ===")
    api_key = "e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa"  # Can be changed to CLI parameter
    run_llm_analysis(kg_root_dir, api_key, llm_summary_path)
    
    print("\n✅ All analyses completed!")
    print(f"📊 Programmatic analysis summary: {programmatic_summary_path}")
    print(f"📊 LLM-driven analysis summary: {llm_summary_path}")

if __name__ == "__main__":
    main()
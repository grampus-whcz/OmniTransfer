import json
import os
import numpy as np
import requests
from collections import defaultdict, Counter
from datetime import datetime

# virtual environment: conda faiss-env

# ====================== Programmatic RCA Analyzer ======================
# Root cause analysis weight configuration (tunable based on business needs)
SCORE_WEIGHTS = {
    "anomaly_count": 0.4,    # Weight for anomaly count
    "time_priority": 0.2,    # Weight for time priority (earlier = higher)
    "topology_impact": 0.25, # Weight for topology impact scope
    "component_weight": 0.15 # Weight for component tier (DB>OS>Docker)
}

# Component base weights
COMPONENT_BASE_WEIGHT = {
    "db": 1.0,
    "os": 0.9,
    "docker": 0.85,
    "unknown": 0.5
}

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
    
    def _load_kg_data(self):
        """Load knowledge graph JSON data"""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load knowledge graph: {e}")
    
    def _extract_entity_features(self):
        """Extract core entity features"""
        # 1. Basic entity information
        entities = {}
        for node in self.kg_data["nodes"]:
            if node["label"] in ["DB", "OS", "DOCKER", "OS_Sub", "DOCKER_Sub"]:
                entity_id = node["id"]
                is_main = node["properties"].get("is_main_entity", True)
                component_type = node["properties"]["entity_type"].lower()
                entities[entity_id] = {
                    "id": entity_id,
                    "component_type": component_type,
                    "is_main": is_main,
                    "main_entity": node["properties"].get("main_entity", entity_id),
                    "anomaly_count": 0,
                    "first_anomaly_ts": float('inf'),
                    "fault_types": set(),
                    "topology_neighbors": set(),
                    "component_weight": COMPONENT_BASE_WEIGHT.get(component_type, 0.5)
                }
        
        # 2. Count anomaly occurrences, first anomaly time, fault types
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                entity_id = rel["target"]
                if entity_id in entities:
                    # Count anomalies
                    entities[entity_id]["anomaly_count"] += 1
                    # Record first anomaly time
                    ts = rel["properties"]["timestamp"]
                    if ts < entities[entity_id]["first_anomaly_ts"]:
                        entities[entity_id]["first_anomaly_ts"] = ts
        
        # 3. Associate fault types
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ATTRIBUTE":
                entity_id = rel["source"]
                attr_id = rel["target"]
                # Find fault type for attribute
                for node in self.kg_data["nodes"]:
                    if node["id"] == attr_id and "fault_type" in node["properties"]:
                        fault_type = node["properties"]["fault_type"]
                        if entity_id in entities:
                            entities[entity_id]["fault_types"].add(fault_type)
        
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
        """Calculate entity root cause scores"""
        # 1. Normalization parameters
        max_anomaly_count = max([e["anomaly_count"] for e in entities.values()], default=1)
        min_ts = min([e["first_anomaly_ts"] for e in entities.values() if e["first_anomaly_ts"] != float('inf')], default=0)
        max_ts = max([e["first_anomaly_ts"] for e in entities.values() if e["first_anomaly_ts"] != float('inf')], default=1)
        max_neighbors = max([len(e["topology_neighbors"]) for e in entities.values()], default=1)
        
        # 2. Calculate scores for each dimension
        for entity_id, entity in entities.items():
            # Skip sub-entities (only analyze main entities)
            if not entity["is_main"]:
                continue
            
            # 2.1 Anomaly count score (0-1)
            count_score = entity["anomaly_count"] / max_anomaly_count if max_anomaly_count > 0 else 0
            
            # 2.2 Time priority score (earlier = higher, 0-1)
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
            
            self.entity_scores[entity_id] = {
                "entity_id": entity_id,
                "component_type": entity["component_type"],
                "total_score": round(total_score, 4),
                "count_score": round(count_score, 4),
                "time_score": round(time_score, 4),
                "topology_score": round(topology_score, 4),
                "component_score": round(component_score, 4),
                "anomaly_count": entity["anomaly_count"],
                "first_anomaly_time": datetime.fromtimestamp(entity["first_anomaly_ts"]).strftime("%Y-%m-%d %H:%M:%S") 
                                     if entity["first_anomaly_ts"] != float('inf') else "N/A",
                "fault_types": list(entity["fault_types"]),
                "neighbor_count": len(entity["topology_neighbors"])
            }
    
    def _filter_root_causes(self):
        """Filter root causes (rule-based)"""
        # 1. Sort by total score
        sorted_entities = sorted(
            self.entity_scores.values(),
            key=lambda x: x["total_score"],
            reverse=True
        )
        
        # 2. Rule-based filtering
        threshold = 0.1  # Minimum score threshold
        for entity in sorted_entities:
            if entity["total_score"] < threshold:
                continue
            # Exclude entities with zero anomalies
            if entity["anomaly_count"] == 0:
                continue
            
            self.root_causes.append(entity)
        
        # 3. Fallback (return at least 1 root cause)
        if not self.root_causes and sorted_entities:
            self.root_causes.append(sorted_entities[0])
    
    def generate_rca_report(self):
        """Generate programmatic root cause analysis report"""
        # 1. Extract features and calculate scores
        entities = self._extract_entity_features()
        self._calculate_entity_scores(entities)
        self._filter_root_causes()
        
        # 2. Build report
        report = []
        report.append(f"# Programmatic Root Cause Analysis Report - Cluster {self.cluster_id}")
        report.append(f"**Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Anomalies**: {self.total_anomalies}")
        report.append(f"**RCA Dimension Weights**: Anomaly Count({SCORE_WEIGHTS['anomaly_count']*100}%) + "
                      f"Time Priority({SCORE_WEIGHTS['time_priority']*100}%) + "
                      f"Topology Impact({SCORE_WEIGHTS['topology_impact']*100}%) + "
                      f"Component Weight({SCORE_WEIGHTS['component_weight']*100}%)")
        report.append("")
        
        # 3. Root cause results
        report.append("## Fault Root Cause Ranking")
        for idx, cause in enumerate(self.root_causes[:5]):  # Show top 5 only
            report.append(f"### Root Cause #{idx+1}")
            report.append(f"- **Entity ID**: {cause['entity_id']}")
            report.append(f"- **Component Type**: {cause['component_type'].upper()}")
            report.append(f"- **Root Cause Confidence**: {cause['total_score']:.4f}")
            report.append(f"- **Anomaly Count**: {cause['anomaly_count']} ({cause['anomaly_count']/self.total_anomalies*100:.1f}%)")
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
                    report.append(f"- **{fault_type}**: {count} entities involved ({count/len(self.root_causes)*100:.1f}%)")
            report.append("")
        
        # 5. Remediation recommendations
        report.append("## Remediation Recommendations")
        primary_cause = self.root_causes[0]
        report.append(f"1. Prioritize investigation of {primary_cause['entity_id']} (confidence: {primary_cause['total_score']:.4f})")
        report.append(f"2. Focus on {', '.join(primary_cause['fault_types'])} issues in {primary_cause['component_type'].upper()} layer")
        report.append(f"3. Verify {primary_cause['neighbor_count']} associated entities of {primary_cause['entity_id']} for cascading failures")
        report.append(f"4. Monitor anomaly frequency and recovery status of {primary_cause['entity_id']}")
        
        return "\n".join(report)

def run_programmatic_analysis(kg_dir):
    """Run programmatic root cause analysis in batch"""
    for root, dirs, files in os.walk(kg_dir):
        for file in files:
            if file.endswith("_kg.json"):
                kg_path = os.path.join(root, file)
                try:
                    analyzer = ProgrammaticRCAAnalyzer(kg_path)
                    report = analyzer.generate_rca_report()
                    # Save report
                    report_path = kg_path.replace("_kg.json", "_programmatic_rca_report.md")
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"✅ Programmatic RCA report generated: {report_path}")
                except Exception as e:
                    print(f"❌ Analysis failed for {kg_path}: {e}")

# ====================== LLM-driven RCA Analyzer ======================
# LLM configuration (GLM-4.7)
LLM_CONFIG = {
    "model": "glm-4.7",  # Model name
    "api_key": "e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa",  # Replace with your API Key
    "api_url": "https://open.bigmodel.cn/api/coding/paas/v4",
    "temperature": 0.7,    # Low randomness for stable results
    "max_tokens": 8192     # Maximum output length
}

class LLMbasedRCAAnalyzer:
    def __init__(self, kg_json_path, llm_config=None):
        """Initialize LLM-driven root cause analyzer"""
        self.kg_json_path = kg_json_path
        self.kg_data = self._load_kg_data()
        self.cluster_id = self.kg_data["cluster_id"]
        self.total_anomalies = self.kg_data["total_anomalies"]
        self.llm_config = llm_config or LLM_CONFIG
        self.llm_response_dict = None  # Use dict instead of Completion object
        self.llm_raw_content = None    # Store only the content
        self.rca_report = None
    
    def _load_kg_data(self):
        """Load knowledge graph data"""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load knowledge graph: {e}")
    
    def _convert_kg_to_prompt(self):
        """Convert knowledge graph to LLM prompt"""
        # 1. Basic information
        time_span = self.kg_data["time_span"]
        prompt = []
        prompt.append("### Analysis Background")
        prompt.append("You are a senior microservice fault root cause analysis expert. Based on the following microservice anomaly knowledge graph data, analyze the fault root causes.")
        prompt.append(f"Analysis Target: Anomaly Cluster {self.cluster_id}")
        prompt.append(f"Time Range: {time_span['start']} to {time_span['end']} (Duration: {time_span['duration_sec']} seconds)")
        prompt.append(f"Total Anomalies: {self.total_anomalies}")
        prompt.append("")
        
        # 2. Entity information
        prompt.append("### Entity Information")
        entities = {}
        for node in self.kg_data["nodes"]:
            if node["label"] in ["DB", "OS", "DOCKER", "OS_Sub", "DOCKER_Sub"]:
                entity_id = node["id"]
                entities[entity_id] = {
                    "id": entity_id,
                    "type": node["properties"]["entity_type"],
                    "is_main": node["properties"].get("is_main_entity", True)
                }
        prompt.append(f"Total Entities Involved: {len(entities)}")
        prompt.append("Entity List:")
        for entity_id, info in entities.items():
            prompt.append(f"- {entity_id} (Type: {info['type']}, Main Entity: {info['is_main']})")
        prompt.append("")
        
        # 3. Anomaly distribution
        prompt.append("### Anomaly Distribution")
        # Count anomalies per entity
        entity_anomaly_count = defaultdict(int)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                entity_anomaly_count[rel["target"]] += 1
        
        prompt.append("Anomaly Count per Entity:")
        for entity_id, count in sorted(entity_anomaly_count.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                prompt.append(f"- {entity_id}: {count} occurrences")
        prompt.append("")
        
        # 4. Fault types
        prompt.append("### Fault Types")
        fault_types = defaultdict(int)
        for node in self.kg_data["nodes"]:
            if node["label"] == "FaultType":
                fault_type = node["properties"]["fault_type"]
                # Count entities involved with this fault type
                count = 0
                for rel in self.kg_data["relationships"]:
                    if rel["type"] == "MAPS_TO_FAULT" and rel["target"] == node["id"]:
                        count += 1
                fault_types[fault_type] = count
        
        prompt.append("Fault Type Distribution:")
        for fault_type, count in fault_types.items():
            prompt.append(f"- {fault_type}: Involved in {count} anomaly attributes")
        prompt.append("")
        
        # 5. Topology relationships
        prompt.append("### Topology Dependencies")
        topology_rels = defaultdict(list)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "TOPOLOGY_DEPENDS_ON":
                topology_rels[rel["source"]].append(rel["target"])
        
        prompt.append("Entity Topology Dependencies:")
        for src, dsts in topology_rels.items():
            prompt.append(f"- {src} depends on: {', '.join(dsts)}")
        prompt.append("")
        
        # 6. Analysis requirements
        prompt.append("### Analysis Requirements")
        prompt.append("1. Identify the most likely fault root cause entities (ranked by confidence, at least 3), explain the reasoning;")
        prompt.append("2. Analyze fault propagation path (time dimension + topology dimension);")
        prompt.append("3. Identify main fault types and impact scope;")
        prompt.append("4. Provide specific, actionable remediation recommendations and troubleshooting steps;")
        prompt.append("5. Output Format: Use Markdown format with clear sections, rigorous logic, and sufficient supporting evidence.")
        
        return "\n".join(prompt)
    
    def _call_llm_api(self, prompt):
        """Call GLM-4.7 LLM API (Coding端点) with serializable response"""
        from zhipuai import ZhipuAI
        
        # Initialize GLM client
        client = ZhipuAI(
            api_key=self.llm_config['api_key'],
            base_url=self.llm_config.get('api_base', 'https://open.bigmodel.cn/api/coding/paas/v4')
        )
        
        # Build messages
        messages = [
            {"role": "system", "content": "You are an expert in microservice fault root cause analysis, skilled at analyzing root causes based on knowledge graphs."},
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
            
            # Convert Completion object to serializable dict
            self.llm_response_dict = {
                "model": self.llm_config['model'],
                "content": response_content,
                "usage": {
                    "prompt_tokens": full_response.usage.prompt_tokens if hasattr(full_response, 'usage') and full_response.usage else 0,
                    "completion_tokens": full_response.usage.completion_tokens if hasattr(full_response, 'usage') and full_response.usage else 0,
                    "total_tokens": full_response.usage.total_tokens if hasattr(full_response, 'usage') and full_response.usage else 0
                },
                "created_at": datetime.now().isoformat()
            }
            
            # Print token usage
            prompt_tokens = self.llm_response_dict['usage']['prompt_tokens']
            completion_tokens = self.llm_response_dict['usage']['completion_tokens']
            total_tokens = self.llm_response_dict['usage']['total_tokens']
            print(f"=={self.llm_config['model']}== input: {prompt_tokens}, output: {completion_tokens}, total: {total_tokens}")
            
            # Warning for token limit
            if total_tokens > 120000:
                print(f"Warning: Token usage ({total_tokens}) approaching 128K limit")
            
            return response_content
            
        except Exception as e:
            raise RuntimeError(f"GLM API call failed: {e}")
    
    def generate_rca_report(self):
        """Generate LLM-driven root cause analysis report"""
        # 1. Generate prompt
        prompt = self._convert_kg_to_prompt()
        
        # 2. Call LLM
        llm_output = self._call_llm_api(prompt)
        
        # 3. Build final report
        report = []
        report.append(f"# LLM-Driven Root Cause Analysis Report - Cluster {self.cluster_id}")
        report.append(f"**Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Model Used**: {self.llm_config['model']}")
        report.append(f"**Knowledge Graph Source**: {self.kg_json_path}")
        report.append("="*80)
        report.append("")
        report.append(llm_output)
        
        self.rca_report = "\n".join(report)
        return self.rca_report
    
    def save_report(self):
        """Save analysis report with serializable data"""
        if not self.rca_report:
            raise ValueError("Please generate analysis report first")
        
        # Save main report
        report_path = self.kg_json_path.replace("_kg.json", "_llm_rca_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.rca_report)
        
        # Save LLM response as serializable dict (fixed the core issue)
        if self.llm_response_dict:
            response_path = self.kg_json_path.replace("_kg.json", "_llm_response.json")
            try:
                with open(response_path, 'w', encoding='utf-8') as f:
                    json.dump(self.llm_response_dict, f, ensure_ascii=False, indent=2)
            except Exception as e:
                # Fallback if any serialization issue
                print(f"⚠️ Warning: Failed to save LLM response JSON: {e}")
                # Save raw content as text
                fallback_path = self.kg_json_path.replace("_kg.json", "_llm_response_raw.txt")
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    f.write(self.llm_raw_content or "No response content")
        
        print(f"✅ LLM-driven RCA report generated: {report_path}")
        return report_path

def run_llm_analysis(kg_dir, api_key):
    """Run LLM-driven root cause analysis in batch"""
    llm_config = LLM_CONFIG.copy()
    llm_config["api_key"] = api_key
    
    for root, dirs, files in os.walk(kg_dir):
        for file in files:
            if file.endswith("_kg.json"):
                kg_path = os.path.join(root, file)
                try:
                    analyzer = LLMbasedRCAAnalyzer(kg_path, llm_config)
                    analyzer.generate_rca_report()
                    analyzer.save_report()
                except Exception as e:
                    print(f"❌ Analysis failed for {kg_path}: {e}")

# ====================== Usage Examples ======================
if __name__ == "__main__":
    # Example 1: Programmatic analysis (single cluster)
    analyzer = ProgrammaticRCAAnalyzer("/root/shared-nvme/work/timeSeries/OmniTransfer_new/1216/knowledge_graphs/cluster_1/cluster_1_kg.json")
    report = analyzer.generate_rca_report()
    print(report)
    
    # Example 2: Batch programmatic analysis
    run_programmatic_analysis("/root/shared-nvme/work/timeSeries/OmniTransfer_new/1216/knowledge_graphs")
    
    # Example 3: LLM-driven analysis (single cluster)
    llm_config = LLM_CONFIG.copy()
    llm_config["api_key"] = "e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa"
    analyzer = LLMbasedRCAAnalyzer("/root/shared-nvme/work/timeSeries/OmniTransfer_new/1216/knowledge_graphs/cluster_1/cluster_1_kg.json", llm_config)
    analyzer.generate_rca_report()
    analyzer.save_report()
    
    # Example 4: Batch LLM-driven analysis
    run_llm_analysis("/root/shared-nvme/work/timeSeries/OmniTransfer_new/1216/knowledge_graphs", "e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa")
import json
import os
import re
import time
import itertools
import numpy as np
import requests
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

# virtual environment: conda faiss-env

# ====================== Configuration ======================
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

# Anomaly severity weight configuration
ANOMALY_SEVERITY_WEIGHT = {
    "critical": 1.0,   # 致命
    "major": 0.8,      # 严重
    "minor": 0.6,      # 次要
    "warning": 0.4,    # 警告
    "info": 0.2        # 信息
}

# LLM configuration (GLM-4.7)
LLM_CONFIG = {
    "model": "glm-4.7",  # Model name
    "api_key": "e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa",  # Replace with your API Key
    "api_url": "https://open.bigmodel.cn/api/coding/paas/v4",
    "temperature": 0.4,    # Reduced for stable results (0.3-0.5)
    "max_tokens": 8192,     # Maximum output length
    "max_retries": 3        # API retry count
}

# LLM_CONFIG = {
#     "model": "deepseek-r1-0528",
#     "api_key": "sk-e8bbbd81c0dc42dfa73d557012d1a3dd",
#     "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
#     "temperature": 0.4,        # Lower temperature for more stable results
#     "max_tokens": 8192,
#     "max_retries": 3           # New: maximum retry attempts
# }

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
        self.analysis_summary = {}  # For final summary
        self.propagation_paths = [] # Fault propagation paths
    
    def _load_kg_data(self):
        """Load knowledge graph JSON data"""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load knowledge graph: {e}")
    
    def _extract_entity_features(self):
        """Extract core entity features (enhanced version)"""
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
                    "last_anomaly_ts": 0,
                    "fault_types": set(),
                    "topology_neighbors": set(),
                    "component_weight": COMPONENT_BASE_WEIGHT.get(component_type, 0.5),
                    # Enhanced features
                    "severity_score": 0.0,
                    "total_duration": 0.0,
                    "business_impact": 0
                }
        
        # 2. Count anomaly occurrences, time, severity and duration
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                entity_id = rel["target"]
                if entity_id in entities:
                    # Count anomalies
                    entities[entity_id]["anomaly_count"] += 1
                    
                    # Record first/last anomaly time
                    ts = rel["properties"]["timestamp"]
                    if ts < entities[entity_id]["first_anomaly_ts"]:
                        entities[entity_id]["first_anomaly_ts"] = ts
                    if ts > entities[entity_id]["last_anomaly_ts"]:
                        entities[entity_id]["last_anomaly_ts"] = ts
                    
                    # Calculate severity score
                    severity = rel["properties"].get("severity", "info")
                    entities[entity_id]["severity_score"] += ANOMALY_SEVERITY_WEIGHT.get(severity, 0.2)
                    
                    # Calculate anomaly duration
                    start_ts = rel["properties"].get("start_ts", ts)
                    end_ts = rel["properties"].get("end_ts", ts)
                    entities[entity_id]["total_duration"] += max(0, end_ts - start_ts)
        
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
        
        # 5. Business impact analysis
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "IMPACTS_BUSINESS":
                entity_id = rel["source"]
                if entity_id in entities:
                    entities[entity_id]["business_impact"] += 1
        
        return entities
    
    def _analyze_causal_relationships(self, entities):
        """Analyze causal relationships between entities (enhanced)"""
        # 1. Build anomaly timeline
        anomaly_timeline = defaultdict(list)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                entity_id = rel["target"]
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
                    # Check topological neighbors
                    neighbors = entities[entity]["topology_neighbors"]
                    if neighbors & prev_entities:
                        source_entity = list(neighbors & prev_entities)[0]
                        propagation_paths.append({
                            "source": source_entity,
                            "target": entity,
                            "timestamp": ts,
                            "confidence": 0.9,
                            "time_str": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                        })
            prev_entities = current_entities
        
        self.propagation_paths = propagation_paths
        return propagation_paths
    
    def _calculate_entity_scores(self, entities):
        """Calculate entity root cause scores (enhanced with causal analysis)"""
        # 1. Normalization parameters
        max_anomaly_count = max([e["anomaly_count"] for e in entities.values()], default=1)
        min_ts = min([e["first_anomaly_ts"] for e in entities.values() if e["first_anomaly_ts"] != float('inf')], default=0)
        max_ts = max([e["first_anomaly_ts"] for e in entities.values() if e["first_anomaly_ts"] != float('inf')], default=1)
        max_neighbors = max([len(e["topology_neighbors"]) for e in entities.values()], default=1)
        max_severity = max([e["severity_score"] for e in entities.values()], default=1)
        max_duration = max([e["total_duration"] for e in entities.values()], default=1)
        max_business_impact = max([e["business_impact"] for e in entities.values()], default=1)
        
        # 2. Analyze causal relationships
        propagation_paths = self._analyze_causal_relationships(entities)
        
        # 3. Build causal weight (source entities get higher scores)
        causal_weight = defaultdict(float)
        for path in propagation_paths:
            causal_weight[path["source"]] += 0.1  # Add 0.1 for each propagation
        
        # 4. Calculate scores for each dimension
        for entity_id, entity in entities.items():
            # Skip sub-entities (only analyze main entities)
            if not entity["is_main"]:
                continue
            
            # 4.1 Basic dimension scores (0-1)
            count_score = entity["anomaly_count"] / max_anomaly_count if max_anomaly_count > 0 else 0
            severity_score = entity["severity_score"] / max_severity if max_severity > 0 else 0
            duration_score = entity["total_duration"] / max_duration if max_duration > 0 else 0
            business_impact_score = entity["business_impact"] / max_business_impact if max_business_impact > 0 else 0
            
            # 4.2 Time priority score (earlier = higher, 0-1)
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
            
            # 4.5 Causal score
            causal_score = min(causal_weight.get(entity_id, 0), 0.5)  # Cap at 0.5
            
            # 4.6 Weighted total score (enhanced)
            total_score = (
                count_score * SCORE_WEIGHTS["anomaly_count"] +
                time_score * SCORE_WEIGHTS["time_priority"] +
                topology_score * SCORE_WEIGHTS["topology_impact"] +
                component_score * SCORE_WEIGHTS["component_weight"] +
                severity_score * 0.1 +          # New: severity weight
                business_impact_score * 0.1 +   # New: business impact weight
                causal_score                    # New: causal weight
            )
            
            # Ensure score is between 0 and 1
            total_score = max(0, min(1, total_score))
            
            # Format first anomaly time
            first_anomaly_time = "N/A"
            if entity["first_anomaly_ts"] != float('inf'):
                first_anomaly_time = datetime.fromtimestamp(entity["first_anomaly_ts"]).strftime("%Y-%m-%d %H:%M:%S")
            
            self.entity_scores[entity_id] = {
                "entity_id": entity_id,
                "component_type": entity["component_type"],
                "total_score": round(total_score, 4),
                # Detailed scores
                "count_score": round(count_score, 4),
                "time_score": round(time_score, 4),
                "topology_score": round(topology_score, 4),
                "component_score": round(component_score, 4),
                "severity_score": round(severity_score, 4),
                "business_impact_score": round(business_impact_score, 4),
                "causal_score": round(causal_score, 4),
                # Raw metrics
                "anomaly_count": entity["anomaly_count"],
                "first_anomaly_time": first_anomaly_time,
                "total_duration": round(entity["total_duration"], 2),
                "business_impact_count": entity["business_impact"],
                "fault_types": list(entity["fault_types"]),
                "neighbor_count": len(entity["topology_neighbors"]),
                "propagation_source_count": len([p for p in propagation_paths if p["source"] == entity_id]),
                "propagation_target_count": len([p for p in propagation_paths if p["target"] == entity_id])
            }
    
    def _filter_root_causes(self):
        """Filter root causes (enhanced rule-based)"""
        # 1. Sort by total score
        sorted_entities = sorted(
            self.entity_scores.values(),
            key=lambda x: x["total_score"],
            reverse=True
        )
        
        # 2. Enhanced rule-based filtering
        threshold = 0.1  # Minimum score threshold
        for entity in sorted_entities:
            if entity["total_score"] < threshold:
                continue
            # Exclude entities with zero anomalies or no fault types
            if entity["anomaly_count"] == 0 and not entity["fault_types"]:
                continue
            
            self.root_causes.append(entity)
        
        # 3. Fallback (return at least 1 root cause)
        if not self.root_causes and sorted_entities:
            self.root_causes.append(sorted_entities[0])
    
    def generate_rca_report(self):
        """Generate programmatic root cause analysis report (enhanced)"""
        # 1. Extract features and calculate scores
        entities = self._extract_entity_features()
        self._calculate_entity_scores(entities)
        self._filter_root_causes()
        
        # 2. Build enhanced report
        report = []
        report.append(f"# Programmatic Root Cause Analysis Report - Cluster {self.cluster_id}")
        report.append(f"**Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Anomalies**: {self.total_anomalies}")
        report.append(f"**Total Entities Analyzed**: {len([e for e in entities.values() if e['is_main']])}")
        report.append(f"**RCA Dimension Weights**: Anomaly Count(40%) + Time Priority(20%) + Topology Impact(25%) + "
                      f"Component Weight(15%) + Severity(10%) + Business Impact(10%) + Causal Propagation")
        report.append("")
        
        # 3. Fault propagation path analysis
        if self.propagation_paths:
            report.append("## Fault Propagation Path Analysis")
            report.append("| Source Entity | Target Entity | Propagation Time | Confidence |")
            report.append("|---------------|---------------|------------------|------------|")
            for path in self.propagation_paths[:10]:  # Show top 10 paths
                report.append(f"| {path['source']} | {path['target']} | {path['time_str']} | {path['confidence']} |")
            report.append("")
        
        # 4. Root cause results (enhanced)
        report.append("## Fault Root Cause Ranking")
        for idx, cause in enumerate(self.root_causes[:5]):  # Show top 5 only
            report.append(f"### Root Cause #{idx+1}")
            report.append(f"- **Entity ID**: {cause['entity_id']}")
            report.append(f"- **Component Type**: {cause['component_type'].upper()}")
            report.append(f"- **Root Cause Confidence**: {cause['total_score']:.4f}")
            report.append(f"- **Anomaly Count**: {cause['anomaly_count']} ({cause['anomaly_count']/self.total_anomalies*100:.1f}%)")
            report.append(f"- **First Anomaly Time**: {cause['first_anomaly_time']}")
            report.append(f"- **Total Anomaly Duration**: {cause['total_duration']}s")
            report.append(f"- **Business Impact**: {cause['business_impact_count']} business lines affected")
            report.append(f"- **Fault Types**: {', '.join(cause['fault_types']) if cause['fault_types'] else 'unknown'}")
            report.append(f"- **Topology Impact Scope**: {cause['neighbor_count']} associated entities")
            report.append(f"- **Propagation Role**: Source of {cause['propagation_source_count']} faults, Target of {cause['propagation_target_count']} faults")
            report.append(f"- **Dimension Score Breakdown**:")
            report.append(f"  - Anomaly Count Score: {cause['count_score']:.4f}")
            report.append(f"  - Time Priority Score: {cause['time_score']:.4f}")
            report.append(f"  - Topology Impact Score: {cause['topology_score']:.4f}")
            report.append(f"  - Component Weight Score: {cause['component_score']:.4f}")
            report.append(f"  - Severity Score: {cause['severity_score']:.4f}")
            report.append(f"  - Business Impact Score: {cause['business_impact_score']:.4f}")
            report.append(f"  - Causal Propagation Score: {cause['causal_score']:.4f}")
            report.append("")
        
        # 5. Fault type analysis
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
        
        # 6. Enhanced remediation recommendations
        report.append("## Remediation Recommendations")
        if self.root_causes:
            primary_cause = self.root_causes[0]
            report.append(f"### Immediate Actions (High Priority)")
            report.append(f"1. Prioritize investigation of {primary_cause['entity_id']} (confidence: {primary_cause['total_score']:.4f})")
            report.append(f"2. Focus on {', '.join(primary_cause['fault_types'])} issues in {primary_cause['component_type'].upper()} layer")
            report.append(f"3. Check {primary_cause['neighbor_count']} associated entities for cascading failures")
            report.append(f"4. Monitor anomaly severity and duration for {primary_cause['entity_id']}")
            
            report.append(f"\n### Preventive Measures (Long Term)")
            report.append(f"1. Strengthen monitoring for {primary_cause['component_type'].upper()} components with high business impact")
            report.append(f"2. Analyze propagation paths to optimize service dependencies")
            report.append(f"3. Set up alerting rules for anomaly severity > 0.8")
        
        # 7. Generate analysis summary for final JSON
        self.analysis_summary = {
            "cluster_id": self.cluster_id,
            "analysis_time": datetime.now().isoformat(),
            "total_anomalies": self.total_anomalies,
            "total_entities_analyzed": len([e for e in entities.values() if e['is_main']]),
            "primary_root_cause": self.root_causes[0] if self.root_causes else {},
            "top_5_root_causes": self.root_causes[:5],
            "fault_type_distribution": dict(fault_counter),
            "propagation_paths": self.propagation_paths[:10],
            "score_weights": SCORE_WEIGHTS
        }
        
        return "\n".join(report), self.analysis_summary

# ====================== LLM-driven RCA Analyzer ======================
class LLMbasedRCAAnalyzer:
    def __init__(self, kg_json_path, llm_config=None):
        """Initialize LLM-driven root cause analyzer (enhanced)"""
        self.kg_json_path = kg_json_path
        self.kg_data = self._load_kg_data()
        self.cluster_id = self.kg_data["cluster_id"]
        self.total_anomalies = self.kg_data["total_anomalies"]
        self.llm_config = llm_config or LLM_CONFIG
        self.llm_response_dict = None  # Use dict instead of Completion object
        self.llm_raw_content = None    # Store only the content
        self.rca_report = None
        self.analysis_summary = {}  # For final summary
    
    def _load_kg_data(self):
        """Load knowledge graph data"""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load knowledge graph: {e}")
    
    def _convert_kg_to_prompt(self):
        """Convert knowledge graph to optimized LLM prompt (structured reasoning)"""
        # 1. System role definition (critical for structured output)
        prompt = [
            """### System Role (MUST FOLLOW STRICTLY)
You are a senior SRE expert with 10+ years of microservice fault diagnosis experience, specialized in root cause analysis (RCA) for telecom systems.
Your analysis must follow the "5-step RCA methodology":
Step 1: Data Validation - Verify anomaly data completeness and topology relationships;
Step 2: Anomaly Clustering - Group anomalies by time, entity, and fault type;
Step 3: Causal Inference - Identify cause-effect relationships (e.g., DB slow query → API timeout → user complaint);
Step 4: Impact Assessment - Evaluate business impact and propagation scope;
Step 5: Root Cause Confirmation - Rank root causes with confidence score (0-1) and concrete evidence.

### Analysis Constraints
- Focus on data-driven reasoning, avoid guesswork
- Prioritize entities with high anomaly severity, business impact, and propagation influence
- Confidence scores must be based on quantitative evidence (not subjective judgment)
- Output must strictly follow the specified format (NO deviation)"""
        ]
        
        # 2. Analysis context (concise and structured)
        time_span = self.kg_data.get("time_span", {"start": "N/A", "end": "N/A", "duration_sec": 0})
        prompt.append(f"""
### Analysis Context
- Target Cluster: {self.cluster_id}
- Time Window: {time_span['start']} to {time_span['end']} (Duration: {time_span['duration_sec']} seconds)
- Total Anomalies: {self.total_anomalies}
- Entity Distribution: 
  - DB Entities: {len([n for n in self.kg_data['nodes'] if n['label']=='DB'])}
  - OS Entities: {len([n for n in self.kg_data['nodes'] if n['label']=='OS'])}
  - Docker Entities: {len([n for n in self.kg_data['nodes'] if n['label']=='DOCKER'])}""")
        
        # 3. Key entity metrics (structured table)
        prompt.append("\n### Key Entity Metrics (Structured)")
        
        # 3.1 Collect entity statistics
        entity_stats = defaultdict(lambda: {
            "count": 0, "severity": 0.0, "first_ts": float('inf'), 
            "last_ts": 0, "fault_types": set(), "business_impact": 0
        })
        
        # Extract anomaly data
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                entity_id = rel["target"]
                stats = entity_stats[entity_id]
                stats["count"] += 1
                
                # Severity calculation
                severity = rel["properties"].get("severity", "info")
                stats["severity"] += ANOMALY_SEVERITY_WEIGHT.get(severity, 0.2)
                
                # Timestamps
                ts = rel["properties"]["timestamp"]
                if ts < stats["first_ts"]:
                    stats["first_ts"] = ts
                if ts > stats["last_ts"]:
                    stats["last_ts"] = ts
        
        # Extract fault types
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ATTRIBUTE":
                entity_id = rel["source"]
                attr_id = rel["target"]
                for node in self.kg_data["nodes"]:
                    if node["id"] == attr_id and "fault_type" in node["properties"]:
                        entity_stats[entity_id]["fault_types"].add(node["properties"]["fault_type"])
        
        # Extract business impact
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "IMPACTS_BUSINESS":
                entity_id = rel["source"]
                entity_stats[entity_id]["business_impact"] += 1
        
        # 3.2 Generate markdown table
        prompt.append("| Entity ID | Anomaly Count | Total Severity | First Anomaly Time | Business Impact | Fault Types |")
        prompt.append("|-----------|---------------|----------------|--------------------|-----------------|-------------|")
        
        for entity_id, stats in sorted(entity_stats.items(), key=lambda x: (x[1]["severity"], x[1]["count"]), reverse=True):
            if stats["count"] == 0:
                continue
                
            # Format time
            first_time = "N/A"
            if stats["first_ts"] != float('inf'):
                first_time = datetime.fromtimestamp(stats["first_ts"]).strftime("%Y-%m-%d %H:%M:%S")
            
            # Format fault types
            fault_types = ", ".join(stats["fault_types"]) if stats["fault_types"] else "unknown"
            
            prompt.append(
                f"| {entity_id} | {stats['count']} | {stats['severity']:.2f} | {first_time} | "
                f"{stats['business_impact']} | {fault_types[:50]} |"  # Truncate long strings
            )
        
        # 4. Topology and propagation data
        prompt.append("\n### Topology Dependencies")
        topology_rels = defaultdict(list)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "TOPOLOGY_DEPENDS_ON":
                topology_rels[rel["source"]].append(rel["target"])
        
        if topology_rels:
            prompt.append("Entity Dependency List:")
            for src, dsts in list(topology_rels.items())[:20]:  # Limit to 20 entries
                prompt.append(f"- {src} depends on: {', '.join(dsts)}")
        else:
            prompt.append("No topology dependency data available")
        
        # 5. Mandatory output format (strict structure)
        prompt.append("""
### MANDATORY OUTPUT FORMAT (NO DEVIATION ALLOWED)
## 1. Root Cause Ranking (Top 3)
| Rank | Entity ID | Confidence Score (0-1) | Core Reason | Evidence |
|------|-----------|------------------------|-------------|----------|
| 1    | [ENTITY_ID] | [0.0-1.0] | [Clear, concise root cause description] | [Quantitative evidence: anomaly count, severity, propagation role, business impact] |
| 2    | [ENTITY_ID] | [0.0-1.0] | [Clear, concise root cause description] | [Quantitative evidence: anomaly count, severity, propagation role, business impact] |
| 3    | [ENTITY_ID] | [0.0-1.0] | [Clear, concise root cause description] | [Quantitative evidence: anomaly count, severity, propagation role, business impact] |

## 2. Fault Propagation Path Analysis
- Primary Propagation Chain: [Root Entity] → [Anomaly Type] → [Affected Entity 1] → [Affected Entity 2] → [Business Impact]
- Time Sequence: [Time 1 (Root Cause)] → [Time 2 (Propagation)] → [Time 3 (Full Impact)]
- Key Propagation Trigger: [Specific event/metric that started the propagation]

## 3. Business Impact Analysis
- Affected Business Lines: [List of affected business lines]
- Impact Severity: [High/Medium/Low] (based on user impact and service degradation)
- Service Degradation Metrics: [Concrete metrics like response time increase, error rate, etc.]

## 4. Remediation Recommendations (Actionable and Specific)
### Immediate Actions (To be completed within 1 hour)
1. [Specific, actionable step with verification method]
2. [Specific, actionable step with verification method]
3. [Specific, actionable step with verification method]

### Preventive Measures (Long-term optimization)
1. [Structural improvement with implementation timeline]
2. [Monitoring enhancement with specific metrics]
3. [Process optimization with clear ownership]
""")
        
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
        """Generate enhanced LLM-driven root cause analysis report"""
        # 1. Generate optimized prompt
        prompt = self._convert_kg_to_prompt()
        
        # 2. Call LLM with retry
        llm_output = self._call_llm_api(prompt)
        
        # 3. Build final report
        report = []
        report.append(f"# LLM-Driven Root Cause Analysis Report - Cluster {self.cluster_id}")
        report.append(f"**Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Model Used**: {self.llm_config['model']} (Temperature: {self.llm_config['temperature']})")
        report.append(f"**Knowledge Graph Source**: {self.kg_json_path}")
        report.append(f"**Token Usage**: {self.llm_response_dict['usage']['total_tokens'] if self.llm_response_dict else 'N/A'}")
        report.append("="*80)
        report.append("")
        report.append(llm_output)
        
        self.rca_report = "\n".join(report)
        
        # 4. Generate analysis summary
        self.analysis_summary = {
            "cluster_id": self.cluster_id,
            "analysis_time": datetime.now().isoformat(),
            "model_used": self.llm_config['model'],
            "temperature": self.llm_config['temperature'],
            "total_anomalies": self.total_anomalies,
            "llm_response_content": llm_output,
            "token_usage": self.llm_response_dict['usage'] if self.llm_response_dict else {},
            "prompt_char_count": len(prompt)
        }
        
        return self.rca_report, self.analysis_summary
    
    def save_report(self):
        """Save analysis report with robust error handling"""
        if not self.rca_report:
            raise ValueError("Please generate analysis report first")
        
        # Save main report
        report_path = self.kg_json_path.replace("_kg.json", "_llm_rca_report.md")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self.rca_report)
        except Exception as e:
            raise RuntimeError(f"Failed to save LLM report: {e}")
        
        # Save LLM response as JSON (with fallback)
        if self.llm_response_dict:
            response_path = self.kg_json_path.replace("_kg.json", "_llm_response.json")
            try:
                with open(response_path, 'w', encoding='utf-8') as f:
                    json.dump(self.llm_response_dict, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ Warning: Failed to save LLM response JSON: {e}")
                # Fallback to text file
                fallback_path = self.kg_json_path.replace("_kg.json", "_llm_response_raw.txt")
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    f.write(self.llm_raw_content or "No response content")
        
        print(f"✅ LLM-driven RCA report generated: {report_path}")
        return report_path

# ====================== Result Merging & Evaluation ======================
def merge_rca_results(programmatic_summary, llm_summary):
    """Merge programmatic and LLM RCA results (weighted fusion)"""
    merged_summary = {
        "analysis_metadata": {
            "merge_time": datetime.now().isoformat(),
            "programmatic_weight": 0.6,  # Programmatic is more objective
            "llm_weight": 0.4            # LLM provides business context
        },
        "clusters": {}
    }
    
    # Process each cluster
    for cluster_id in programmatic_summary["clusters"].keys():
        if cluster_id not in llm_summary["clusters"]:
            print(f"⚠️ Cluster {cluster_id} not found in LLM results, skipping merge")
            continue
        
        prog_result = programmatic_summary["clusters"][cluster_id]
        llm_result = llm_summary["clusters"][cluster_id]
        
        # 1. Extract programmatic scores (quantitative)
        prog_root_causes = {}
        for rc in prog_result.get("top_5_root_causes", []):
            prog_root_causes[rc["entity_id"]] = rc["total_score"]
        
        # 2. Parse LLM scores from structured output
        llm_root_causes = {}
        llm_content = llm_result.get("llm_response_content", "")
        
        # Extract root cause table using regex
        table_pattern = r"\| Rank \| Entity ID \| Confidence Score \(0-1\) \|.*?\|(.*?)\|"
        matches = re.findall(table_pattern, llm_content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            rows = match.strip().split("\n")
            for row in rows:
                if row.strip() and not row.startswith("|------") and not row.startswith("| Rank"):
                    parts = [p.strip() for p in row.split("|") if p.strip()]
                    if len(parts) >= 3:  # Rank, Entity ID, Confidence Score
                        try:
                            entity_id = parts[1]
                            confidence = float(parts[2])
                            llm_root_causes[entity_id] = confidence
                        except (ValueError, IndexError):
                            continue
        
        # 3. Weighted fusion of scores
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
        
        # 4. Sort merged results
        sorted_merged = sorted(
            merged_root_causes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 5. Compile merged result
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
    """Evaluate RCA accuracy against ground truth (if available)"""
    evaluation = {
        "evaluation_time": datetime.now().isoformat(),
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
        print("⚠️ Ground truth file not provided/found, skipping evaluation")
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
        
        # Check top-1 accuracy
        if top_5_merged and top_5_merged[0] == true_root_cause:
            evaluation["metrics"]["top_1_correct"] += 1
        
        # Check top-3 accuracy
        if true_root_cause in top_5_merged[:3]:
            evaluation["metrics"]["top_3_correct"] += 1
        
        # Per-cluster evaluation
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
    print(f"   Total clusters evaluated: {total}")
    print(f"   Top-1 Accuracy: {evaluation['metrics']['top_1_accuracy']:.2%}")
    print(f"   Top-3 Accuracy: {evaluation['metrics']['top_3_accuracy']:.2%}")
    
    return evaluation

# ====================== Batch Execution Functions ======================
def run_programmatic_analysis(kg_dir, summary_output_path):
    """Run programmatic root cause analysis in batch"""
    programmatic_summary = {
        "analysis_metadata": {
            "analysis_time": datetime.now().isoformat(),
            "score_weights": SCORE_WEIGHTS,
            "component_base_weights": COMPONENT_BASE_WEIGHT,
            "anomaly_severity_weights": ANOMALY_SEVERITY_WEIGHT
        },
        "clusters": {}
    }
    
    # Process all knowledge graph files
    processed_count = 0
    failed_count = 0
    
    for root, dirs, files in os.walk(kg_dir):
        for file in files:
            if file.endswith("_kg.json"):
                kg_path = os.path.join(root, file)
                try:
                    # Run analysis
                    analyzer = ProgrammaticRCAAnalyzer(kg_path)
                    report, cluster_summary = analyzer.generate_rca_report()
                    
                    # Extract cluster name
                    cluster_name = os.path.basename(os.path.dirname(kg_path))
                    
                    # Save individual report
                    report_path = kg_path.replace("_kg.json", "_programmatic_rca_report.md")
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    
                    # Add to summary
                    programmatic_summary["clusters"][cluster_name] = cluster_summary
                    processed_count += 1
                    print(f"✅ Programmatic RCA completed: {report_path}")
                    
                except Exception as e:
                    failed_count += 1
                    print(f"❌ Analysis failed for {kg_path}: {e}")
    
    # Save summary JSON
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(programmatic_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 Programmatic RCA Summary:")
    print(f"   Total files processed: {processed_count}")
    print(f"   Total failures: {failed_count}")
    print(f"   Summary saved to: {summary_output_path}")
    
    return programmatic_summary

def run_llm_analysis(kg_dir, api_key, summary_output_path):
    """Run LLM-driven root cause analysis in batch"""
    llm_config = LLM_CONFIG.copy()
    llm_config["api_key"] = api_key
    
    llm_summary = {
        "analysis_metadata": {
            "analysis_time": datetime.now().isoformat(),
            "model_used": llm_config['model'],
            "temperature": llm_config['temperature'],
            "max_tokens": llm_config['max_tokens'],
            "max_retries": llm_config['max_retries']
        },
        "clusters": {}
    }
    
    # Process all knowledge graph files
    processed_count = 0
    failed_count = 0
    
    for root, dirs, files in os.walk(kg_dir):
        for file in files:
            if file.endswith("_kg.json"):
                kg_path = os.path.join(root, file)
                try:
                    # Run LLM analysis
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
                    print(f"❌ LLM analysis failed for {kg_path}: {e}")
    
    # Save summary JSON
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(llm_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 LLM RCA Summary:")
    print(f"   Total files processed: {processed_count}")
    print(f"   Total failures: {failed_count}")
    print(f"   Summary saved to: {summary_output_path}")
    
    return llm_summary

# ====================== Main Execution ======================
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Root Cause Analysis for Telecom Cluster Anomalies")
    parser.add_argument("--date_online", required=True, help="Date string like 2020_04_11")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0000_0030")
    parser.add_argument("--output_folder_name", type=str, default="1216",
                        help="Output folder name (e.g., experiment ID)")
    parser.add_argument("--api_key", type=str, default="e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa",
                        help="GLM API key (override default)")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Path to ground truth JSON file for evaluation (optional)")
    args = parser.parse_args()
    
    # Base paths
    base_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"
    kg_root_dir = f"{base_dir}/knowledge_graphs/{args.date_online}_{args.output_suffix}"
    
    # Validate input directory
    if not os.path.exists(kg_root_dir):
        print(f"❌ Error: Knowledge graph directory not found - {kg_root_dir}")
        return
    
    # Create output directory if needed
    os.makedirs(base_dir, exist_ok=True)
    
    # Define output paths
    programmatic_summary_path = f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_programmatic_rca_summary.json"
    llm_summary_path = f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_llm_rca_summary.json"
    merged_summary_path = f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_merged_rca_summary.json"
    evaluation_path = f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_rca_evaluation.json"
    
    # 1. Run programmatic analysis
    print("\n" + "="*60)
    print("Starting Programmatic RCA Analysis")
    print("="*60)
    programmatic_summary = run_programmatic_analysis(kg_root_dir, programmatic_summary_path)
    
    # 2. Run LLM-driven analysis
    print("\n" + "="*60)
    print("Starting LLM-driven RCA Analysis")
    print("="*60)
    llm_summary = run_llm_analysis(kg_root_dir, args.api_key, llm_summary_path)
    
    # 3. Merge results
    print("\n" + "="*60)
    print("Merging RCA Results")
    print("="*60)
    merged_summary = merge_rca_results(programmatic_summary, llm_summary)
    
    # Save merged results
    with open(merged_summary_path, 'w', encoding='utf-8') as f:
        json.dump(merged_summary, f, ensure_ascii=False, indent=2)
    print(f"✅ Merged RCA summary saved to: {merged_summary_path}")
    
    # 4. Evaluate results (if ground truth provided)
    if args.ground_truth:
        print("\n" + "="*60)
        print("Evaluating RCA Results")
        print("="*60)
        evaluation = evaluate_rca_results(merged_summary, args.ground_truth)
        
        # Save evaluation results
        with open(evaluation_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, ensure_ascii=False, indent=2)
        print(f"✅ RCA evaluation saved to: {evaluation_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("All Analyses Completed Successfully!")
    print("="*60)
    print(f"📊 Programmatic summary: {programmatic_summary_path}")
    print(f"📊 LLM summary: {llm_summary_path}")
    print(f"📊 Merged summary: {merged_summary_path}")
    if args.ground_truth:
        print(f"📊 Evaluation report: {evaluation_path}")

if __name__ == "__main__":
    main()
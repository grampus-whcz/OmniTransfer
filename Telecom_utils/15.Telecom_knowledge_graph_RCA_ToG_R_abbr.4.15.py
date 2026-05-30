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
from zhipuai import ZhipuAI
import sys

# virtual environment: conda faiss-env

# Anomaly severity weight configuration
ANOMALY_SEVERITY_WEIGHT = {
    "critical": 1.0,   # 致命
    "major": 0.8,      # 严重
    "minor": 0.6,      # 次要
    "warning": 0.4,    # 警告
    "info": 0.2        # 信息
}

# LLM配置（GLM-4.7）
LLM_CONFIG = {
    "model": "glm-4.7",
    "api_key": "xxx",
    "api_url": "https://xxx",
    "temperature": 0.7,
    "max_tokens": 8192
}

BEIJING_TZ = timezone(timedelta(hours=8))

TOG_SEARCH_CONFIG = {
    "depth": 3,
    "width": 3,
    "num_retain_entity": 5,
    "max_candidate_relations": 6,
    "max_candidate_entities": 6
}

TOG_ENTITY_LABELS = {"DB", "OS", "DOCKER", "OS_Sub", "DOCKER_Sub"}


def call_llm_api(llm_config, messages, cluster_id="unknown"):
    """Shared LLM caller used by the ToG-style RCA pipeline."""
    temperature = llm_config.get("temperature", 0.4)
    max_output_tokens = llm_config.get("max_tokens", 8192)
    max_retries = llm_config.get("max_retries", 3)

    if "glm" in llm_config["model"]:
        client = ZhipuAI(
            api_key=llm_config["api_key"],
            base_url=llm_config.get("api_base", "https://open.bigmodel.cn/api/coding/paas/v4")
        )
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=llm_config["model"],
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    top_p=0.95
                )
                content = response.choices[0].message.content
                
                # Print token usage
                total_tokens = getattr(response.usage, "total_tokens", 0)
                print(f"✅ LLM API ({llm_config['model']}) call successful - Cluster {cluster_id} (Tokens: {total_tokens})")
                
                return {
                    "model": llm_config["model"],
                    "content": content,
                    "usage": {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens": getattr(response.usage, "total_tokens", 0)
                    },
                    "created_at": datetime.now(BEIJING_TZ).isoformat(),
                    "temperature": temperature,
                    "cluster_id": cluster_id
                }
            except Exception as e:
                if retry == max_retries - 1:
                    raise RuntimeError(f"GLM API call failed for {cluster_id}: {e}")
                time.sleep(2 ** retry)
    else:
        from openai import OpenAI

        client = OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["api_base"]
        )
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=llm_config["model"],
                    messages=messages,
                    temperature=temperature
                )
                content = response.choices[0].message.content
                
                # Print token usage
                total_tokens = getattr(response.usage, "total_tokens", 0)
                print(f"✅ LLM API ({llm_config['model']}) call successful - Cluster {cluster_id} (Tokens: {total_tokens})")
                
                return {
                    "model": llm_config["model"],
                    "content": content,
                    "usage": {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens": getattr(response.usage, "total_tokens", 0)
                    },
                    "created_at": datetime.now(BEIJING_TZ).isoformat(),
                    "temperature": temperature,
                    "cluster_id": cluster_id
                }
            except Exception as e:
                if retry == max_retries - 1:
                    raise RuntimeError(f"{llm_config['model']} API call failed for {cluster_id}: {e}")
                time.sleep(2 ** retry)


def _extract_json_object(text):
    """Extract the first JSON object from a model response."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_scores(raw_scores, size):
    if size <= 0:
        return []
    if not raw_scores:
        return [1.0 / size] * size
    clipped = [max(0.0, float(score)) for score in raw_scores[:size]]
    if len(clipped) < size:
        clipped.extend([0.0] * (size - len(clipped)))
    total = sum(clipped)
    if total <= 0:
        return [1.0 / size] * size
    return [score / total for score in clipped]

class TelecomToGAnalyzer:
    """ToG-R: Lightweight Think-on-Graph (ICLR2024).
    Drastically Reduce LLM Calls, Cut Token Usage by Over 90%, While Retaining Strong RCA Performance."""

    def __init__(self, kg_json_path, kg_data, llm_config=None):
        self.kg_json_path = kg_json_path
        self.kg_data = kg_data
        self.cluster_id = kg_data.get("cluster_id", "unknown")
        self.total_anomalies = kg_data.get("total_anomalies", 0)
        self.llm_config = llm_config or LLM_CONFIG
        self.cache_path = kg_json_path.replace("_kg.json", "_tog_cache.json")

        # === ToG-R 核心配置（低成本）
        self.use_tog_r = True  # 开启 ToG-R：省 90% LLM
        self.entity_prune_method = "uniform"  # uniform / rule_based（无 LLM）

        self.nodes_by_id = {node["id"]: node for node in kg_data.get("nodes", [])}
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        for rel in kg_data.get("relationships", []):
            src = rel.get("source")
            dst = rel.get("target")
            if src:
                self.out_edges[src].append(rel)
            if dst:
                self.in_edges[dst].append(rel)

        self.entity_ids = self._collect_entity_ids()
        self.entity_profiles = self._build_entity_profiles()
        self.root_question = (
            "Identify the most likely root cause entity in this telecom anomaly cluster using only the knowledge graph."
        )

    def _collect_entity_ids(self):
        entity_ids = set()
        for node in self.kg_data.get("nodes", []):
            if node.get("label") in TOG_ENTITY_LABELS:
                entity_ids.add(node["id"])
        for rel in self.kg_data.get("relationships", []):
            if rel.get("type") == "HAS_ANOMALY" and rel.get("target"):
                entity_ids.add(rel["target"])
        return entity_ids

    def _build_entity_profiles(self):
        profiles = {}
        for entity_id in self.entity_ids:
            node = self.nodes_by_id.get(entity_id, {})
            props = node.get("properties", {})
            profiles[entity_id] = {
                "entity_id": entity_id,
                "label": node.get("label", "UNKNOWN"),
                "component_type": props.get("entity_type", node.get("label", "unknown")).lower(),
                "is_main": props.get("is_main_entity", True),
                "main_entity": props.get("main_entity", entity_id),
                "anomaly_count": 0,
                "severity_score": 0.0,
                "first_anomaly_ts": float("inf"),
                "last_anomaly_ts": 0,
                "fault_types": set(),
                "neighbors": set(),
                "business_impact": 0,
                "key_evidence": []
            }

        for rel in self.kg_data.get("relationships", []):
            t = rel.get("type")
            s, tar = rel.get("source"), rel.get("target")
            p = rel.get("properties", {})
            if t == "HAS_ANOMALY" and tar in profiles:
                e = profiles[tar]
                e["anomaly_count"] += 1
                e["severity_score"] += ANOMALY_SEVERITY_WEIGHT.get(p.get("severity", "info"), 0.2)
                ts = p.get("timestamp", 0)
                e["first_anomaly_ts"] = min(e["first_anomaly_ts"], ts)
                e["last_anomaly_ts"] = max(e["last_anomaly_ts"], ts)
                m = p.get("metric_name") or p.get("metric") or p.get("anomaly_type") or "anomaly"
                e["key_evidence"].append(f"{m} sev={p.get('severity')} ts={ts}")
            elif t == "HAS_ATTRIBUTE" and s in profiles:
                ft = self.nodes_by_id.get(tar, {}).get("properties", {}).get("fault_type")
                if ft:
                    profiles[s]["fault_types"].add(ft)
            elif t == "TOPOLOGY_DEPENDS_ON":
                if s in profiles and tar: profiles[s]["neighbors"].add(tar)
                if tar in profiles and s: profiles[tar]["neighbors"].add(s)
            elif t == "IMPACTS_BUSINESS" and s in profiles:
                profiles[s]["business_impact"] += 1

        for eid, e in profiles.items():
            if e["first_anomaly_ts"] == float("inf"):
                e["first_anomaly_ts"] = None
            e["fault_types"] = sorted(e["fault_types"])
        return profiles

    def _entity_sort_key(self, entity_id):
        p = self.entity_profiles[entity_id]
        ts = p["first_anomaly_ts"] or float("inf")
        return (-p["anomaly_count"], -p["severity_score"], ts, -len(p["neighbors"]), entity_id)

    def _format_timestamp(self, ts):
        if ts is None: return "N/A"
        return datetime.fromtimestamp(ts, BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")

    def _entity_snapshot(self, entity_id):
        p = self.entity_profiles.get(entity_id)
        if not p:
            n = self.nodes_by_id.get(entity_id, {})
            return f"{entity_id} ({n.get('label','UNKNOWN')})"
        ft = ", ".join(p["fault_types"]) or "unknown"
        return f"{entity_id}[{p['label']}] cnt={p['anomaly_count']} sev={p['severity_score']:.1f} ts={self._format_timestamp(p['first_anomaly_ts'])}"

    def _available_relation_keys(self, entity_id):
        r = set()
        for e in self.out_edges.get(entity_id, []): r.add(f"out::{e.get('type')}")
        for e in self.in_edges.get(entity_id, []): r.add(f"in::{e.get('type')}")
        return sorted(r)

    def _expand_relation(self, entity_id, rkey):
        if "::" not in rkey: return []
        d, t = rkey.split("::", 1)
        edges = self.out_edges.get(entity_id, []) if d == "out" else self.in_edges.get(entity_id, [])
        res = []
        for rel in edges:
            if rel.get("type") != t: continue
            s = entity_id if d == "out" else rel.get("source")
            tar = rel.get("target") if d == "out" else entity_id
            nxt = tar if d == "out" and tar in self.entity_profiles else (s if s in self.entity_profiles else None)
            res.append({
                "rkey": rkey, "rtype": t,
                "triplet": f"({self._entity_snapshot(s)})-[{t}]->({self._entity_snapshot(tar)})",
                "next": nxt, "s": s, "t": tar
            })
        return res

    # === ToG-R：全局一次关系剪枝（整轮深度只调用 1 次 LLM）
    def _prompt_global_relation_prune(self, all_rkeys):
        lines = "\n".join(f"{i+1}. {k}" for i,k in enumerate(all_rkeys))
        return f"""TOG_RELATION_PRUNE (GLOBAL)
Question: {self.root_question}
Cluster: {self.cluster_id}
All relations:
{lines}
Return top 3 useful relations in format:
1. {{key (Score: 0.xx)}}: reason
2. {{key (Score: 0.xx)}}: reason
"""

    def _parse_relation_scores(self, txt, keys):
        import re
        pat = r"{\s*([^()]+?)\s+\(Score:\s*([\d.]+)\)\s*}"
        allowed = set(keys)
        res = []
        for k, sc in re.findall(pat, txt or ""):
            k = k.strip()
            if k in allowed:
                res.append((k, float(sc)))
        if not res:
            return [(k, 1.0/len(keys)) for k in keys[:3]]
        total = sum(s for _,s in res) or 1
        return [(k, s/total) for k,s in res[:3]]

    # === ToG-R：实体打分 100% 无 LLM
    def _get_entity_scores(self, candidates):
        n = len(candidates)
        if n == 0: return []
        if self.entity_prune_method == "uniform":
            return [1.0/n]*n
        elif self.entity_prune_method == "rule_based":
            scores = []
            for c in candidates:
                nid = c["next"]
                if not nid:
                    scores.append(0.0)
                    continue
                p = self.entity_profiles[nid]
                score = p["anomaly_count"]*0.4 + p["severity_score"]*0.3 + (1 if p["is_main"] else 0.2)
                scores.append(max(0.001, score))
            s = sum(scores)
            return [x/s for x in scores]

    def _prompt_sufficiency(self, chains):
        if not self.use_tog_r:
            return "YES"
        txt = "\n".join(chains[-8:])
        return f"""Check if enough for RCA:\n{txt}\nAnswer ONLY YES or NO."""

    def _prompt_final_reasoning(self, states, chains):
        ents = sorted({s["entity_id"] for s in states}, key=self._entity_sort_key)[:6]
        eblk = "\n".join(f"- {self._entity_snapshot(e)}" for e in ents)
        cblk = "\n".join(f"- {c}" for c in chains[:15])
        return f"""RCA using KG. Return JSON only.
Entities:
{eblk}
Chains:
{cblk}
Output:
{{
  "root_cause_ranking": [{{"entity_id":"","confidence":0.0,"reason":"","evidence":[],"fault_types":[],"first_anomaly_time":""}}],
  "fault_propagation": "",
  "summary": ""
}}"""

    def _llm(self, prompt):
        msg = [{"role":"system","content":"RCA expert. Concise."},{"role":"user","content":prompt}]
        return call_llm_api(self.llm_config, msg, self.cluster_id)

    def _check_cache(self):
        if not os.path.exists(self.cache_path): return None
        try:
            with open(self.cache_path) as f:
                c = json.load(f)
            return c if c.get("model_used") == self.llm_config["model"] else None
        except: return None

    def _save_cache(self, data):
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _seed(self):
        ordered = sorted(self.entity_ids, key=self._entity_sort_key)
        top = ordered[:TOG_SEARCH_CONFIG["num_retain_entity"]]
        return [{"entity_id": e, "score":1, "path":[], "vis":[e]} for e in top]

    # ========================== 核心：ToG-R 超低 Token 推理 ==========================
    def run(self):
        cache = self._check_cache()
        if cache: return cache

        states = self._seed()
        chains = []
        trace = []
        llm_trace = []

        for depth in range(1, TOG_SEARCH_CONFIG["depth"]+1):
            next_states = []
            all_relations = set()

            # === 收集所有关系（ToG-R：全局一次剪枝）
            for s in states:
                for r in self._available_relation_keys(s["entity_id"]):
                    all_relations.add(r)
            all_relations = sorted(list(all_relations))[:6]
            if not all_relations: break

            # === 【全局只调用 1 次】关系剪枝（整轮深度 1 次 LLM）
            r_prompt = self._prompt_global_relation_prune(all_relations)
            r_resp = self._llm(r_prompt)
            print(f"Sleep 30 seconds after all relations prune LLM call at depth {depth}")
            time.sleep(30)
            llm_trace.append({"stage":"global_relation_prune","depth":depth,"resp":r_resp})
            selected = self._parse_relation_scores(r_resp.get("content"), all_relations)

            # === 扩展所有实体（无 LLM）
            for state in states:
                eid = state["entity_id"]
                for rk, rw in selected:
                    candidates = self._expand_relation(eid, rk)
                    if not candidates: continue
                    # === 关键：无 LLM 实体打分 ===
                    escore = self._get_entity_scores(candidates)
                    for i, c in enumerate(candidates):
                        nid = c["next"] or eid
                        if c["next"] and c["next"] in state["vis"]:
                            continue
                        scr = state["score"] * rw * escore[i]
                        next_states.append({
                            "entity_id": nid,
                            "score": scr,
                            "path": state["path"] + [c["triplet"]],
                            "vis": state["vis"] + ([nid] if c["next"] else [])
                        })
                        chains.append(c["triplet"])

            if not next_states: break
            uniq = {}
            for s in sorted(next_states, key=lambda x:x["score"], reverse=True):
                k = (s["entity_id"], tuple(s["path"]))
                if k not in uniq:
                    uniq[k] = s
            states = list(uniq.values())[:TOG_SEARCH_CONFIG["width"]]

            # === 充分性判断（1 次 LLM）
            suf_prompt = self._prompt_sufficiency(chains)
            suf_resp = self._llm(suf_prompt)
            print(f"Sleep 30 seconds after chains prune LLM call at depth {depth}")
            time.sleep(30)
            llm_trace.append({"stage":"sufficiency","depth":depth,"resp":suf_resp})
            if suf_resp.get("content","").strip().lower() in ["yes","{yes}"]:
                break

        # === 最终推理（1 次 LLM）
        final_prompt = self._prompt_final_reasoning(states, chains)
        final_resp = self._llm(final_prompt)
        print(f"Sleep 30 seconds after final reasoning LLM call")
        time.sleep(30)
        llm_trace.append({"stage":"final","resp":final_resp})
        j = _extract_json_object(final_resp.get("content",""))

        # 构造结果
        rank = []
        for item in (j or {}).get("root_cause_ranking", []):
            eid = item.get("entity_id")
            if eid not in self.entity_profiles: continue
            p = self.entity_profiles[eid]
            rank.append({
                "rank": len(rank)+1,
                "entity_id": eid,
                "component_type": p["component_type"],
                "confidence": round(float(item.get("confidence",0)),4),
                "reason": item.get("reason",""),
                "evidence": item.get("evidence",[]),
                "fault_types": item.get("fault_types") or p["fault_types"],
                "first_anomaly_time": item.get("first_anomaly_time") or self._format_timestamp(p["first_anomaly_ts"]),
                "anomaly_count": p["anomaly_count"]
            })

        if not rank:
            fallback = sorted(self.entity_ids, key=self._entity_sort_key)[:3]
            for i,eid in enumerate(fallback,1):
                p = self.entity_profiles[eid]
                rank.append({
                    "rank":i, "entity_id":eid, "component_type":p["component_type"],
                    "confidence": round(1.0 - 0.2*(i-1),4),
                    "reason":"Fallback (ToG-R rule based)",
                    "evidence": p["key_evidence"][:2],
                    "fault_types": p["fault_types"],
                    "first_anomaly_time": self._format_timestamp(p["first_anomaly_ts"]),
                    "anomaly_count": p["anomaly_count"]
                })

        res = {
            "analysis_method": "tog-r (low-token)",
            "cluster_id": self.cluster_id,
            "model_used": self.llm_config["model"],
            "primary_root_cause": rank[0],
            "top_5_root_causes": rank[:5],
            "reasoning_chains": chains[:15],
            "fault_propagation": (j or {}).get("fault_propagation",""),
            "summary": (j or {}).get("summary",""),
            "llm_trace": llm_trace,
            "final_response": final_resp
        }
        self._save_cache(res)
        return res

    def render_programmatic_report(self, res):
        lines = [f"# ToG-R (Low Token) RCA Report {self.cluster_id}"]
        lines.append(f"Model: {res['model_used']} | Method: ToG-R (ICLR2024)")
        for item in res["top_5_root_causes"]:
            lines.append(f"Rank {item['rank']} | {item['entity_id']} | conf={item['confidence']}")
        return "\n".join(lines)

    def render_llm_report(self, res):
        return self.render_programmatic_report(res)

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
                    
        elif "MiniMax-M2.5" in self.llm_config['model'] or "DeepSeek" in self.llm_config['model'] or "deepseek" in self.llm_config['model'] or "qwen" in self.llm_config['model'] or "Qwen3" in self.llm_config['model'] or "qwen" in self.llm_config['model']:
            
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
        """Generate an LLM report backed by the shared ToG-style RCA engine."""
        tog = TelecomToGAnalyzer(self.kg_json_path, self.kg_data, self.llm_config)
        self.analysis_summary = tog.run()
        self.llm_response_dict = self.analysis_summary.get("final_response")
        self.llm_raw_content = (self.llm_response_dict or {}).get("content")
        self.rca_report = tog.render_llm_report(self.analysis_summary)
        self.analysis_summary["llm_response_content"] = self.llm_raw_content or ""
        self.analysis_summary["token_usage"] = (self.llm_response_dict or {}).get("usage", {})
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

def run_llm_analysis(kg_dir, api_key, summary_output_path):
    """Run LLM-driven root cause analysis in batch"""
    llm_config = LLM_CONFIG.copy()
    # llm_config["api_key"] = api_key
    
    llm_summary = {
        "analysis_metadata": {
            "analysis_time": datetime.now().isoformat(),
            "analysis_method": "tog",
            "model_used": llm_config['model'],
            "temperature": llm_config['temperature'],
            "max_tokens": llm_config['max_tokens'],
            "max_retries": llm_config['max_retries'],
            "search_config": TOG_SEARCH_CONFIG
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

class FlushFile:
    def __init__(self, file):
        self.file = file
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()  # 强制立即写入硬盘
        
    def flush(self):
        self.file.flush()

# ====================== Main Execution ======================
def main():
           
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Root Cause Analysis for Telecom Cluster Anomalies")
    parser.add_argument("--date_online", required=True, help="Date string like 2020_04_11")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0000_0030")
    parser.add_argument("--output_folder_name", type=str, default="1216",
                        help="Output folder name (e.g., experiment ID)")
    parser.add_argument("--api_key", type=str, default="xxx",
                        help="GLM API key (override default)")
    args = parser.parse_args()
    
    # Base paths
    base_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new6/{args.output_folder_name}"
    kg_root_dir = f"{base_dir}/knowledge_graphs/{args.date_online}_{args.output_suffix}"
    
    # Validate input directory
    if not os.path.exists(kg_root_dir):
        print(f"❌ Error: Knowledge graph directory not found - {kg_root_dir}")
        return
    
    # Create output directory if needed
    os.makedirs(base_dir, exist_ok=True)
    
    # Define output paths
    llm_summary_path = f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_llm_rca_summary.json"
    
    # # 0. log redirection setup
    # import sys
    # # 1) Open the log file (use 'a' mode for appending to avoid overwriting existing logs; 'w' mode means overwriting)
    # log_file = open(f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_knowledge_graph.log", 'a', encoding='utf-8')

    # # 2) Redirect both standard output (print content) and standard error (error messages)
    # sys.stdout = FlushFile(log_file)
    # sys.stderr = FlushFile(log_file)
    
    # 1. Run LLM-driven analysis
    print("\n" + "="*60)
    print("Starting LLM-driven RCA Analysis")
    print("="*60)
    llm_summary = run_llm_analysis(kg_root_dir, args.api_key, llm_summary_path)
    
    # Final summary
    print("\n" + "="*60)
    print("All Analyses Completed Successfully!")
    print("="*60)
    print(f"📊 LLM summary: {llm_summary_path}")

if __name__ == "__main__":
    main()

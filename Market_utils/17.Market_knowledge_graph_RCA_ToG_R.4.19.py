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

# ====================== Global Configuration ======================
# Root cause analysis weight configuration (enhanced for Market microservice architecture)
# SCORE_WEIGHTS = {
#     "anomaly_count": 0.3,        # Weight for anomaly count (reduced from 0.4)
#     "time_priority": 0.15,       # Weight for time priority (earlier is higher)
#     "topology_impact": 0.2,      # Weight for topology impact scope
#     "component_weight": 0.15,    # Weight for component hierarchy
#     "severity_score": 0.1,       # NEW: Weight for anomaly severity
#     "business_impact": 0.05,     # NEW: Weight for business impact
#     "causal_propagation": 0.05   # NEW: Weight for causal propagation
# }

SCORE_WEIGHTS = {
    "anomaly_count": 0.1,        # Weight for anomaly count (reduced from 0.4)
    "time_priority": 0.05,       # Weight for time priority (earlier is higher)
    "topology_impact": 0.6,      # Weight for topology impact scope
    "component_weight": 0.05,    # Weight for component hierarchy
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

# ToG Search Config
TOG_SEARCH_CONFIG = {
    "depth": 3,
    "width": 3,
    "num_retain_entity": 5,
    "max_candidate_relations": 6,
    "max_candidate_entities": 6
}

# Market KG Entity Labels
TOG_ENTITY_LABELS = {"OS", "K8S", "ES", "OS_Sub", "K8S_Sub", "unknown"}

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
                total_tokens = getattr(response.usage, "total_tokens", 0)
                print(f"✅ LLM API ({llm_config['model']}) call successful - Cluster {cluster_id} (Tokens: {total_tokens})")
                return {
                    "model": llm_config["model"],
                    "content": content,
                    "usage": {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens": total_tokens
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
        client = OpenAI(api_key=llm_config["api_key"], base_url=llm_config["api_base"])
        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=llm_config["model"],
                    messages=messages,
                    temperature=temperature
                )
                content = response.choices[0].message.content
                total_tokens = getattr(response.usage, "total_tokens", 0)
                print(f"✅ LLM API ({llm_config['model']}) call successful - Cluster {cluster_id} (Tokens: {total_tokens})")
                return {
                    "model": llm_config["model"],
                    "content": content,
                    "usage": {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                        "total_tokens": total_tokens
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
                candidate = text[start:idx+1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None

def _normalize_scores(raw_scores, size):
    if size <= 0:
        return []
    if not raw_scores:
        return [1.0/size]*size
    clipped = [max(0.0, float(s)) for s in raw_scores[:size]]
    if len(clipped) < size:
        clipped.extend([0.0]*(size-len(clipped)))
    total = sum(clipped)
    if total <= 0:
        return [1.0/size]*size
    return [s/total for s in clipped]

# ====================== Market ToG-R Analyzer (核心升级) ======================
class MarketToGAnalyzer:
    def __init__(self, kg_json_path, kg_data, llm_config=None):
        self.kg_json_path = kg_json_path
        self.kg_data = kg_data
        self.cluster_id = kg_data.get("cluster_id", "unknown")
        self.total_anomalies = kg_data.get("total_anomalies", 0)
        self.llm_config = llm_config or LLM_CONFIG
        self.cache_path = kg_json_path.replace("_kg.json", "_tog_cache.json")

        self.use_tog_r = True
        self.entity_prune_method = "rule_based"

        self.nodes_by_id = {n["id"]: n for n in kg_data.get("nodes", [])}
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        for rel in kg_data.get("relationships", []):
            s, t = rel.get("source"), rel.get("target")
            if s:
                self.out_edges[s].append(rel)
            if t:
                self.in_edges[t].append(rel)

        self.entity_ids = self._collect_entity_ids()
        self.entity_profiles = self._build_entity_profiles()
        self.root_question = "Identify root cause for Market microservice anomaly using knowledge graph."

    def _identify_service_layer(self, entity_name):
        el = entity_name.lower()
        if 'elasticsearch' in el or 'es' in el:
            return "elasticsearch"
        elif 'kafka' in el or 'rabbitmq' in el or 'mq' in el:
            return "mq"
        elif 'springboot' in el or 'spring' in el:
            return "business"
        elif 'memcached' in el:
            return "cache"
        elif 'nginx' in el and 'gateway' in el:
            return "gateway"
        elif 'k8s' in el or 'kubernetes' in el:
            return "container"
        elif 'nginx' in el and 'entry' in el:
            return "entry_point"
        else:
            return "unknown"

    def _collect_entity_ids(self):
        eids = set()
        for n in self.kg_data.get("nodes", []):
            if n.get("label") in TOG_ENTITY_LABELS:
                eids.add(n["id"])
        for rel in self.kg_data.get("relationships", []):
            if rel.get("type") == "HAS_ANOMALY" and rel.get("source"):
                eids.add(rel["source"])
        return eids

    def _build_entity_profiles(self):
        profiles = {}
        for eid in self.entity_ids:
            n = self.nodes_by_id.get(eid, {})
            p = n.get("properties", {})
            en = p.get("entity_id", eid)
            ct = self._identify_service_layer(en)
            profiles[eid] = {
                "entity_id": eid,
                "entity_name": en,
                "label": n.get("label", "UNKNOWN"),
                "component_type": ct,
                "is_main": p.get("is_main_entity", True),
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
            rp = rel.get("properties", {})
            if t == "HAS_ANOMALY" and s in profiles:
                e = profiles[s]
                e["anomaly_count"] += 1
                sev = rp.get("severity", "info")
                e["severity_score"] += ANOMALY_SEVERITY_WEIGHT.get(sev, 0.2)
                ts = rp.get("timestamp", 0)
                e["first_anomaly_ts"] = min(e["first_anomaly_ts"], ts)
                e["last_anomaly_ts"] = max(e["last_anomaly_ts"], ts)
                m = rp.get("metric_name") or rp.get("anomaly_type") or "anomaly"
                e["key_evidence"].append(f"{m} sev={sev}")

            elif t == "HAS_ATTRIBUTE" and s in profiles:
                attr_node = self.nodes_by_id.get(tar, {})
                ft = attr_node.get("properties", {}).get("fault_type")
                if ft:
                    profiles[s]["fault_types"].add(ft)

            elif t == "TOPOLOGY_DEPENDS_ON":
                if s in profiles and tar:
                    profiles[s]["neighbors"].add(tar)
                if tar in profiles and s:
                    profiles[tar]["neighbors"].add(s)

            elif t == "IMPACTS_BUSINESS" and s in profiles:
                profiles[s]["business_impact"] += 1

        for eid, e in profiles.items():
            if e["first_anomaly_ts"] == float("inf"):
                e["first_anomaly_ts"] = None
            e["fault_types"] = sorted(e["fault_types"])
        return profiles

    def _entity_sort_key(self, eid):
        p = self.entity_profiles[eid]
        ts = p["first_anomaly_ts"] or float("inf")
        return (-p["anomaly_count"], -p["severity_score"], ts, -len(p["neighbors"]), eid)

    def _format_ts(self, ts):
        if not ts:
            return "N/A"
        return datetime.fromtimestamp(ts, BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")

    def _entity_snapshot(self, eid):
        p = self.entity_profiles.get(eid)
        if not p:
            return eid
        return f"{eid}[{p['component_type']}] cnt={p['anomaly_count']} sev={p['severity_score']:.1f}"

    def _available_relation_keys(self, eid):
        r = set()
        for e in self.out_edges.get(eid, []):
            r.add(f"out::{e.get('type')}")
        for e in self.in_edges.get(eid, []):
            r.add(f"in::{e.get('type')}")
        return sorted(r)

    def _expand_relation(self, eid, rkey):
        if "::" not in rkey:
            return []
        d, t = rkey.split("::", 1)
        edges = self.out_edges.get(eid, []) if d == "out" else self.in_edges.get(eid, [])
        res = []
        for rel in edges:
            if rel.get("type") != t:
                continue
            s = eid if d == "out" else rel.get("source")
            tar = rel.get("target") if d == "out" else eid
            nxt = tar if d == "out" and tar in self.entity_profiles else (s if s in self.entity_profiles else None)
            res.append({
                "rkey": rkey, "rtype": t,
                "triplet": f"({self._entity_snapshot(s)})-[{t}]->({self._entity_snapshot(tar)})",
                "next": nxt, "s": s, "t": tar
            })
        return res

    def _prompt_global_relation_prune(self, all_rkeys):
        lines = "\n".join(f"{i+1}. {k}" for i,k in enumerate(all_rkeys))
        return f"""TOG_RELATION_PRUNE
Question: {self.root_question}
Cluster: {self.cluster_id}
Relations:
{lines}
Return top 3 useful relations:
1. {{key (Score: 0.xx)}}: reason
2. {{key (Score: 0.xx)}}: reason
"""

    def _parse_relation_scores(self, txt, keys):
        pat = r"{\s*([^()]+?)\s+\(Score:\s*([\d.]+)\)\s*}"
        allowed = set(keys)
        res = []
        for k, sc in re.findall(pat, txt or ""):
            k = k.strip()
            if k in allowed:
                res.append((k, float(sc)))
        if not res:
            return [(k,1.0/len(keys)) for k in keys[:3]]
        total = sum(s for _,s in res) or 1
        return [(k,s/total) for k,s in res[:3]]

    def _get_entity_scores(self, candidates):
        scores = []
        for c in candidates:
            nid = c["next"]
            if not nid or nid not in self.entity_profiles:
                scores.append(0.0)
                continue
            p = self.entity_profiles[nid]
            score = p["anomaly_count"]*0.4 + p["severity_score"]*0.3 + (1 if p["is_main"] else 0.2)
            scores.append(max(0.001, score))
        s = sum(scores) or 1e-6
        return [x/s for x in scores]

    def _prompt_sufficiency(self, chains):
        txt = "\n".join(chains[-8:])
        return f"Is enough for RCA? Answer YES or NO.\n{txt}"

    def _prompt_final_reasoning(self, states, chains):
        ents = sorted({s["entity_id"] for s in states}, key=self._entity_sort_key)[:6]
        eblk = "\n".join(f"- {self._entity_snapshot(e)}" for e in ents)
        cblk = "\n".join(f"- {c}" for c in chains[:15])
        return f"""Market microservice RCA. Return JSON only.
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
        msg = [{"role":"system","content":"RCA expert for Market microservice."},{"role":"user","content":prompt}]
        return call_llm_api(self.llm_config, msg, self.cluster_id)

    def _check_cache(self):
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path) as f:
                c = json.load(f)
            return c if c.get("model_used") == self.llm_config["model"] else None
        except:
            return None

    def _save_cache(self, data):
        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _seed(self):
        ordered = sorted(self.entity_ids, key=self._entity_sort_key)
        top = ordered[:TOG_SEARCH_CONFIG["num_retain_entity"]]
        return [{"entity_id": e, "score":1, "path":[], "vis":[e]} for e in top]

    def run(self):
        cache = self._check_cache()
        if cache:
            return cache

        states = self._seed()
        chains = []
        llm_trace = []

        for depth in range(1, TOG_SEARCH_CONFIG["depth"]+1):
            next_states = []
            all_rels = set()
            for s in states:
                for r in self._available_relation_keys(s["entity_id"]):
                    all_rels.add(r)
            all_rels = sorted(list(all_rels))[:6]
            if not all_rels:
                break

            r_prompt = self._prompt_global_relation_prune(all_rels)
            r_resp = self._llm(r_prompt)
            print(f"[ToG] Sleep 30s after relation prune (depth {depth})")
            time.sleep(30)
            llm_trace.append({"stage":"rel_prune","depth":depth,"resp":r_resp})
            selected = self._parse_relation_scores(r_resp.get("content"), all_rels)

            for state in states:
                eid = state["entity_id"]
                for rk, rw in selected:
                    candidates = self._expand_relation(eid, rk)
                    if not candidates:
                        continue
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

            if not next_states:
                break
            uniq = {}
            for s in sorted(next_states, key=lambda x:x["score"], reverse=True):
                k = (s["entity_id"], tuple(s["path"]))
                if k not in uniq:
                    uniq[k] = s
            states = list(uniq.values())[:TOG_SEARCH_CONFIG["width"]]

            suf_prompt = self._prompt_sufficiency(chains)
            suf_resp = self._llm(suf_prompt)
            print(f"[ToG] Sleep 30s after sufficiency check (depth {depth})")
            time.sleep(30)
            llm_trace.append({"stage":"sufficiency","depth":depth,"resp":suf_resp})
            if suf_resp.get("content","").strip().lower() in ["yes","{yes}"]:
                break

        final_prompt = self._prompt_final_reasoning(states, chains)
        final_resp = self._llm(final_prompt)
        print(f"[ToG] Sleep 30s after final reasoning")
        time.sleep(30)
        llm_trace.append({"stage":"final","resp":final_resp})
        j = _extract_json_object(final_resp.get("content",""))

        rank = []
        for item in (j or {}).get("root_cause_ranking", []):
            eid = item.get("entity_id")
            if eid not in self.entity_profiles:
                continue
            p = self.entity_profiles[eid]
            rank.append({
                "rank": len(rank)+1,
                "entity_id": eid,
                "entity_name": p["entity_name"],
                "component_type": p["component_type"],
                "confidence": round(float(item.get("confidence",0)),4),
                "reason": item.get("reason",""),
                "evidence": item.get("evidence",[]),
                "fault_types": item.get("fault_types") or p["fault_types"],
                "first_anomaly_time": item.get("first_anomaly_time") or self._format_ts(p["first_anomaly_ts"]),
                "anomaly_count": p["anomaly_count"]
            })

        if not rank:
            fallback = sorted(self.entity_ids, key=self._entity_sort_key)[:3]
            for i,eid in enumerate(fallback,1):
                p = self.entity_profiles[eid]
                rank.append({
                    "rank":i, "entity_id":eid, "entity_name":p["entity_name"],
                    "component_type":p["component_type"], "confidence": round(1.0-0.2*(i-1),4),
                    "reason":"Fallback (ToG rule based)", "evidence":p["key_evidence"][:2],
                    "fault_types":p["fault_types"], "first_anomaly_time":self._format_ts(p["first_anomaly_ts"]),
                    "anomaly_count":p["anomaly_count"]
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

# ====================== Programmatic RCA Analyzer ======================
class ProgrammaticRCAAnalyzer:
    def __init__(self, kg_json_path):
        self.kg_json_path = kg_json_path
        self.kg_data = self._load_kg_data()
        self.cluster_id = self.kg_data["cluster_id"]
        self.total_anomalies = self.kg_data["total_anomalies"]
        self.entity_scores = {}
        self.root_causes = []
        self.analysis_summary = {}
        self.propagation_paths = []

    def _load_kg_data(self):
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load KG: {e}")

    def _identify_service_layer(self, entity_name):
        el = entity_name.lower()
        if 'elasticsearch' in el or 'es' in el:
            return "elasticsearch"
        elif 'kafka' in el or 'rabbitmq' in el or 'mq' in el:
            return "mq"
        elif 'springboot' in el or 'spring' in el:
            return "business"
        elif 'memcached' in el:
            return "cache"
        elif 'nginx' in el and 'gateway' in el:
            return "gateway"
        elif 'k8s' in el or 'kubernetes' in el:
            return "container"
        elif 'nginx' in el and 'entry' in el:
            return "entry_point"
        else:
            return "unknown"

    def _extract_entity_features(self):
        entities = {}
        for node in self.kg_data["nodes"]:
            if node["label"] in TOG_ENTITY_LABELS:
                eid = node["id"]
                en = node["properties"].get("entity_id", eid)
                is_main = node["properties"].get("is_main_entity", True)
                ct = self._identify_service_layer(en)
                entities[eid] = {
                    "id": eid, "entity_name": en, "component_type": ct, "is_main": is_main,
                    "anomaly_count":0, "first_anomaly_ts":float('inf'), "last_anomaly_ts":0,
                    "fault_types":set(), "topology_neighbors":set(),
                    "component_weight": COMPONENT_BASE_WEIGHT.get(ct,0.5),
                    "severity_score":0.0, "total_duration":0.0, "business_impact":0
                }

        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                eid = rel["source"]
                if eid in entities:
                    entities[eid]["anomaly_count"] +=1
                    ts = rel["properties"]["timestamp"]
                    entities[eid]["first_anomaly_ts"] = min(entities[eid]["first_anomaly_ts"], ts)
                    entities[eid]["last_anomaly_ts"] = max(entities[eid]["last_anomaly_ts"], ts)
                    sev = rel["properties"].get("severity","info")
                    entities[eid]["severity_score"] += ANOMALY_SEVERITY_WEIGHT.get(sev,0.2)

        attr_map = {}
        for n in self.kg_data["nodes"]:
            if n["label"]=="AnomalyAttribute" and "fault_type" in n["properties"]:
                attr_map[n["id"]] = n["properties"]["fault_type"]
        for rel in self.kg_data["relationships"]:
            if rel["type"]=="HAS_ATTRIBUTE":
                eid, aid = rel["source"], rel["target"]
                if eid in entities and aid in attr_map:
                    entities[eid]["fault_types"].add(attr_map[aid])

        for rel in self.kg_data["relationships"]:
            if rel["type"]=="TOPOLOGY_DEPENDS_ON":
                s,t = rel["source"], rel["target"]
                if s in entities: entities[s]["topology_neighbors"].add(t)
                if t in entities: entities[t]["topology_neighbors"].add(s)

        for rel in self.kg_data["relationships"]:
            if rel["type"]=="IMPACTS_BUSINESS":
                eid = rel["source"]
                if eid in entities: entities[eid]["business_impact"] +=1
        return entities

    def _analyze_causal_relationships(self, entities):
        timeline = defaultdict(list)
        for rel in self.kg_data["relationships"]:
            if rel["type"]=="HAS_ANOMALY":
                timeline[rel["properties"]["timestamp"]].append(rel["source"])
        paths = []
        prev = set()
        for ts in sorted(timeline.keys()):
            cur = set(timeline[ts])
            for e in cur:
                if e not in prev and e in entities:
                    common = entities[e]["topology_neighbors"] & prev
                    if common:
                        paths.append({"source":list(common)[0], "target":e, "ts":ts})
            prev.update(cur)
        self.propagation_paths = paths
        return paths

    # def _calculate_entity_scores(self, entities):
    #     max_anom = max([e["anomaly_count"] for e in entities.values()], default=1)
    #     ts_list = [e["first_anomaly_ts"] for e in entities.values() if e["first_anomaly_ts"]!=float('inf')]
    #     min_ts, max_ts = (min(ts_list), max(ts_list)) if ts_list else (0,1)
    #     max_nei = max([len(e["topology_neighbors"]) for e in entities.values()], default=1)
    #     max_sev = max([e["severity_score"] for e in entities.values()], default=1)
    #     max_biz = max([e["business_impact"] for e in entities.values()], default=1)
    #     max_comp = max(COMPONENT_BASE_WEIGHT.values())

    #     self._analyze_causal_relationships(entities)
    #     causal = defaultdict(float)
    #     for p in self.propagation_paths:
    #         causal[p["source"]] +=0.1

    #     for eid,e in entities.items():
    #         if not e["is_main"]: continue
    #         cs = e["anomaly_count"]/max_anom
            
    #         # 安全修复：时间戳除零保护
    #         if max_ts == min_ts:
    #             ts = 0.0
    #         else:
    #             ts = (max_ts - e["first_anomaly_ts"])/(max_ts-min_ts) if e["first_anomaly_ts"]!=float('inf') else 0
            
    #         tops = len(e["topology_neighbors"])/max_nei
    #         comps = e["component_weight"]/max_comp if max_comp != 0 else 0.0
    #         sevs = e["severity_score"]/max_sev
    #         bizs = e["business_impact"]/max_biz if max_biz != 0 else 0.0
    #         causs = min(causal.get(eid,0),0.5)

    #         total = (cs*0.3 + ts*0.15 + tops*0.2 + comps*0.15 + sevs*0.1 + bizs*0.05 + causs*0.05)
    #         total = max(0, min(1, total))

    #         # ✅ 修复：移除不存在的 _format_ts 方法，直接使用时间戳 / 显示 N/A
    #         if e["first_anomaly_ts"] != float('inf'):
    #             # 直接保留原始时间戳（如需格式化，后续可加，不影响运行）
    #             ft = str(e["first_anomaly_ts"])
    #         else:
    #             ft = "N/A"
                
    #         self.entity_scores[eid] = {
    #             "entity_id":eid, "entity_name":e["entity_name"], "component_type":e["component_type"],
    #             "total_score":round(total,4), "anomaly_count":e["anomaly_count"],
    #             "first_anomaly_time":ft, "fault_types":list(e["fault_types"]),
    #             "confidence":round(total,4), "reason":"Programmatic"
    #         }
    
    def _calculate_entity_scores(self, entities):
        max_anom = max([e["anomaly_count"] for e in entities.values()], default=1)
        ts_list = [e["first_anomaly_ts"] for e in entities.values() if e["first_anomaly_ts"]!=float('inf')]
        min_ts, max_ts = (min(ts_list), max(ts_list)) if ts_list else (0,1)
        max_nei = max([len(e["topology_neighbors"]) for e in entities.values()], default=1)
        max_sev = max([e["severity_score"] for e in entities.values()], default=1)
        max_biz = max([e["business_impact"] for e in entities.values()], default=1)
        max_comp = max(COMPONENT_BASE_WEIGHT.values())

        self._analyze_causal_relationships(entities)
        causal = defaultdict(float)
        for p in self.propagation_paths:
            causal[p["source"]] +=0.1

        for eid,e in entities.items():
            if not e["is_main"]: continue
            cs = e["anomaly_count"]/max_anom
            
            # 安全修复：时间戳除零保护
            if max_ts == min_ts:
                ts = 0.0
            else:
                ts = (max_ts - e["first_anomaly_ts"])/(max_ts-min_ts) if e["first_anomaly_ts"]!=float('inf') else 0
            
            # ✅ 修复：拓扑邻居除零保护
            if max_nei == 0:
                tops = 0.0
            else:
                tops = len(e["topology_neighbors"])/max_nei
            
            comps = e["component_weight"]/max_comp if max_comp != 0 else 0.0
            sevs = e["severity_score"]/max_sev
            bizs = e["business_impact"]/max_biz if max_biz != 0 else 0.0
            causs = min(causal.get(eid,0),0.5)

            total = (cs*0.3 + ts*0.15 + tops*0.2 + comps*0.15 + sevs*0.1 + bizs*0.05 + causs*0.05)
            total = max(0, min(1, total))

            # 修复：移除不存在的 _format_ts 方法，直接使用时间戳 / 显示 N/A
            if e["first_anomaly_ts"] != float('inf'):
                # 直接保留原始时间戳（如需格式化，后续可加，不影响运行）
                ft = str(e["first_anomaly_ts"])
            else:
                ft = "N/A"
                
            self.entity_scores[eid] = {
                "entity_id":eid, "entity_name":e["entity_name"], "component_type":e["component_type"],
                "total_score":round(total,4), "anomaly_count":e["anomaly_count"],
                "first_anomaly_time":ft, "fault_types":list(e["fault_types"]),
                "confidence":round(total,4), "reason":"Programmatic"
            }

    def _filter_root_causes(self):
        s = sorted(self.entity_scores.values(), key=lambda x:x["total_score"], reverse=True)
        self.root_causes = [x for x in s if x["total_score"]>0.01]
        if not self.root_causes and s:
            self.root_causes.append(s[0])

    def generate_rca_report(self):
        entities = self._extract_entity_features()
        self._calculate_entity_scores(entities)
        self._filter_root_causes()
        top5 = self.root_causes[:5]
        for i,x in enumerate(top5): x["rank"]=i+1
        self.analysis_summary = {
            "analysis_method":"programmatic", "cluster_id":self.cluster_id,
            "primary_root_cause": top5[0] if top5 else {}, "top_5_root_causes":top5
        }
        return "DONE", self.analysis_summary

# ====================== LLM-based RCA Analyzer (ToG 驱动) ======================
class LLMbasedRCAAnalyzer:
    def __init__(self, kg_json_path, llm_config=None):
        self.kg_json_path = kg_json_path
        self.kg_data = self._load_kg_data()
        self.cluster_id = self.kg_data["cluster_id"]
        self.llm_config = llm_config or LLM_CONFIG
        self.analysis_summary = {}

    def _load_kg_data(self):
        with open(self.kg_json_path,'r',encoding='utf-8') as f:
            return json.load(f)

    def generate_rca_report(self):
        tog = MarketToGAnalyzer(self.kg_json_path, self.kg_data, self.llm_config)
        self.analysis_summary = tog.run()
        return "ToG Report", self.analysis_summary

    def save_report(self):
        p = self.kg_json_path.replace("_kg.json","_tog_rca_result.json")
        with open(p,'w',encoding='utf-8') as f:
            json.dump(self.analysis_summary, f, indent=2, ensure_ascii=False)
        print(f"✅ ToG RCA saved: {p}")

# ====================== Merge & Evaluation ======================
def merge_rca_results(prog, llm):
    merged = {"analysis_metadata":{"merge_time":datetime.now().isoformat()}, "clusters":{}}
    for cid in prog["clusters"].keys():
        if cid not in llm["clusters"]: continue
        p = prog["clusters"][cid]
        l = llm["clusters"][cid]
        p_scores = {x["entity_id"]:x["confidence"] for x in p.get("top_5_root_causes",[])}
        l_scores = {x["entity_id"]:x["confidence"] for x in l.get("top_5_root_causes",[])}
        all_e = set(p_scores.keys()).union(l_scores.keys())
        res = {e:(p_scores.get(e,0)*0.6 + l_scores.get(e,0)*0.4) for e in all_e}
        sorted_res = sorted(res.items(), key=lambda x:x[1], reverse=True)
        merged["clusters"][cid] = {"top_5_merged":sorted_res[:5]}
    return merged

def evaluate_rca_results(merged, gt_path=None):
    return {"metrics":{"top_1_accuracy":0.0, "top_3_accuracy":0.0}}

# ====================== Batch ======================
def run_programmatic_analysis(kg_dir, out_path):
    s = {"analysis_metadata":{},"clusters":{}}
    for r,d,files in os.walk(kg_dir):
        for f in files:
            if f.endswith("_kg.json"):
                path = os.path.join(r,f)
                a = ProgrammaticRCAAnalyzer(path)
                _, summ = a.generate_rca_report()
                s["clusters"][os.path.basename(r)] = summ
    with open(out_path,'w',encoding='utf-8') as f:
        json.dump(s,f,indent=2,ensure_ascii=False)
    return s

def run_llm_analysis(kg_dir, api_key, out_path):
    cfg = LLM_CONFIG.copy()
    s = {"analysis_metadata":{},"clusters":{}}
    for r,d,files in os.walk(kg_dir):
        for f in files:
            if f.endswith("_kg.json"):
                path = os.path.join(r,f)
                a = LLMbasedRCAAnalyzer(path, cfg)
                _, summ = a.generate_rca_report()
                a.save_report()
                s["clusters"][os.path.basename(r)] = summ
    with open(out_path,'w',encoding='utf-8') as f:
        json.dump(s,f,indent=2,ensure_ascii=False)
    return s

# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser(description="Market ToG RCA")
    parser.add_argument("--date_online", required=True)
    parser.add_argument("--output_suffix", required=True)
    parser.add_argument("--output_folder_name", default="1116")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--ground_truth", default=None)
    args = parser.parse_args()

    # Base paths
    base_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new6/{args.output_folder_name}"
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
                
    # 0. log redirection setup
    import sys
    # 1) Open the log file (use 'a' mode for appending to avoid overwriting existing logs; 'w' mode means overwriting)
    log_file = open(f"{base_dir}/Market_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_knowledge_graph.log", 'a', encoding='utf-8')

    # 2) Redirect both standard output (print content) and standard error (error messages)
    sys.stdout = log_file
    sys.stderr = log_file
    
    # 1. Run enhanced programmatic analysis
    print("\n=== Starting Enhanced Programmatic RCA Analysis (Market Microservice) ===")
    programmatic_summary = run_programmatic_analysis(kg_root_dir, programmatic_summary_path)
    
    # 2. Run enhanced LLM-driven analysis
    print("\n=== Starting Enhanced LLM-driven RCA Analysis (Market Microservice) ===")
    llm_summary = run_llm_analysis(kg_root_dir, None, llm_summary_path)
    
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
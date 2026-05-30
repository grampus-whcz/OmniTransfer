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
    """A local adaptation of Think-on-Graph for telecom RCA over the generated KG."""

    def __init__(self, kg_json_path, kg_data, llm_config=None):
        self.kg_json_path = kg_json_path
        self.kg_data = kg_data
        self.cluster_id = kg_data.get("cluster_id", "unknown")
        self.total_anomalies = kg_data.get("total_anomalies", 0)
        self.llm_config = llm_config or LLM_CONFIG
        self.cache_path = kg_json_path.replace("_kg.json", "_tog_cache.json")
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
            "Identify the most likely root cause entity in this telecom anomaly cluster using only the "
            "knowledge graph evidence and explored reasoning chains."
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
            rel_type = rel.get("type")
            source = rel.get("source")
            target = rel.get("target")
            props = rel.get("properties", {})
            if rel_type == "HAS_ANOMALY" and target in profiles:
                profile = profiles[target]
                profile["anomaly_count"] += 1
                profile["severity_score"] += ANOMALY_SEVERITY_WEIGHT.get(props.get("severity", "info"), 0.2)
                ts = props.get("timestamp", 0)
                profile["first_anomaly_ts"] = min(profile["first_anomaly_ts"], ts)
                profile["last_anomaly_ts"] = max(profile["last_anomaly_ts"], ts)
                metric = props.get("metric_name") or props.get("metric") or props.get("anomaly_type") or "anomaly"
                profile["key_evidence"].append(
                    f"{metric} severity={props.get('severity', 'info')} ts={ts}"
                )
            elif rel_type == "HAS_ATTRIBUTE" and source in profiles:
                fault_type = self.nodes_by_id.get(target, {}).get("properties", {}).get("fault_type")
                if fault_type:
                    profiles[source]["fault_types"].add(fault_type)
            elif rel_type == "TOPOLOGY_DEPENDS_ON":
                if source in profiles and target:
                    profiles[source]["neighbors"].add(target)
                if target in profiles and source:
                    profiles[target]["neighbors"].add(source)
            elif rel_type == "IMPACTS_BUSINESS" and source in profiles:
                profiles[source]["business_impact"] += 1

        for entity_id, profile in profiles.items():
            if profile["first_anomaly_ts"] == float("inf"):
                profile["first_anomaly_ts"] = None
            profile["fault_types"] = sorted(profile["fault_types"])
        return profiles

    def _entity_sort_key(self, entity_id):
        profile = self.entity_profiles[entity_id]
        first_ts = profile["first_anomaly_ts"] if profile["first_anomaly_ts"] is not None else float("inf")
        return (
            -profile["anomaly_count"],
            -profile["severity_score"],
            first_ts,
            -len(profile["neighbors"]),
            entity_id
        )

    def _format_timestamp(self, ts):
        if ts is None:
            return "N/A"
        return datetime.fromtimestamp(ts, BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")

    def _entity_snapshot(self, entity_id):
        profile = self.entity_profiles.get(entity_id)
        if not profile:
            node = self.nodes_by_id.get(entity_id, {})
            return f"{entity_id} ({node.get('label', 'UNKNOWN')})"
        fault_text = ", ".join(profile["fault_types"]) if profile["fault_types"] else "unknown"
        return (
            f"{entity_id} [{profile['label']}] anomalies={profile['anomaly_count']} "
            f"severity={profile['severity_score']:.2f} first_ts={self._format_timestamp(profile['first_anomaly_ts'])} "
            f"neighbors={len(profile['neighbors'])} business_impact={profile['business_impact']} "
            f"fault_types={fault_text}"
        )

    def _available_relation_keys(self, entity_id):
        relations = []
        for rel in self.out_edges.get(entity_id, []):
            relations.append(f"out::{rel.get('type', 'UNKNOWN')}")
        for rel in self.in_edges.get(entity_id, []):
            relations.append(f"in::{rel.get('type', 'UNKNOWN')}")
        return sorted(set(relations))

    def _expand_relation(self, entity_id, relation_key):
        direction, rel_type = relation_key.split("::", 1)
        rels = self.out_edges.get(entity_id, []) if direction == "out" else self.in_edges.get(entity_id, [])
        candidates = []
        for rel in rels:
            if rel.get("type") != rel_type:
                continue
            if direction == "out":
                source_id = entity_id
                target_id = rel.get("target")
                next_entity = target_id if target_id in self.entity_profiles else None
            else:
                source_id = rel.get("source")
                target_id = entity_id
                next_entity = source_id if source_id in self.entity_profiles else None
            source_name = self._entity_snapshot(source_id) if source_id else "UNKNOWN"
            target_name = self._entity_snapshot(target_id) if target_id else "UNKNOWN"
            triplet = f"({source_name}) -[{rel_type}]-> ({target_name})"
            evidence = rel.get("properties", {})
            candidates.append({
                "relation_key": relation_key,
                "relation_type": rel_type,
                "triplet": triplet,
                "source_id": source_id,
                "target_id": target_id,
                "next_entity": next_entity,
                "evidence": evidence
            })
        return candidates

    def _prompt_relation_prune(self, entity_id, relation_keys):
        relation_lines = "\n".join(
            f"{idx + 1}. {relation_key}" for idx, relation_key in enumerate(relation_keys)
        )
        return f"""TOG_RELATION_PRUNE
Question: {self.root_question}
Cluster: {self.cluster_id}
Current entity: {self._entity_snapshot(entity_id)}
Available relations:
{relation_lines}

Return the most useful relations for continuing root-cause reasoning in this exact format:
1. {{relation_key (Score: 0.60)}}: brief explanation
2. {{relation_key (Score: 0.25)}}: brief explanation
3. {{relation_key (Score: 0.15)}}: brief explanation
Only use relation keys from the list above."""

    def _parse_relation_scores(self, text, relation_keys):
        pattern = r"{\s*([^()]+?)\s+\(Score:\s*([0-9.]+)\)\s*}"
        parsed = []
        allowed = set(relation_keys)
        for relation_key, score_text in re.findall(pattern, text or ""):
            relation_key = relation_key.strip()
            if relation_key not in allowed:
                continue
            parsed.append((relation_key, float(score_text)))
        if not parsed:
            uniform = 1.0 / max(1, len(relation_keys))
            return [(relation_key, uniform) for relation_key in relation_keys[:3]]
        total = sum(score for _, score in parsed) or 1.0
        return [(relation_key, score / total) for relation_key, score in parsed[:3]]

    def _prompt_entity_score(self, relation_key, candidates):
        candidate_lines = "\n".join(
            f"{idx + 1}. {candidate['triplet']}" for idx, candidate in enumerate(candidates)
        )
        return f"""TOG_ENTITY_SCORE
Question: {self.root_question}
Cluster: {self.cluster_id}
Selected relation: {relation_key}
Candidate triplets:
{candidate_lines}

Score each candidate's usefulness for finding the root cause. Return ONLY one line:
Score: {', '.join(['0.0'] * len(candidates))}
The scores must sum to 1."""

    def _parse_entity_scores(self, text, expected_size):
        numbers = re.findall(r"\d+\.\d+|\d+", text or "")
        return _normalize_scores([float(number) for number in numbers], expected_size)

    def _prompt_sufficiency(self, chains):
        chain_text = "\n".join(f"- {chain}" for chain in chains[-12:])
        return f"""TOG_SUFFICIENCY_CHECK
Question: {self.root_question}
Cluster: {self.cluster_id}
Explored knowledge triplets:
{chain_text}

Is the explored evidence already sufficient to identify the most likely root cause? Reply with only {{Yes}} or {{No}}."""

    def _prompt_final_reasoning(self, explored_states, reasoning_chains):
        candidate_entities = []
        for state in explored_states:
            entity_id = state.get("entity_id")
            if entity_id and entity_id not in candidate_entities:
                candidate_entities.append(entity_id)
        if not candidate_entities:
            candidate_entities = sorted(self.entity_ids, key=self._entity_sort_key)[:TOG_SEARCH_CONFIG["width"]]
        entity_block = "\n".join(
            f"- {self._entity_snapshot(entity_id)}" for entity_id in candidate_entities[:8]
        )
        chain_block = "\n".join(f"- {chain}" for chain in reasoning_chains[:20])
        return f"""TOG_FINAL_REASONING
You are doing telecom root cause analysis with a Think-on-Graph style reasoning process.
Question: {self.root_question}
Cluster: {self.cluster_id}
Candidate entities:
{entity_block}

Explored reasoning chains:
{chain_block}

Return JSON only:
{{
  "root_cause_ranking": [
    {{
      "entity_id": "entity id from candidate entities",
      "confidence": 0.0,
      "reason": "short grounded reason",
      "evidence": ["fact 1", "fact 2"],
      "fault_types": ["fault type if available"],
      "first_anomaly_time": "YYYY-MM-DD HH:MM:SS"
    }}
  ],
  "fault_propagation": "short propagation summary",
  "summary": "one sentence overall RCA conclusion"
}}"""

    def _llm(self, prompt):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a telecom SRE expert following Think-on-Graph reasoning over a local "
                    "knowledge graph. Be grounded, concise, and do not invent entities."
                )
            },
            {"role": "user", "content": prompt}
        ]
        return call_llm_api(self.llm_config, messages, cluster_id=self.cluster_id)

    def _check_cache(self):
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if cache.get("model_used") != self.llm_config.get("model"):
                return None
            return cache
        except Exception:
            return None

    def _save_cache(self, payload):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _seed_entities(self):
        ordered = sorted(self.entity_ids, key=self._entity_sort_key)
        retained = ordered[:TOG_SEARCH_CONFIG["num_retain_entity"]]
        return [
            {
                "entity_id": entity_id,
                "score": 1.0,
                "path": [],
                "visited_entities": [entity_id]
            }
            for entity_id in retained
        ]

    def run(self):
        cached = self._check_cache()
        if cached:
            return cached

        explored_states = self._seed_entities()
        all_reasoning_chains = []
        exploration_trace = []
        llm_trace = []

        for depth in range(1, TOG_SEARCH_CONFIG["depth"] + 1):
            next_states = []
            for state in explored_states:
                entity_id = state["entity_id"]
                relation_keys = self._available_relation_keys(entity_id)[:TOG_SEARCH_CONFIG["max_candidate_relations"]]
                if not relation_keys:
                    continue
                relation_prompt = self._prompt_relation_prune(entity_id, relation_keys)
                relation_response = self._llm(relation_prompt)
                print(f"Sleep 30 seconds after relation prune LLM call for entity {entity_id} at depth {depth}")
                time.sleep(30)
                llm_trace.append({
                    "stage": "relation_prune",
                    "depth": depth,
                    "entity_id": entity_id,
                    "response": relation_response
                })
                selected_relations = self._parse_relation_scores(relation_response.get("content", ""), relation_keys)

                for relation_key, relation_weight in selected_relations:
                    candidates = self._expand_relation(entity_id, relation_key)
                    if not candidates:
                        continue
                    candidates = candidates[:TOG_SEARCH_CONFIG["max_candidate_entities"]]
                    entity_prompt = self._prompt_entity_score(relation_key, candidates)
                    entity_response = self._llm(entity_prompt)
                    print(f"Sleep 30 seconds after entity score LLM call for relation {relation_key} at depth {depth}")
                    time.sleep(30)
                    llm_trace.append({
                        "stage": "entity_score",
                        "depth": depth,
                        "entity_id": entity_id,
                        "relation_key": relation_key,
                        "response": entity_response
                    })
                    entity_scores = self._parse_entity_scores(entity_response.get("content", ""), len(candidates))

                    for idx, candidate in enumerate(candidates):
                        candidate_score = state["score"] * relation_weight * entity_scores[idx]
                        new_path = state["path"] + [candidate["triplet"]]
                        next_entity = candidate["next_entity"] or entity_id
                        if candidate["next_entity"] and candidate["next_entity"] in state["visited_entities"]:
                            continue
                        next_states.append({
                            "entity_id": next_entity,
                            "score": candidate_score,
                            "path": new_path,
                            "visited_entities": state["visited_entities"] + ([next_entity] if next_entity != entity_id else [])
                        })
                        all_reasoning_chains.extend(new_path[-1:])

            if not next_states:
                break

            unique_states = {}
            for state in sorted(next_states, key=lambda item: item["score"], reverse=True):
                key = (state["entity_id"], tuple(state["path"]))
                if key not in unique_states:
                    unique_states[key] = state
            explored_states = list(unique_states.values())[:TOG_SEARCH_CONFIG["width"]]
            exploration_trace.append({
                "depth": depth,
                "states": [
                    {
                        "entity_id": state["entity_id"],
                        "score": round(state["score"], 6),
                        "path_length": len(state["path"]),
                        "last_triplet": state["path"][-1] if state["path"] else ""
                    }
                    for state in explored_states
                ]
            })

            sufficiency_prompt = self._prompt_sufficiency(all_reasoning_chains)
            sufficiency_response = self._llm(sufficiency_prompt)
            llm_trace.append({
                "stage": "sufficiency_check",
                "depth": depth,
                "response": sufficiency_response
            })
            if "{yes}" in sufficiency_response.get("content", "").strip().lower() or "yes" == sufficiency_response.get("content", "").strip().lower():
                break

        final_prompt = self._prompt_final_reasoning(explored_states, all_reasoning_chains)
        final_response = self._llm(final_prompt)
        llm_trace.append({
            "stage": "final_reasoning",
            "response": final_response
        })
        final_json = _extract_json_object(final_response.get("content", ""))

        ranking = []
        for idx, item in enumerate((final_json or {}).get("root_cause_ranking", []), start=1):
            entity_id = item.get("entity_id")
            if entity_id not in self.entity_profiles:
                continue
            profile = self.entity_profiles[entity_id]
            ranking.append({
                "rank": idx,
                "entity_id": entity_id,
                "component_type": profile["component_type"],
                "confidence": round(float(item.get("confidence", 0.0)), 4),
                "reason": item.get("reason", ""),
                "evidence": item.get("evidence", []),
                "fault_types": item.get("fault_types") or profile["fault_types"],
                "first_anomaly_time": item.get("first_anomaly_time") or self._format_timestamp(profile["first_anomaly_ts"]),
                "anomaly_count": profile["anomaly_count"]
            })

        if not ranking:
            fallback_entities = sorted(self.entity_ids, key=self._entity_sort_key)[:TOG_SEARCH_CONFIG["width"]]
            for idx, entity_id in enumerate(fallback_entities, start=1):
                profile = self.entity_profiles[entity_id]
                ranking.append({
                    "rank": idx,
                    "entity_id": entity_id,
                    "component_type": profile["component_type"],
                    "confidence": round(max(0.1, 1.0 - 0.15 * (idx - 1)), 4),
                    "reason": "Fallback ToG ranking based on anomaly concentration and graph position.",
                    "evidence": profile["key_evidence"][:3],
                    "fault_types": profile["fault_types"],
                    "first_anomaly_time": self._format_timestamp(profile["first_anomaly_ts"]),
                    "anomaly_count": profile["anomaly_count"]
                })

        result = {
            "analysis_method": "tog",
            "cluster_id": self.cluster_id,
            "analysis_time": datetime.now().isoformat(),
            "model_used": self.llm_config["model"],
            "search_config": TOG_SEARCH_CONFIG,
            "total_anomalies": self.total_anomalies,
            "primary_root_cause": ranking[0],
            "top_5_root_causes": ranking[:5],
            "reasoning_chains": all_reasoning_chains[:20],
            "exploration_trace": exploration_trace,
            "fault_propagation": (final_json or {}).get("fault_propagation", ""),
            "summary": (final_json or {}).get("summary", ""),
            "llm_trace": llm_trace,
            "final_response": final_response
        }
        self._save_cache(result)
        return result

    def render_programmatic_report(self, result):
        report = [
            f"# ToG-Style Root Cause Analysis Report - Cluster {self.cluster_id}",
            f"**Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Analysis Method**: Think-on-Graph style telecom RCA",
            f"**Model Used**: {result['model_used']}",
            f"**Search Config**: depth={TOG_SEARCH_CONFIG['depth']}, width={TOG_SEARCH_CONFIG['width']}, retain={TOG_SEARCH_CONFIG['num_retain_entity']}",
            ""
        ]
        report.append("## Root Cause Ranking")
        for item in result["top_5_root_causes"]:
            report.append(
                f"- Rank {item['rank']}: {item['entity_id']} | confidence={item['confidence']:.4f} | "
                f"reason={item['reason']} | fault_types={', '.join(item['fault_types']) if item['fault_types'] else 'unknown'}"
            )
        report.append("")
        report.append("## Explored Reasoning Chains")
        for chain in result["reasoning_chains"]:
            report.append(f"- {chain}")
        if result.get("fault_propagation"):
            report.append("")
            report.append("## Fault Propagation")
            report.append(result["fault_propagation"])
        if result.get("summary"):
            report.append("")
            report.append("## Summary")
            report.append(result["summary"])
        return "\n".join(report)

    def render_llm_report(self, result):
        report = [
            f"# LLM-Driven ToG RCA Report - Cluster {self.cluster_id}",
            f"**Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Model Used**: {result['model_used']}",
            f"**Knowledge Graph Source**: {self.kg_json_path}",
            ""
        ]
        report.append("## Final Root Cause Ranking")
        for item in result["top_5_root_causes"]:
            evidence_text = "; ".join(item["evidence"]) if item["evidence"] else "N/A"
            report.append(
                f"- Rank {item['rank']}: {item['entity_id']} | confidence={item['confidence']:.4f} | "
                f"first_anomaly={item['first_anomaly_time']} | evidence={evidence_text}"
            )
        report.append("")
        report.append("## Final LLM Reasoning Output")
        report.append(result["final_response"].get("content", ""))
        return "\n".join(report)

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
        """Generate programmatic RCA with a ToG-style graph exploration core."""
        tog = TelecomToGAnalyzer(self.kg_json_path, self.kg_data, LLM_CONFIG)
        self.analysis_summary = tog.run()
        self.root_causes = self.analysis_summary.get("top_5_root_causes", [])
        report = tog.render_programmatic_report(self.analysis_summary)
        return report, self.analysis_summary

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

# ====================== Result Merging & Evaluation ======================
def merge_rca_results(programmatic_summary, llm_summary):
    """Merge programmatic and LLM RCA results (weighted fusion)"""
    merged_summary = {
        "analysis_metadata": {
            "merge_time": datetime.now().isoformat(),
            "programmatic_weight": 0.5,
            "llm_weight": 0.5,
            "analysis_method": "tog_fusion"
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
        
        # 1. Extract programmatic scores
        prog_root_causes = {}
        for rc in prog_result.get("top_5_root_causes", []):
            prog_root_causes[rc["entity_id"]] = rc.get("confidence", rc.get("total_score", 0.0))
        
        # 2. Extract LLM scores, preferring structured summaries from the ToG engine
        llm_root_causes = {}
        for rc in llm_result.get("top_5_root_causes", []):
            llm_root_causes[rc["entity_id"]] = rc.get("confidence", rc.get("total_score", 0.0))

        if not llm_root_causes:
            llm_content = llm_result.get("llm_response_content", "")
            final_json = _extract_json_object(llm_content)
            for rc in (final_json or {}).get("root_cause_ranking", []):
                entity_id = rc.get("entity_id")
                if entity_id:
                    llm_root_causes[entity_id] = float(rc.get("confidence", 0.0))
        
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
            "analysis_method": "tog_fusion",
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
            "analysis_method": "tog",
            "search_config": TOG_SEARCH_CONFIG,
            "model_used": LLM_CONFIG["model"]
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
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Path to ground truth JSON file for evaluation (optional)")
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
    programmatic_summary_path = f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_programmatic_rca_summary.json"
    llm_summary_path = f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_llm_rca_summary.json"
    merged_summary_path = f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_merged_rca_summary.json"
    evaluation_path = f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_rca_evaluation.json"
    
    # # 0. log redirection setup
    # import sys
    # # 1) Open the log file (use 'a' mode for appending to avoid overwriting existing logs; 'w' mode means overwriting)
    # log_file = open(f"{base_dir}/Telecom_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_knowledge_graph.log", 'a', encoding='utf-8')

    # # 2) Redirect both standard output (print content) and standard error (error messages)
    # sys.stdout = log_file
    # sys.stderr = log_file
    
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

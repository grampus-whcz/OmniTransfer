import json
import os
import numpy as np
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

# virtual environment: conda faiss-env

# ====================== 全局配置 ======================
# 根因分析权重配置（适配Bank微服务架构）
SCORE_WEIGHTS = {
    "anomaly_count": 0.4,      # 异常数量权重
    "time_priority": 0.2,      # 时间优先级权重（越早越高）
    "topology_impact": 0.25,    # 拓扑影响范围权重
    "component_weight": 0.15    # 组件层级权重（DB>缓存>业务>网关>容器>入口）
}

# Bank微服务组件基础权重（适配Bank的分层架构）
COMPONENT_BASE_WEIGHT = {
    "database": 1.0,    # Mysql - 最高优先级
    "cache": 0.95,      # Redis - 次高
    "business": 0.9,    # Tomcat - 业务层
    "governance": 0.85, # MG - 治理层
    "gateway": 0.8,     # IG - 网关层
    "container": 0.75,  # Docker - 容器层
    "entry_point": 0.7, # Apache - 入口层
    "service_test": 0.65, # 测试服务
    "unknown": 0.5      # 未知组件
}

# LLM配置（GLM-4.7）
LLM_CONFIG = {
    "model": "glm-4.7",
    "api_key": "e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa",
    "api_url": "https://open.bigmodel.cn/api/coding/paas/v4",
    "temperature": 0.7,
    "max_tokens": 8192
}

# Bank时区配置
BEIJING_TZ = timezone(timedelta(hours=8))

# ====================== 程序式RCA分析器 ======================
class ProgrammaticRCAAnalyzer:
    def __init__(self, kg_json_path):
        """初始化程序式根因分析器"""
        self.kg_json_path = kg_json_path
        self.kg_data = self._load_kg_data()
        self.cluster_id = self.kg_data["cluster_id"]
        self.total_anomalies = self.kg_data["total_anomalies"]
        
        # 核心分析结果
        self.entity_scores = {}  # 实体根因分数
        self.root_causes = []    # 排序后的根因结果
        self.analysis_summary = {}  # 最终汇总
    
    def _load_kg_data(self):
        """加载知识图谱JSON数据"""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"加载知识图谱失败: {e}")
    
    def _identify_service_layer(self, entity_name):
        """识别实体所属的Bank微服务层级（适配Bank架构）"""
        entity_lower = entity_name.lower()
        
        if 'mysql' in entity_lower:
            return "database"
        elif 'redis' in entity_lower:
            return "cache"
        elif 'tomcat' in entity_lower:
            return "business"
        elif 'mg' in entity_lower:
            return "governance"
        elif 'ig' in entity_lower:
            return "gateway"
        elif 'docker' in entity_lower:
            return "container"
        elif 'apache' in entity_lower:
            return "entry_point"
        elif 'servicetest' in entity_lower:
            return "service_test"
        else:
            return "unknown"
    
    def _extract_entity_features(self):
        """提取Bank实体核心特征"""
        # 1. 基础实体信息
        entities = {}
        for node in self.kg_data["nodes"]:
            if node["label"] in ["OS", "DOCKER", "DB", "OS_Sub", "DOCKER_Sub", "unknown"]:
                entity_id = node["id"]
                entity_name = node["properties"].get("entity_id", entity_id)
                is_main = node["properties"].get("is_main_entity", True)
                
                # 识别Bank微服务层级
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
        
        # 2. 统计异常数量、首次异常时间、故障类型
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                # 修正：HAS_ANOMALY的source是实体，target是属性
                entity_id = rel["source"]
                if entity_id in entities:
                    # 统计异常数
                    entities[entity_id]["anomaly_count"] += 1
                    # 记录首次异常时间
                    ts = rel["properties"]["timestamp"]
                    if ts < entities[entity_id]["first_anomaly_ts"]:
                        entities[entity_id]["first_anomaly_ts"] = ts
        
        # 3. 关联故障类型
        attr_fault_map = {}
        # 先建立属性ID到故障类型的映射
        for node in self.kg_data["nodes"]:
            if node["label"] == "AnomalyAttribute" and "fault_type" in node["properties"]:
                attr_fault_map[node["id"]] = node["properties"]["fault_type"]
        
        # 再关联到实体
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ATTRIBUTE":
                entity_id = rel["source"]
                attr_id = rel["target"]
                if entity_id in entities and attr_id in attr_fault_map:
                    entities[entity_id]["fault_types"].add(attr_fault_map[attr_id])
        
        # 4. 统计拓扑邻居（影响范围）
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
        """计算实体根因分数（适配Bank架构）"""
        # 1. 归一化参数
        max_anomaly_count = max([e["anomaly_count"] for e in entities.values()], default=1)
        valid_ts = [e["first_anomaly_ts"] for e in entities.values() if e["first_anomaly_ts"] != float('inf')]
        min_ts = min(valid_ts, default=0)
        max_ts = max(valid_ts, default=1)
        max_neighbors = max([len(e["topology_neighbors"]) for e in entities.values()], default=1)
        
        # 2. 计算各维度分数
        for entity_id, entity in entities.items():
            # 只分析主实体
            if not entity["is_main"]:
                continue
            
            # 2.1 异常数量分数 (0-1)
            count_score = entity["anomaly_count"] / max_anomaly_count if max_anomaly_count > 0 else 0
            
            # 2.2 时间优先级分数（越早越高，0-1）
            if entity["first_anomaly_ts"] == float('inf'):
                time_score = 0
            elif max_ts == min_ts:
                time_score = 1.0
            else:
                time_score = (max_ts - entity["first_anomaly_ts"]) / (max_ts - min_ts)
            
            # 2.3 拓扑影响分数（邻居越多越高，0-1）
            topology_score = len(entity["topology_neighbors"]) / max_neighbors if max_neighbors > 0 else 0
            
            # 2.4 组件权重分数（0-1）
            component_score = entity["component_weight"] / max(COMPONENT_BASE_WEIGHT.values())
            
            # 3. 加权总分
            total_score = (
                count_score * SCORE_WEIGHTS["anomaly_count"] +
                time_score * SCORE_WEIGHTS["time_priority"] +
                topology_score * SCORE_WEIGHTS["topology_impact"] +
                component_score * SCORE_WEIGHTS["component_weight"]
            )
            
            # 转换时间戳为北京时区
            if entity["first_anomaly_ts"] != float('inf'):
                first_time = datetime.fromtimestamp(entity["first_anomaly_ts"], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
            else:
                first_time = "N/A"
            
            self.entity_scores[entity_id] = {
                "entity_id": entity_id,
                "entity_name": entity["entity_name"],
                "component_type": entity["component_type"],
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
        """基于规则过滤根因（适配Bank微服务）"""
        # 1. 按总分排序
        sorted_entities = sorted(
            self.entity_scores.values(),
            key=lambda x: x["total_score"],
            reverse=True
        )
        
        # 2. 规则过滤
        threshold = 0.1  # 最低分数阈值
        for entity in sorted_entities:
            if entity["total_score"] < threshold:
                continue
            # 排除无异常的实体
            if entity["anomaly_count"] == 0:
                continue
            
            self.root_causes.append(entity)
        
        # 3. 兜底（至少返回1个根因）
        if not self.root_causes and sorted_entities:
            self.root_causes.append(sorted_entities[0])
    
    def generate_rca_report(self):
        """生成Bank专用程序式根因分析报告"""
        # 1. 提取特征并计算分数
        entities = self._extract_entity_features()
        self._calculate_entity_scores(entities)
        self._filter_root_causes()
        
        # 2. 构建报告
        report = []
        report.append(f"# 程序式根因分析报告 - 集群 {self.cluster_id} (Bank微服务)")
        report.append(f"**分析时间**: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S CST')}")
        report.append(f"**总异常数**: {self.total_anomalies}")
        report.append(f"**RCA维度权重**: 异常数量({SCORE_WEIGHTS['anomaly_count']*100}%) + "
                      f"时间优先级({SCORE_WEIGHTS['time_priority']*100}%) + "
                      f"拓扑影响({SCORE_WEIGHTS['topology_impact']*100}%) + "
                      f"组件权重({SCORE_WEIGHTS['component_weight']*100}%)")
        report.append(f"**Bank组件权重**: Database(1.0) > Cache(0.95) > Business(0.9) > Governance(0.85) > Gateway(0.8) > Container(0.75) > EntryPoint(0.7)")
        report.append("")
        
        # 3. 根因结果
        report.append("## 故障根因排名")
        for idx, cause in enumerate(self.root_causes[:5]):  # 只显示前5
            report.append(f"### 根因 #{idx+1}")
            report.append(f"- **实体ID**: {cause['entity_id']}")
            report.append(f"- **实体名称**: {cause['entity_name']}")
            report.append(f"- **组件类型**: {cause['component_type'].upper()}")
            report.append(f"- **根因置信度**: {cause['total_score']:.4f}")
            anomaly_pct = (cause['anomaly_count']/self.total_anomalies*100) if self.total_anomalies >0 else 0
            report.append(f"- **异常数量**: {cause['anomaly_count']} ({anomaly_pct:.1f}%)")
            report.append(f"- **首次异常时间**: {cause['first_anomaly_time']}")
            report.append(f"- **故障类型**: {', '.join(cause['fault_types']) if cause['fault_types'] else 'unknown'}")
            report.append(f"- **拓扑影响范围**: {cause['neighbor_count']} 个关联实体")
            report.append(f"- **维度分数拆解**:")
            report.append(f"  - 异常数量分数: {cause['count_score']:.4f}")
            report.append(f"  - 时间优先级分数: {cause['time_score']:.4f}")
            report.append(f"  - 拓扑影响分数: {cause['topology_score']:.4f}")
            report.append(f"  - 组件权重分数: {cause['component_score']:.4f}")
            report.append("")
        
        # 4. 故障类型分析
        all_fault_types = []
        for cause in self.root_causes:
            all_fault_types.extend(cause['fault_types'])
        fault_counter = Counter(all_fault_types)
        if fault_counter:
            report.append("## 故障类型分布")
            for fault_type, count in fault_counter.most_common():
                if fault_type != 'unknown':
                    pct = (count/len(self.root_causes)*100) if self.root_causes else 0
                    report.append(f"- **{fault_type}**: {count} 个实体涉及 ({pct:.1f}%)")
            report.append("")
        
        # 5. 整改建议（适配Bank架构）
        report.append("## 整改建议")
        if self.root_causes:
            primary_cause = self.root_causes[0]
            report.append(f"1. 优先排查 {primary_cause['entity_name']} (置信度: {primary_cause['total_score']:.4f})")
            report.append(f"2. 重点关注 {primary_cause['component_type'].upper()} 层的 {', '.join(primary_cause['fault_types'])} 问题")
            report.append(f"3. 核查 {primary_cause['entity_name']} 的 {primary_cause['neighbor_count']} 个关联实体是否存在级联故障")
            # Bank特定建议
            if primary_cause['component_type'] == 'database':
                report.append(f"4. 检查Mysql数据库连接池、查询性能和数据一致性")
            elif primary_cause['component_type'] == 'cache':
                report.append(f"4. 检查Redis缓存命中率、内存使用和持久化配置")
            elif primary_cause['component_type'] == 'business':
                report.append(f"4. 检查Tomcat应用服务的线程池、JVM内存和业务逻辑异常")
            elif primary_cause['component_type'] == 'gateway':
                report.append(f"4. 检查IG网关的路由配置、限流策略和请求转发性能")
            report.append(f"5. 持续监控 {primary_cause['entity_name']} 的异常频率和恢复状态")
        else:
            report.append("1. 未识别到明确根因，建议全面检查Bank微服务调用链")
        
        # 6. 生成分析汇总
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
    """批量运行程序式根因分析并生成汇总"""
    programmatic_summary = {
        "analysis_metadata": {
            "analysis_time": datetime.now(BEIJING_TZ).isoformat(),
            "score_weights": SCORE_WEIGHTS,
            "component_base_weights": COMPONENT_BASE_WEIGHT,
            "analysis_type": "Bank_microservice_programmatic_rca"
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
                    
                    # 提取集群名称
                    cluster_name = os.path.basename(os.path.dirname(kg_path))
                    # 保存单独的报告
                    report_path = kg_path.replace("_kg.json", "_programmatic_rca_report.md")
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"✅ 程序式RCA报告已生成: {report_path}")
                    
                    # 添加到汇总
                    programmatic_summary["clusters"][cluster_name] = cluster_summary
                    
                except Exception as e:
                    print(f"❌ 分析 {kg_path} 失败: {e}")
    
    # 保存汇总JSON
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(programmatic_summary, f, ensure_ascii=False, indent=2)
    print(f"✅ 程序式RCA汇总已保存到: {summary_output_path}")
    
    return programmatic_summary

# ====================== LLM驱动的RCA分析器 ======================
class LLMbasedRCAAnalyzer:
    def __init__(self, kg_json_path, llm_config=None):
        """初始化LLM驱动的根因分析器（Bank专用）"""
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
        """加载知识图谱数据"""
        try:
            with open(self.kg_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"加载知识图谱失败: {e}")
    
    def _identify_service_layer(self, entity_name):
        """识别Bank微服务层级"""
        entity_lower = entity_name.lower()
        if 'mysql' in entity_lower:
            return "Database (Mysql)"
        elif 'redis' in entity_lower:
            return "Cache (Redis)"
        elif 'tomcat' in entity_lower:
            return "Business Layer (Tomcat)"
        elif 'mg' in entity_lower:
            return "Governance Layer (MG)"
        elif 'ig' in entity_lower:
            return "Gateway Layer (IG)"
        elif 'docker' in entity_lower:
            return "Container Layer (Docker)"
        elif 'apache' in entity_lower:
            return "Entry Point (Apache)"
        else:
            return "Unknown Layer"
    
    def _convert_kg_to_prompt(self):
        """将Bank知识图谱转换为LLM提示词"""
        # 1. 基础信息
        time_span = self.kg_data["time_span"]
        start_time = datetime.fromtimestamp(time_span['start'], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
        end_time = datetime.fromtimestamp(time_span['end'], BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S CST")
        
        prompt = []
        prompt.append("### 分析背景")
        prompt.append("你是资深的Bank微服务架构故障根因分析专家，熟悉apache→IG→Tomcat→MG→docker→mysql/redis的调用链架构。")
        prompt.append("请基于以下Bank微服务异常知识图谱数据，分析故障根因。")
        prompt.append(f"分析目标: 异常集群 {self.cluster_id}")
        prompt.append(f"时间范围: {start_time} 至 {end_time} (持续时间: {time_span['duration_sec']} 秒)")
        prompt.append(f"总异常数: {self.total_anomalies}")
        prompt.append("Bank微服务层级优先级: Database(Mysql) > Cache(Redis) > Business(Tomcat) > Governance(MG) > Gateway(IG) > Container(Docker) > EntryPoint(Apache)")
        prompt.append("")
        
        # 2. 实体信息
        prompt.append("### 实体信息")
        entities = {}
        for node in self.kg_data["nodes"]:
            if node["label"] in ["OS", "DOCKER", "DB", "OS_Sub", "DOCKER_Sub", "unknown"]:
                entity_id = node["id"]
                entity_name = node["properties"].get("entity_id", entity_id)
                layer = self._identify_service_layer(entity_name)
                entities[entity_id] = {
                    "id": entity_id,
                    "name": entity_name,
                    "layer": layer,
                    "is_main": node["properties"].get("is_main_entity", True)
                }
        
        prompt.append(f"涉及实体总数: {len(entities)}")
        prompt.append("实体列表（含Bank微服务层级）:")
        for entity_id, info in entities.items():
            prompt.append(f"- {entity_id} (名称: {info['name']}, 层级: {info['layer']}, 主实体: {info['is_main']})")
        prompt.append("")
        
        # 3. 异常分布
        prompt.append("### 异常分布")
        entity_anomaly_count = defaultdict(int)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "HAS_ANOMALY":
                entity_anomaly_count[rel["source"]] += 1
        
        prompt.append("各实体异常数量:")
        sorted_counts = sorted(entity_anomaly_count.items(), key=lambda x: x[1], reverse=True)
        for entity_id, count in sorted_counts:
            if count > 0 and entity_id in entities:
                prompt.append(f"- {entities[entity_id]['name']} ({entity_id}): {count} 次异常")
        prompt.append("")
        
        # 4. 故障类型
        prompt.append("### 故障类型")
        fault_types = defaultdict(int)
        attr_fault_map = {}
        
        # 构建属性-故障类型映射
        for node in self.kg_data["nodes"]:
            if node["label"] == "AnomalyAttribute" and "fault_type" in node["properties"]:
                attr_fault_map[node["id"]] = node["properties"]["fault_type"]
        
        # 统计故障类型
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "MAPS_TO_FAULT" and rel["source"] in attr_fault_map:
                fault_type = attr_fault_map[rel["source"]]
                fault_types[fault_type] += 1
        
        prompt.append("故障类型分布:")
        for fault_type, count in fault_types.items():
            prompt.append(f"- {fault_type}: 涉及 {count} 个异常属性")
        prompt.append("")
        
        # 5. 拓扑关系
        prompt.append("### 拓扑依赖关系")
        topology_rels = defaultdict(list)
        for rel in self.kg_data["relationships"]:
            if rel["type"] == "TOPOLOGY_DEPENDS_ON":
                src_id = rel["source"]
                dst_id = rel["target"]
                src_name = entities.get(src_id, {}).get("name", src_id)
                dst_name = entities.get(dst_id, {}).get("name", dst_id)
                topology_rels[src_name].append(dst_name)
        
        prompt.append("实体拓扑依赖（Bank调用链）:")
        for src, dsts in topology_rels.items():
            prompt.append(f"- {src} 依赖于: {', '.join(dsts)}")
        prompt.append("")
        
        # 6. 分析要求（Bank专用）
        prompt.append("### 分析要求")
        prompt.append("1. 识别最可能的故障根因实体（按置信度排名，至少3个），结合Bank微服务层级优先级解释推理过程；")
        prompt.append("2. 分析故障传播路径（时间维度+Bank调用链拓扑维度）；")
        prompt.append("3. 识别主要故障类型和影响范围；")
        prompt.append("4. 提供针对Bank微服务架构的具体、可操作的整改建议和故障排查步骤；")
        prompt.append("5. 重点关注数据库(Mysql)和缓存(Redis)层的异常，这是Bank核心数据层；")
        prompt.append("6. 输出格式: 使用Markdown格式，分节清晰，逻辑严谨，有充分的支撑依据。")
        
        return "\n".join(prompt)
    
    def _call_llm_api(self, prompt):
        """调用GLM-4.7 API（适配Bank分析）"""
        from zhipuai import ZhipuAI
        
        # 初始化GLM客户端
        client = ZhipuAI(
            api_key=self.llm_config['api_key'],
            base_url=self.llm_config.get('api_base', 'https://open.bigmodel.cn/api/coding/paas/v4')
        )
        
        # 构建消息
        messages = [
            {"role": "system", "content": "你是Bank微服务故障根因分析专家，精通apache→IG→Tomcat→MG→docker→mysql/redis架构的故障分析。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # 获取配置参数
            temperature = self.llm_config.get('temperature', 0.7)
            max_output_tokens = self.llm_config.get('max_tokens', 8192)
            
            # 调用GLM API
            full_response = client.chat.completions.create(
                model=self.llm_config['model'],
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
                top_p=0.95
            )
            
            # 提取响应内容
            response_content = full_response.choices[0].message.content
            self.llm_raw_content = response_content
            
            # 转换为可序列化的字典
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
            
            # 打印token使用情况
            prompt_tokens = self.llm_response_dict['usage']['prompt_tokens']
            completion_tokens = self.llm_response_dict['usage']['completion_tokens']
            total_tokens = self.llm_response_dict['usage']['total_tokens']
            print(f"=={self.llm_config['model']}== 输入token: {prompt_tokens}, 输出token: {completion_tokens}, 总计: {total_tokens}")
            
            # Token超限警告
            if total_tokens > 120000:
                print(f"警告: Token使用量({total_tokens})接近128K上限")
            
            return response_content
            
        except Exception as e:
            raise RuntimeError(f"GLM API调用失败: {e}")
    
    def generate_rca_report(self):
        """生成LLM驱动的Bank根因分析报告"""
        # 1. 生成提示词
        prompt = self._convert_kg_to_prompt()
        
        # 2. 调用LLM
        llm_output = self._call_llm_api(prompt)
        
        # 3. 构建最终报告
        report = []
        report.append(f"# LLM驱动根因分析报告 - 集群 {self.cluster_id} (Bank微服务)")
        report.append(f"**分析时间**: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S CST')}")
        report.append(f"**使用模型**: {self.llm_config['model']}")
        report.append(f"**知识图谱源**: {self.kg_json_path}")
        report.append(f"**Bank架构**: apache → IG → Tomcat → MG → docker → Mysql/Redis")
        report.append("="*80)
        report.append("")
        report.append(llm_output)
        
        self.rca_report = "\n".join(report)
        
        # 4. 生成分析汇总
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
        """保存分析报告"""
        if not self.rca_report:
            raise ValueError("请先生成分析报告")
        
        # 保存主报告
        report_path = self.kg_json_path.replace("_kg.json", "_llm_rca_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.rca_report)
        
        # 保存LLM响应
        if self.llm_response_dict:
            response_path = self.kg_json_path.replace("_kg.json", "_llm_response.json")
            try:
                with open(response_path, 'w', encoding='utf-8') as f:
                    json.dump(self.llm_response_dict, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ 保存LLM响应JSON失败: {e}")
                # 降级保存原始内容
                fallback_path = self.kg_json_path.replace("_kg.json", "_llm_response_raw.txt")
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    f.write(self.llm_raw_content or "无响应内容")
        
        print(f"✅ LLM驱动RCA报告已生成: {report_path}")
        return report_path

def run_llm_analysis(kg_dir, api_key, summary_output_path):
    """批量运行LLM驱动的根因分析"""
    llm_config = LLM_CONFIG.copy()
    llm_config["api_key"] = api_key
    
    llm_summary = {
        "analysis_metadata": {
            "analysis_time": datetime.now(BEIJING_TZ).isoformat(),
            "model_used": llm_config['model'],
            "temperature": llm_config['temperature'],
            "max_tokens": llm_config['max_tokens'],
            "analysis_type": "Bank_microservice_llm_rca"
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
                    
                    # 提取集群名称
                    cluster_name = os.path.basename(os.path.dirname(kg_path))
                    # 添加到汇总
                    llm_summary["clusters"][cluster_name] = cluster_summary
                    
                except Exception as e:
                    print(f"❌ 分析 {kg_path} 失败: {e}")
    
    # 保存汇总JSON
    with open(summary_output_path, 'w', encoding='utf-8') as f:
        json.dump(llm_summary, f, ensure_ascii=False, indent=2)
    print(f"✅ LLM驱动RCA汇总已保存到: {summary_output_path}")
    
    return llm_summary

# ====================== 主程序 ======================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Bank数据集根因分析程序（程序式+LLM驱动）")
    parser.add_argument("--date_online", required=True, help="日期字符串，如 2021_03_04")
    parser.add_argument("--output_suffix", required=True, help="时间窗口，如 0230_0300")
    parser.add_argument("--output_folder_name", type=str, default="1116",
                        help="输出文件夹名称（如实验ID）")
    args = parser.parse_args()
    
    # 基础路径
    base_dir = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"
    kg_root_dir = f"{base_dir}/knowledge_graphs/{args.date_online}_{args.output_suffix}"
    
    # 汇总输出路径
    programmatic_summary_path = f"{base_dir}/Bank_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_programmatic_rca_summary.json"
    llm_summary_path = f"{base_dir}/Bank_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}_llm_rca_summary.json"
    
    # 验证输入目录
    if not os.path.exists(kg_root_dir):
        print(f"❌ 错误: 知识图谱目录不存在 - {kg_root_dir}")
        return
    
    # 1. 运行程序式分析
    print("\n=== 开始程序式RCA分析 (Bank微服务) ===")
    run_programmatic_analysis(kg_root_dir, programmatic_summary_path)
    
    # 2. 运行LLM驱动分析
    print("\n=== 开始LLM驱动RCA分析 (Bank微服务) ===")
    api_key = "e2bb1c9dcfea446896cdfb3735c98a10.ZwHWlBTzph3t6RIa"  # 可改为CLI参数
    run_llm_analysis(kg_root_dir, api_key, llm_summary_path)
    
    print("\n✅ 所有分析已完成！")
    print(f"📊 程序式分析汇总: {programmatic_summary_path}")
    print(f"📊 LLM驱动分析汇总: {llm_summary_path}")

if __name__ == "__main__":
    main()
import os
import json
import re
import hashlib

def generate_robust_mermaid_kg(json_data: dict) -> str:
    """
    高度鲁棒的运维知识图谱 Mermaid 生成器
    彻底规避跨子图干扰，采用扁平化渲染 + 强样式隔离策略
    """
    lines = ["graph LR"]
    
    # 1. 声明标准化的微服务多模态样式
    styles = [
        "    classDef style_time fill:#eee,stroke:#999,stroke-width:1px;",
        "    classDef style_infra fill:#f9f,stroke:#333,stroke-width:2px;",
        "    classDef style_docker fill:#bbf,stroke:#333,stroke-width:1px;",
        "    classDef style_subEntity fill:#ddf,stroke:#333,stroke-dasharray: 5 5;",
        "    classDef style_attr fill:#fee,stroke:#f66,stroke-width:1px;",
        "    classDef style_fault fill:#f33,stroke:#333,stroke-width:2px,color:#fff;",
        "    classDef style_default fill:#fff,stroke:#777,stroke-width:1px;"
    ]
    lines.extend(styles)
    lines.append("")

    # 2. 激进型 ID 清洗：将原始 ID 进行不重复的格式化转换，避免特殊字符导致解析崩溃
    id_mapping = {}
    def get_safe_id(raw_id: str) -> str:
        if raw_id not in id_mapping:
            # 过滤掉非英文字符，通过哈希或下划线保护
            clean_prefix = re.sub(r'[^a-zA-Z0-9]', '_', raw_id)[:15]
            hasher = hashlib.md5(raw_id.encode('utf-8')).hexdigest()[:8]
            id_mapping[raw_id] = f"node_{clean_prefix}_{hasher}"
        return id_mapping[raw_id]

    nodes = json_data.get("nodes", [])
    relationships = json_data.get("relationships", [])

    # 3. 提取并渲染所有节点（扁平渲染，靠样式区分层级）
    lines.append("    %% ==================== 节点定义 ====================")
    for node in nodes:
        n_id = node.get("id")
        label = node.get("label", "UNKNOWN")
        props = node.get("properties", {})
        
        safe_id = get_safe_id(n_id)
        
        # 提取展示文本
        if label == "Time":
            display_name = f"🕒 {props.get('time_str', n_id)}"
            style_class = "style_time"
        elif label == "AnomalyAttribute":
            display_name = f"📊 {props.get('attribute', n_id)}"
            style_class = "style_attr"
        elif label == "FaultType":
            display_name = f"🚨 Fault: {props.get('fault_type', n_id)}"
            style_class = "style_fault"
        elif label in ["DB", "OS"]:
            display_name = f"💾 {label}::{n_id}"
            style_class = "style_infra"
        elif label == "DOCKER":
            display_name = f"📦 Docker::{n_id}"
            style_class = "style_docker"
        elif label == "DOCKER_Sub":
            # 简化调用链过长的方法名展示
            short_id = n_id.split(",")[-1] if "," in n_id else n_id
            display_name = f"⚙️ {short_id}"
            style_class = "style_subEntity"
        else:
            display_name = n_id
            style_class = "style_default"

        # 针对不同实体定制图形形状
        if label == "DB":
            lines.append(f'    {safe_id}[("{display_name}")]:::{style_class}')
        elif label == "FaultType":
            lines.append(f'    {safe_id}((["{display_name}"])):::{style_class}')
        else:
            lines.append(f'    {safe_id}["{display_name}"]:::{style_class}')
            
    lines.append("")

    # 4. 建立关系连接（处理拓扑网络与双向依赖）
    lines.append("    %% ==================== 边依赖关系 ====================")
    seen_topo_edges = set()

    for rel in relationships:
        src_raw = rel.get("source")
        tgt_raw = rel.get("target")
        rel_type = rel.get("type", "DEPENDS_ON")
        props = rel.get("properties", {})
        
        if not src_raw or not tgt_raw:
            continue
            
        src_safe = get_safe_id(src_raw)
        tgt_safe = get_safe_id(tgt_raw)

        # 针对不同 AIOps 关系类型定制连线样式
        if rel_type == "TOPOLOGY_DEPENDS_ON":
            edge_key = tuple(sorted([src_safe, tgt_safe]))
            if edge_key in seen_topo_edges:
                continue
            seen_topo_edges.add(edge_key)
            lines.append(f"    {src_safe} <-->|拓扑依赖| {tgt_safe}")
            
        elif rel_type == "HAS_ATTRIBUTE":
            lines.append(f"    {src_safe} -->|拥有属性| {tgt_safe}")
            
        elif rel_type == "MAPS_TO_FAULT":
            conf = props.get("confidence", 0.5)
            lines.append(f"    {src_safe} ==>|根因映射 conf:{conf}| {tgt_safe}")
            
        elif rel_type == "HAS_ANOMALY":
            # 虚线代表时间线切片上的异常快照投射
            lines.append(f"    {src_safe} -.->|捕获异常| {tgt_safe}")
            
        else:
            lines.append(f"    {src_safe} -->|{rel_type}| {tgt_safe}")

    return "\n".join(lines)


def process_kg_file(file_path: str):
    """读取指定的 JSON 文件，解析生成修复版 Mermaid 代码并自动保存"""
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到指定的 JSON 文件 \n路径: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ 错误: JSON 格式解析失败。\n{e}")
            return

    # 调用重构后的鲁棒生成器
    mermaid_code = generate_robust_mermaid_kg(json_data)

    cluster_id = json_data.get("cluster_id", "Unknown")
    total_anomalies = json_data.get("total_anomalies", "Unknown")
    
    md_content = f"""# ADS-KGRCA 故障分析图谱 (Cluster {cluster_id}), 数据源: {file_path}, 包含异常事件数: {total_anomalies}, mermaid: {mermaid_code}"""

    output_md_path = os.path.splitext(file_path)[0] + "_fixed_visualization.md"
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"✅ 新版图谱已成功渲染并保存至:\n👉 {output_md_path}\n")

if __name__ == "__main__":
    target_path = "/root/shared-nvme/work/agent/ADS-KGRCA/experiments/report_result/Telecom_adskg_tog_llm_gemini-2.5-pro/knowledge_graphs/2020_05_28_0330_0400/cluster_1/cluster_1_kg.json"
    process_kg_file(target_path)

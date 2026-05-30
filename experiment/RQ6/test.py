import networkx as nx
import matplotlib.pyplot as plt
import os
import json

# ================== 1. 从指定文件路径读取图数据 ==================
# 你提供的文件绝对路径
json_file_path = "/root/shared-nvme/work/agent/ADS-KGRCA/experiments/report_result/Telecom_adskg_tog_llm_gemini-2.5-pro/knowledge_graphs/2020_05_26_0400_0430/cluster_5/cluster_5_kg.json"

# 读取本地 JSON 文件
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ================== 2. 构建 NetworkX 有向图 ==================
G = nx.DiGraph()

# 定义节点颜色映射
color_map = {
    "DOCKER": "skyblue",
    "DOCKER_Sub": "lightcyan",
    "DB": "wheat",
    "AnomalyAttribute": "lightcoral",
    "FaultType": "lightgreen",
    "Time": "lightgray"
}

# 添加节点
for node in data["nodes"]:
    node_id = node["id"]
    label = node["label"]
    node_type = label
    color = color_map.get(node_type, "white")
    
    G.add_node(node_id, label=label, type=node_type, color=color)

# 添加边
for rel in data["relationships"]:
    G.add_edge(rel["source"], rel["target"], type=rel["type"])

# ================== 3. 绘图设置 ==================
plt.figure(figsize=(16, 10))

# 布局：shell_layout 分层展示
pos = nx.shell_layout(G)

# 绘制节点
node_colors = [G.nodes[n]['color'] for n in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5000, alpha=0.9, edgecolors='black', linewidths=1.5)

# ================== 所有边类型单独绘制（区分颜色/线型） ==================
# 1. TOPOLOGY_DEPENDS_ON (蓝色点划线)
edges_topo = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'TOPOLOGY_DEPENDS_ON']
nx.draw_networkx_edges(G, pos, edgelist=edges_topo, edge_color='blue', style='-.', width=2.5, arrowsize=25)

# 2. HAS_ANOMALY (灰色实线)
edges_anomaly = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'HAS_ANOMALY']
nx.draw_networkx_edges(G, pos, edgelist=edges_anomaly, edge_color='gray', style='-', width=2.5, arrowsize=25)

# 3. MAPS_TO_FAULT (红色虚线)
edges_fault = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'MAPS_TO_FAULT']
nx.draw_networkx_edges(G, pos, edgelist=edges_fault, edge_color='red', style='--', width=2.5, arrowsize=25)

# 4. HAS_ATTRIBUTE (橙色粗实线) —— 新增独立样式
edges_attr = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'HAS_ATTRIBUTE']
nx.draw_networkx_edges(G, pos, edgelist=edges_attr, edge_color='darkorange', style='-', width=2.0, arrowsize=22)

# 5. BELONGS_TO (紫色点虚线) —— 新增独立样式
edges_belong = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'BELONGS_TO']
nx.draw_networkx_edges(G, pos, edgelist=edges_belong, edge_color='purple', style=':', width=2.0, arrowsize=22)

# 绘制标签（长标签自动换行）
labels = {}
for node in G.nodes:
    raw_label = G.nodes[node]['label']
    if len(raw_label) > 15:
        labels[node] = raw_label.replace(',', ',\n')
    else:
        labels[node] = raw_label

nx.draw_networkx_labels(G, pos, labels, font_size=11, font_weight='bold', font_family='sans-serif')

# 添加完整图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='DOCKER', markerfacecolor='skyblue', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='DOCKER_Sub', markerfacecolor='lightcyan', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='DB', markerfacecolor='wheat', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='AnomalyAttribute', markerfacecolor='lightcoral', markersize=12),
    Line2D([0], [0], marker='o', color='w', label='FaultType', markerfacecolor='lightgreen', markersize=12),
    Line2D([0], [0], color='blue', linestyle='-.', lw=2, label='TOPOLOGY_DEPENDS_ON'),
    Line2D([0], [0], color='gray', linestyle='-', lw=2, label='HAS_ANOMALY'),
    Line2D([0], [0], color='red', linestyle='--', lw=2, label='MAPS_TO_FAULT'),
    Line2D([0], [0], color='darkorange', linestyle='-', lw=2, label='HAS_ATTRIBUTE'),
    Line2D([0], [0], color='purple', linestyle=':', lw=2, label='BELONGS_TO')
]
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True, shadow=True)

plt.title(f"Cluster {data['cluster_id']} Fault Knowledge Graph", fontsize=18, pad=20)
plt.axis('off')

# ================== 4. 保存图片 ==================
output_file = "cluster_5_kg_schematic.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 示意图已成功保存为: {os.path.abspath(output_file)}")

# 如需直接显示图片，取消注释
# plt.show()
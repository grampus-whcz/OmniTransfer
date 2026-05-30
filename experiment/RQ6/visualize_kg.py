import json
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# ================== 1. 配置与路径 ==================
# 根据你提供的路径设置
BASE_DIR = "/root/shared-nvme/work/timeSeries/OmniTransfer_new8/1204/knowledge_graphs/2021_03_04_1900_1930/cluster_1"
JSON_FILE = os.path.join(BASE_DIR, "cluster_1_kg.json")

# 输出图片的保存路径
OUTPUT_DIR = os.path.join(BASE_DIR, "visualization_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_kg_data(filepath):
    """
    读取并解析 JSON 文件
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功加载数据: {filepath}")
        print(f"   - 节点数: {len(data['nodes'])}")
        print(f"   - 边数: {len(data['relationships'])}")
        return data
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None

def build_graph(data):
    """
    构建 NetworkX 图结构，并根据语义进行分类
    """
    G = nx.DiGraph() # 有向图

    # --- 节点分类容器 ---
    entities = {}      # 核心实体 (Docker, DB)
    attributes = {}    # 异常指标
    faults = {}        # 故障类型
    topology_nodes = {} # 拓扑节点

    # 1. 添加节点
    for node in data['nodes']:
        node_id = node['id']
        label = node['label']
        props = node['properties']
        
        # 提取显示标签
        if 'entity_id' in props:
            node_label = props['entity_id']
            entities[node_id] = node_label
        elif 'attribute_name' in props:
            # 简化长属性名
            attr_name = props['attribute_name']
            node_label = attr_name.split('_')[-1][:15] + "..." # 取后缀，截断
            attributes[node_id] = node_label
        elif 'fault_type' in props:
            node_label = props['fault_type']
            faults[node_id] = node_label
        elif 'entity_type' in props and props['entity_type'] == 'os_sub':
            node_label = "OS_Sub"
            topology_nodes[node_id] = node_label
        else:
            node_label = label
            
        # 节点颜色映射
        if label == "DOCKER" or label == "DB":
            color = 'lightblue'
        elif label == "AnomalyAttribute":
            color = 'lightcoral'
        elif label == "FaultType":
            color = 'lightgreen'
        elif label == "OS_SUB":
            color = 'wheat'
        else:
            color = 'silver'
            
        G.add_node(node_id, label=node_label, color=color, type=label)

    # 2. 添加边
    for rel in data['relationships']:
        source = rel['source']
        target = rel['target']
        rel_type = rel['type']
        
        # 边颜色和样式映射
        if rel_type == "HAS_ANOMALY":
            edge_color = 'gray'
            edge_style = '-'
            edge_width = 0.5
        elif rel_type == "MAPS_TO_FAULT":
            edge_color = 'red'
            edge_style = '--'
            edge_width = 1.5
        elif rel_type == "TOPOLOGY_DEPENDS_ON":
            edge_color = 'blue'
            edge_style = '-.'
            edge_width = 1.0
        else:
            edge_color = 'lightgray'
            edge_style = ':'
            edge_width = 0.5
            
        G.add_edge(source, target, color=edge_color, style=edge_style, width=edge_width)

    return G, entities, attributes, faults, topology_nodes

def visualize_detailed(G, output_path):
    """
    绘制详细图（包含拓扑依赖）
    """
    plt.figure(figsize=(24, 20))
    
    # 使用 kamada_kawai_layout 布局，适合复杂网络
    pos = nx.kamada_kawai_layout(G, weight=None)
    
    # 绘制节点
    node_colors = [G.nodes[node]['color'] for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, alpha=0.9, edgecolors='gray', linewidths=0.5)

    # 分类绘制边，以便控制层级
    edges = G.edges(data=True)
    
    # 先画拓扑依赖 (蓝色点划线)，在最下层
    topo_edges = [(u, v) for u, v, d in edges if d['style'] == '-.']
    nx.draw_networkx_edges(G, pos, edgelist=topo_edges, edge_color='blue', style='-.', width=1, alpha=0.4)

    # 再画异常关联 (灰色实线)
    anomaly_edges = [(u, v) for u, v, d in edges if d['style'] == '-']
    nx.draw_networkx_edges(G, pos, edgelist=anomaly_edges, edge_color='gray', style='-', width=0.5, alpha=0.3)

    # 最后画故障映射 (红色虚线)，在最上层，突出显示
    fault_edges = [(u, v) for u, v, d in edges if d['style'] == '--']
    nx.draw_networkx_edges(G, pos, edgelist=fault_edges, edge_color='red', style='--', width=2, alpha=0.7)

    # 标签 (只显示 Docker/DB 和 FaultType，避免遮挡)
    labels = {}
    for node in G.nodes():
        node_type = G.nodes[node]['type']
        if node_type in ['DOCKER', 'DB', 'FaultType']:
            labels[node] = G.nodes[node]['label']
            
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family="SimHei", bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    plt.title("知识图谱详细视图 (包含拓扑依赖)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "detailed_topology_view.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 详细视图已保存至: {os.path.join(output_path, 'detailed_topology_view.png')}")

def visualize_core_faults(G, entities, faults, output_path):
    """
    绘制核心故障视图（仅显示实体 -> 故障，忽略中间指标）
    """
    # 提取子图：只保留实体和故障节点
    core_nodes = set(entities.keys()) | set(faults.keys())
    core_subgraph = G.subgraph(core_nodes).copy()

    plt.figure(figsize=(18, 14))
    pos = nx.spring_layout(core_subgraph, k=5, iterations=100, seed=123)

    # 节点颜色
    node_colors = [G.nodes[node]['color'] for node in core_subgraph.nodes]

    nx.draw_networkx_nodes(core_subgraph, pos, node_color=node_colors, node_size=800, alpha=0.9, edgecolors='black')

    # 只画指向 FaultType 的边 (红色虚线)
    fault_edges = []
    for u, v in core_subgraph.edges():
        if G.nodes[v]['type'] == 'FaultType': # 只画指向故障的边
            fault_edges.append((u, v))
            
    nx.draw_networkx_edges(core_subgraph, pos, edgelist=fault_edges, edge_color='red', style='--', width=2, alpha=0.8)

    # 标签
    labels = {node: G.nodes[node]['label'] for node in core_subgraph.nodes}
    nx.draw_networkx_labels(core_subgraph, pos, labels, font_size=10, font_weight='bold')

    plt.title("核心故障关联视图 (实体 -> 故障类型)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "core_faults_view.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"🚨 核心故障视图已保存至: {os.path.join(output_path, 'core_faults_view.png')}")

def main():
    # 1. 加载数据
    data = load_kg_data(JSON_FILE)
    if not data:
        return

    # 2. 构建图
    G, entities, attributes, faults, topology = build_graph(data)

    # 3. 生成可视化
    print("🚀 开始生成可视化图像...")
    
    # 生成核心故障视图 (更清晰)
    visualize_core_faults(G, entities, faults, OUTPUT_DIR)
    
    # 生成详细视图 (信息全，但较乱)
    visualize_detailed(G, OUTPUT_DIR)

    print(f"🎉 所有图像已生成完毕！请查看目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
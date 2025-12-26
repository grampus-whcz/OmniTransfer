import csv
import json
from collections import defaultdict
import os

# 合法节点集合（用于过滤）
VALID_NODES = {
    "apache01", "apache02",
    "Tomcat01", "Tomcat02", "Tomcat03", "Tomcat04",
    "MG01", "MG02",
    "IG01", "IG02",
    "Mysql01", "Mysql02",
    "Redis01", "Redis02",
    "dockerA1", "dockerA2", "dockerB1", "dockerB2",
}

def build_call_graph_for_trace(spans):
    """构建单个 trace 的调用图（返回边的 frozenset）"""
    span_to_cmdb = {}
    for s in spans:
        cmdb = s['cmdb_id']
        if cmdb in VALID_NODES:
            span_to_cmdb[s['span_id']] = cmdb

    edges = set()
    for s in spans:
        cmdb = s['cmdb_id']
        if cmdb not in VALID_NODES:
            continue
        parent_id = s['parent_id']
        if parent_id in span_to_cmdb:
            caller = span_to_cmdb[parent_id]
            callee = cmdb
            if caller != callee:
                edges.add((caller, callee))
    return frozenset(edges)

def normalize_graph_to_list(graph_frozen):
    """将 frozenset of (u,v) 转为排序后的 list of [u,v]"""
    edge_list = [[u, v] for u, v in graph_frozen]
    # 排序以保证一致性（先按 caller，再按 callee）
    edge_list.sort(key=lambda x: (x[0], x[1]))
    return edge_list

def main(csv_path, output_json_path):
    # Step 1: Load and group by trace_id
    trace_spans = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            span_data = {
                'cmdb_id': row['cmdb_id'],
                'parent_id': row['parent_id'],
                'span_id': row['span_id'],
                'trace_id': row['trace_id'],
            }
            trace_spans[row['trace_id']].append(span_data)

    print(f"Loaded {len(trace_spans)} traces.")

    # Step 2: Build unique graphs
    unique_graphs_set = set()
    for spans in trace_spans.values():
        graph = build_call_graph_for_trace(spans)
        unique_graphs_set.add(graph)

    print(f"Found {len(unique_graphs_set)} unique call graph structures.")

    # Step 3: Convert to JSON-serializable list
    graphs_list = []
    for graph in unique_graphs_set:
        edge_list = normalize_graph_to_list(graph)
        if edge_list:  # 可选：跳过空图（无有效调用）
            graphs_list.append(edge_list)

    # 可选：对整个图列表排序（按第一个边，或边数）
    graphs_list.sort(key=lambda g: (len(g), g))

    # Step 4: Save to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(graphs_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Unique call graphs saved to: {output_json_path}")
    print(f"Total non-empty unique graphs: {len(graphs_list)}")

if __name__ == "__main__":
    csv_file = "/root/shared-nvme/work/agent/OpenRCA/dataset/Bank/telemetry/2021_03_05/trace/trace_span.csv"
    output_file = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ2/unique_call_graphs.json"

    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found: {csv_file}")
    else:
        main(csv_file, output_file)
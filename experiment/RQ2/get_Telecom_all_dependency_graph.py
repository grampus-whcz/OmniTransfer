import csv
import json
import os
from collections import defaultdict

# 合法节点集合（用于过滤）
VALID_NODES = {
    "os_001","os_002","os_003","os_004","os_005","os_006","os_007",
    "os_008","os_009","os_010","os_011","os_012","os_013","os_014","os_015","os_016",
    "os_017","os_018","os_019","os_020","os_021","os_022",
    "docker_001","docker_002","docker_003","docker_004","docker_005","docker_006","docker_007","docker_008",
    "db_001","db_002","db_003","db_004","db_005","db_006","db_007","db_008","db_009","db_010","db_011","db_012","db_013"
}

def build_call_graph_for_trace(spans):
    """
    构建单个 trace 的调用图（返回边的 frozenset）
    此函数专门处理Telecom数据，构建 cmdb_id -> dsName 的依赖关系。
    """
    dependency_edges = set()
    internal_call_map = {}

    for s in spans:
        cmdb_id = s['cmdb_id']
        ds_name = s['dsName'].strip()
        span_id = s['id']
        parent_span_id = s['pid']

        # 1. 处理依赖关系 (cmdb_id -> dsName)
        if cmdb_id in VALID_NODES and ds_name and ds_name in VALID_NODES:
            dependency_edges.add((cmdb_id, ds_name))
        
        # 2. 为内部调用准备映射 (仅处理cmdb_id合法的情况)
        if cmdb_id in VALID_NODES:
            internal_call_map[span_id] = cmdb_id

    # 3. 处理内部调用关系 (pid -> id)，基于internal_call_map
    for s in spans:
        parent_span_id = s['pid']
        current_span_id = s['id']
        # 如果父span和当前span都在map中，且它们指向不同的cmdb_id，则构成一条内部调用边
        if parent_span_id in internal_call_map and current_span_id in internal_call_map:
            caller_cmdb = internal_call_map[parent_span_id]
            callee_cmdb = internal_call_map[current_span_id]
            if caller_cmdb != callee_cmdb:
                 dependency_edges.add((caller_cmdb, callee_cmdb))

    return frozenset(dependency_edges)


def normalize_graph_to_list(graph_frozen):
    """将 frozenset of (u,v) 转为排序后的 list of [u,v]"""
    edge_list = [[u, v] for u, v in graph_frozen]
    # 排序以保证一致性（先按 caller，再按 callee）
    edge_list.sort(key=lambda x: (x[0], x[1]))
    return edge_list


def process_single_csv(csv_path, all_unique_graphs_set):
    """
    处理单个CSV文件，并将新发现的图添加到all_unique_graphs_set中。
    """
    print(f"Processing file: {csv_path}")
    
    # Step 1: Load and group by trace_id
    trace_spans = defaultdict(list)
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trace_spans[row['traceId']].append(row)
    except FileNotFoundError:
        print(f"Warning: File not found: {csv_path}")
        return
    except Exception as e:
        print(f"Error processing file {csv_path}: {e}")
        return

    print(f"  Loaded {len(trace_spans)} traces from {os.path.basename(csv_path)}.")

    # Step 2: Build unique graphs for this file and add to the global set
    for spans in trace_spans.values():
        graph = build_call_graph_for_trace(spans)
        all_unique_graphs_set.add(graph)


def main(telemetry_base_dir, output_json_path):
    """
    遍历所有日期目录，处理其中的trace_span.csv文件，汇总所有唯一的图。
    """
    # Step 1: Find all CSV files
    csv_files = []
    for date_dir in os.listdir(telemetry_base_dir):
        full_date_path = os.path.join(telemetry_base_dir, date_dir)
        if os.path.isdir(full_date_path):
            csv_path = os.path.join(full_date_path, "trace", "trace_span.csv")
            if os.path.exists(csv_path):
                csv_files.append(csv_path)
            else:
                print(f"Warning: trace_span.csv not found in {full_date_path}")

    print(f"Found {len(csv_files)} trace files to process.\n")

    # Step 2: Initialize a set to store all unique graphs across all files
    all_unique_graphs_set = set()

    # Step 3: Process each CSV file
    for csv_file_path in csv_files:
        process_single_csv(csv_file_path, all_unique_graphs_set)

    print(f"\nAggregated total unique graphs across all dates: {len(all_unique_graphs_set)}.")

    # Step 4: Convert to JSON-serializable list
    graphs_list = []
    for graph in all_unique_graphs_set:
        edge_list = normalize_graph_to_list(graph)
        if edge_list:  # 可选：跳过空图（无有效调用）
            graphs_list.append(edge_list)

    # 可选：对整个图列表排序（按第一个边，或边数）
    graphs_list.sort(key=lambda g: (len(g), g))

    # Step 5: Save to JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(graphs_list, f, indent=2, ensure_ascii=False)

    print(f"✅ All unique dependency graphs saved to: {output_json_path}")
    print(f"Total non-empty unique graphs in final list: {len(graphs_list)}")


if __name__ == "__main__":
    base_directory = "/root/shared-nvme/work/agent/OpenRCA/dataset/Telecom/telemetry"
    output_file = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ2/all_telecom_unique_dependency_graphs.json"

    main(base_directory, output_file)
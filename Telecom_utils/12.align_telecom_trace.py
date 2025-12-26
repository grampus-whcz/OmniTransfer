#!/usr/bin/env python3
# /root/shared-nvme/work/timeSeries/OmniTransfer_new/Telecom_utils/12.align_telecom_trace.py
import json
import numpy as np
import argparse

def load_meta_and_data(npy_path, meta_path):
    data = np.load(npy_path)  # (E, T, F)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    edges = meta["edges"]
    assert len(edges) == data.shape[0], f"Edge count mismatch in {meta_path}"
    return data, edges, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_npy", required=True)
    parser.add_argument("--offline_meta", required=True)
    parser.add_argument("--online_npy", required=True)
    parser.add_argument("--online_meta", required=True)
    parser.add_argument("--output_offline", required=True)
    parser.add_argument("--output_online", required=True)
    parser.add_argument("--output_info", required=True)
    args = parser.parse_args()

    offline_data, offline_edges, offline_meta = load_meta_and_data(args.offline_npy, args.offline_meta)
    online_data, online_edges, online_meta = load_meta_and_data(args.online_npy, args.online_meta)

    # 提取日期（从 meta.json 的 date_dir 字段）
    date_offline_str = offline_meta.get("date_dir", "unknown")
    date_online_str = online_meta.get("date_dir", "unknown")

    # 对齐 edges
    common_edges_set = set(offline_edges) & set(online_edges)
    common_edges = sorted(common_edges_set)  # 字典序
    num_common = len(common_edges)

    print(f"Offline edges: {len(offline_edges)} (date: {date_offline_str})")
    print(f"Online edges: {len(online_edges)} (date: {date_online_str})")
    print(f"Common edges after alignment: {num_common}")

    # 构建索引映射
    offline_idx_map = {edge: i for i, edge in enumerate(offline_edges)}
    online_idx_map = {edge: i for i, edge in enumerate(online_edges)}

    # 对齐数据
    aligned_offline = np.zeros((num_common, offline_data.shape[1], offline_data.shape[2]), dtype=offline_data.dtype)
    aligned_online = np.zeros((num_common, online_data.shape[1], online_data.shape[2]), dtype=online_data.dtype)

    for i, edge in enumerate(common_edges):
        aligned_offline[i] = offline_data[offline_idx_map[edge]]
        aligned_online[i] = online_data[online_idx_map[edge]]

    # 保存对齐后的数据
    np.save(args.output_offline, aligned_offline)
    np.save(args.output_online, aligned_online)

    # 构造 info.json
    info = {
        "level": "pod",
        "reason": "trace anomaly",
        "component": "call_graph",
        "fault_date": date_online_str.replace('_', '-'),
        "normal_date": date_offline_str.replace('_', '-'),
        "fault_start_time": f"{date_online_str[:4]}-{date_online_str[5:7]}-{date_online_str[8:]} 00:00:00",
        "normal_start_time": f"{date_offline_str[:4]}-{date_offline_str[5:7]}-{date_offline_str[8:]} 00:00:00",
        "bucket_sec": offline_meta["bucket_sec"],
        "duration_minutes": min(offline_data.shape[1], online_data.shape[1]) * offline_meta["bucket_sec"] // 60,
        "original_online_shape": list(online_data.shape),
        "original_offline_shape": list(offline_data.shape),
        "aligned_shape": [num_common, aligned_offline.shape[1], aligned_offline.shape[2]],
        "num_common_edges": num_common,
        "features": offline_meta["features"],
        "common_edges": common_edges,
        "note": "Edges aligned by intersection of fault and normal days. Sorted lexicographically."
    }

    with open(args.output_info, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"✅ Alignment completed. Info saved to: {args.output_info}")

if __name__ == "__main__":
    main()
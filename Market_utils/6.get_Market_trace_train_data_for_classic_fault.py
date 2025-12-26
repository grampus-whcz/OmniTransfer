import pandas as pd
import numpy as np
import json
from pathlib import Path

# ----------------------------
# 配置
# ----------------------------
RECORD_CSV = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Market_utils/cloudbed-2_record_with_normal_intervals.csv"
TRACE_DIR = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Market/trace/"
THRESHOLD = 1
BUCKET_SEC = 60  # 固定使用 60s 粒度

# ----------------------------
# 辅助函数：加载 trace metadata
# ----------------------------
def load_trace_metadata(cloudbed, date_str, bucket_sec=60):
    meta_path = TRACE_DIR + f"{cloudbed}_{date_str}_trace_edge_bucket_{bucket_sec}.meta.json"
    with open(meta_path, 'r') as f:
        return json.load(f)

def timestamp_to_trace_index(ts, meta):
    start_ts = meta["global_start_sec"]
    bucket_sec = meta["bucket_sec"]
    idx = int((ts - start_ts) // bucket_sec)
    if not (0 <= idx < meta["num_buckets"]):
        raise ValueError(
            f"Trace timestamp {ts} out of range [{start_ts}, {meta['global_end_sec']}) "
            f"for {meta['cloudbed']} {meta['date_dir']}"
        )
    return idx

# ----------------------------
# 注意不同日期中的trace边的排序不一样，且数目也不一样。需要对齐
# 按完整边（source->target:method）对齐两个 trace
# ----------------------------
def align_two_traces_by_edges(meta_a, trace_a, meta_b, trace_b):
    """
    对两个 trace 按完整边（视为原子单元）进行对齐。
    返回对齐后的 trace 和更新的 meta。
    """
    edges_a = meta_a["edges"]  # list of "src->tgt:method"
    edges_b = meta_b["edges"]

    set_a = set(edges_a)
    set_b = set(edges_b)
    common_edges = sorted(set_a & set_b)  # 字典序排序，确保确定性

    if not common_edges:
        raise ValueError("No common edges between two days!")

    # 构建边到索引的映射
    edge_to_idx_a = {edge: i for i, edge in enumerate(edges_a)}
    edge_to_idx_b = {edge: i for i, edge in enumerate(edges_b)}

    # 获取公共边在各自 trace 中的索引
    indices_a = [edge_to_idx_a[edge] for edge in common_edges]
    indices_b = [edge_to_idx_b[edge] for edge in common_edges]

    # 重排 trace: (num_edges, T, F) -> (K, T, F)
    aligned_trace_a = trace_a[indices_a]
    aligned_trace_b = trace_b[indices_b]

    # 更新 meta
    meta_a_aligned = {k: v for k, v in meta_a.items()}
    meta_b_aligned = {k: v for k, v in meta_b.items()}

    meta_a_aligned["edges"] = common_edges
    meta_b_aligned["edges"] = common_edges
    meta_a_aligned["num_edges"] = len(common_edges)
    meta_b_aligned["num_edges"] = len(common_edges)
    meta_a_aligned["common_edges_count"] = len(common_edges)

    return aligned_trace_a, meta_a_aligned, aligned_trace_b, meta_b_aligned

# ----------------------------
# 主程序
# ----------------------------
def main():
    df = pd.read_csv(RECORD_CSV)
    df = df.reset_index(drop=False)
    df.rename(columns={'index': 'csv_row_index'}, inplace=True)
    df['csv_row_number'] = df['csv_row_index'] + 1

    df['datetime'] = pd.to_datetime(df['datetime'])
    df['fault_slot_start'] = pd.to_datetime(df['fault_slot_start'])
    df['date_str'] = df['datetime'].dt.strftime('%Y-%m-%d')

    unique_dates = sorted(df['date_str'].unique())
    if len(unique_dates) != 2:
        raise ValueError(f"Expected 2 dates, got: {unique_dates}")
    date_a, date_b = unique_dates
    print(f"✅ 检测到两天: {date_a}, {date_b}\n")

    df['fault_pattern'] = df['level'].astype(str) + " | " + df['reason'].astype(str)
    pattern_counts = df['fault_pattern'].value_counts()
    typical_patterns = pattern_counts[pattern_counts >= THRESHOLD].index

    print(f"🔍 共 {len(typical_patterns)} 种典型故障模式\n")

    for i, pattern in enumerate(typical_patterns, 1):
        row = df[df['fault_pattern'] == pattern].iloc[0]
        fault_date = row['date_str']
        normal_date = date_b if fault_date == date_a else date_a
        row_num = int(row['csv_row_number'])

        print(f"[{i}] 模式: {pattern}")
        print(f"    组件: {row['component']}, CSV行号: {row_num}")

        # --- 加载原始 trace 和 meta ---
        meta_f = load_trace_metadata("cloudbed-1", fault_date.replace('-', '_'), BUCKET_SEC)
        meta_n = load_trace_metadata("cloudbed-1", normal_date.replace('-', '_'), BUCKET_SEC)

        trace_f = np.load(TRACE_DIR + f"cloudbed-1_{fault_date.replace('-', '_')}_trace_edge_bucket_{BUCKET_SEC}.npy")
        trace_n = np.load(TRACE_DIR + f"cloudbed-1_{normal_date.replace('-', '_')}_trace_edge_bucket_{BUCKET_SEC}.npy")

        # --- 按边对齐 ---
        try:
            aligned_trace_f, aligned_meta_f, aligned_trace_n, aligned_meta_n = align_two_traces_by_edges(
                meta_f, trace_f, meta_n, trace_n
            )
        except ValueError as e:
            print(f"    ⚠️ 跳过模式 {i}: {e}")
            continue

        # --- 时间索引计算（基于对齐后的 meta）---
        fault_slot_start_ts = int(row['fault_slot_start'].timestamp())
        normal_start_ts = row['normal_start_timestamp']

        start_idx_f = timestamp_to_trace_index(fault_slot_start_ts, aligned_meta_f)
        start_idx_n = timestamp_to_trace_index(normal_start_ts, aligned_meta_n)

        end_idx_f = start_idx_f + 30
        end_idx_n = start_idx_n + 30

        # 截取 30 个时间桶（5 分钟）
        online_trace = aligned_trace_f[:, start_idx_f:end_idx_f, :]  # (K, 30, 2)
        offline_trace = aligned_trace_n[:, start_idx_n:end_idx_n, :]  # (K, 30, 2)

        # --- 保存 ---
        comp_safe = row['component'].replace('-', '_').replace('.', '_')
        output_dir = Path(f"./trace_fault_case_set/fault_case_{i:02d}_row{row_num}_{row['level']}_{comp_safe}_{fault_date}")
        output_dir.mkdir(exist_ok=True)

        np.save(output_dir / "online_data.npy", online_trace)
        np.save(output_dir / "offline_data.npy", offline_trace)

        # --- 保存 info.json ---
        info_path = output_dir / "info.json"
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
        else:
            info = {}

        info.update({
            "trace_bucket_sec": BUCKET_SEC,
            "fault_exact_time_stamp": str(row['timestamp']),
            "fault_exact_time": str(row['datetime']),
            "fault_slot_start": str(row['fault_slot_start']),
            "csv_row_number": row_num,
            "fault_date": fault_date,
            "normal_date": normal_date,
            "trace_online_shape": online_trace.shape,
            "trace_offline_shape": offline_trace.shape,
            "trace_num_edges": aligned_meta_f["num_edges"],
            "trace_features": aligned_meta_f["features"],
            "trace_edges_sample": aligned_meta_f["edges"][:3],
            "note_on_edge_format": "Each edge is treated as an atomic unit in the format 'source->target:method'. Do not split.",
            "aligned_common_edges_count": len(aligned_meta_f["edges"]),
            "aligned_common_edges_sample": aligned_meta_f["edges"]
        })

        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"    ✅ 对齐并保存 Trace 数据到: {output_dir.absolute()}")
        print("-" * 80)

    print("\n🎉 所有典型故障模式的对齐 trace 数据提取完成！")

if __name__ == "__main__":
    main()
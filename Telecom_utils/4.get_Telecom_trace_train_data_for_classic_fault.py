import pandas as pd
import numpy as np
import json
from pathlib import Path

# ----------------------------
# 配置
# ----------------------------
REPRESENTATIVES_CSV = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Telecom_utils/telecom_fault_patterns_representatives.csv"
TRACE_DIR = Path("/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Telecom/trace")
OUTPUT_ROOT = Path("./telecom_trace_case_set")
BUCKET_SEC = 60
DURATION_BUCKETS = 30  # 30 minutes

def load_trace_meta_and_data(date_str):
    meta_path = TRACE_DIR / f"telecom_{date_str}_trace_edge_bucket_{BUCKET_SEC}.meta.json"
    data_path = TRACE_DIR / f"telecom_{date_str}_trace_edge_bucket_{BUCKET_SEC}.npy"
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    data = np.load(data_path)
    return meta, data

def timestamp_to_index(ts, meta):
    start = meta["global_start_sec"]
    bucket_sec = meta["bucket_sec"]
    idx = int((ts - start) // bucket_sec)
    if not (0 <= idx < meta["num_buckets"]):
        raise ValueError(f"Timestamp {ts} out of range for trace on {meta['date_dir']}")
    return idx

# important: align edges
def align_edges_and_data(meta_f, data_f, meta_n, data_n):
    edges_f = meta_f["edges"]
    edges_n = meta_n["edges"]
    
    set_f = set(edges_f)
    set_n = set(edges_n)
    common_edges = sorted(set_f & set_n)  # 排序确保顺序一致
    
    if not common_edges:
        raise ValueError("No common edges between fault and normal day!")
    
    # 获取索引
    idx_f = [edges_f.index(e) for e in common_edges]
    idx_n = [edges_n.index(e) for e in common_edges]
    
    # 提取对齐后的数据
    aligned_f = data_f[idx_f]  # (E_common, T, F)
    aligned_n = data_n[idx_n]  # (E_common, T, F)
    
    return common_edges, aligned_f, aligned_n

def main():
    df = pd.read_csv(REPRESENTATIVES_CSV)
    df = df.reset_index(drop=False)
    df.rename(columns={'index': 'case_id'}, inplace=True)
    df['case_id'] = df['case_id'] + 1

    OUTPUT_ROOT.mkdir(exist_ok=True)

    for _, row in df.iterrows():
        case_id = row['case_id']
        level = row['level']
        reason = row['reason']
        component = row['component']
        fault_date = row['datetime'].split(' ')[0]  # e.g., "2020-04-11"
        normal_date = "2020-04-20"

        fault_start_ts = int(row['fault_start_timestamp'])
        normal_start_ts = int(row['normal_start_timestamp'])

        comp_safe = component.replace('-', '_').replace('.', '_')
        dir_name = f"fault_case_{case_id:02d}_{level}_{reason.replace(' ', '_')}_{comp_safe}_{fault_date}"
        output_dir = OUTPUT_ROOT / dir_name
        output_dir.mkdir(exist_ok=True)

        print(f"[{case_id}] 处理 trace: {level} | {reason} | {component} | {fault_date}")

        try:
            date_f_str = fault_date.replace('-', '_')
            date_n_str = normal_date.replace('-', '_')

            # 加载故障日和正常日的 trace 数据与元信息
            meta_f, data_f = load_trace_meta_and_data(date_f_str)
            meta_n, data_n = load_trace_meta_and_data(date_n_str)

            # 对齐公共边
            common_edges, aligned_f, aligned_n = align_edges_and_data(meta_f, data_f, meta_n, data_n)

            # 时间索引
            start_idx_f = timestamp_to_index(fault_start_ts, meta_f)
            start_idx_n = timestamp_to_index(normal_start_ts, meta_n)
            end_idx_f = start_idx_f + DURATION_BUCKETS
            end_idx_n = start_idx_n + DURATION_BUCKETS

            # 截取 30 分钟窗口
            online = aligned_f[:, start_idx_f:end_idx_f, :]   # (E, 30, 2)
            offline = aligned_n[:, start_idx_n:end_idx_n, :]  # (E, 30, 2)

            # 保存 .npy
            np.save(output_dir / "online_data.npy", online)
            np.save(output_dir / "offline_data.npy", offline)

            # 构建 info.json
            info = {
                "level": level,
                "reason": reason,
                "component": component,
                "fault_date": fault_date,
                "normal_date": normal_date,
                "fault_start_time": row['fault_start_time'],
                "normal_start_time": row['normal_start_time'],
                "bucket_sec": BUCKET_SEC,
                "duration_minutes": DURATION_BUCKETS,
                "original_online_shape": list(data_f.shape),
                "original_offline_shape": list(data_n.shape),
                "aligned_shape": list(online.shape),
                "num_common_edges": len(common_edges),
                "features": meta_f["features"],
                "common_edges_sample": common_edges,
                "note": "Edges aligned by intersection of fault and normal days. Sorted lexicographically."
            }

            # 可选：保存完整公共边列表（若太多可注释）
            info["common_edges"] = common_edges

            with open(output_dir / "info.json", 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            print(f"    ✅ 对齐后 shape: {online.shape} | 公共边数: {len(common_edges)}")

        except Exception as e:
            print(f"    ❌ 失败: {e}")

        print("-" * 80)

    print("\n🎉 Telecom trace 数据已对齐并保存完成！输出目录:", OUTPUT_ROOT.resolve())

if __name__ == "__main__":
    main()
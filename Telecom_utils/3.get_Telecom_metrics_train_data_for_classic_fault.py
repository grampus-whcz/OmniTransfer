import pandas as pd
import numpy as np
import json
from pathlib import Path

# ----------------------------
# 配置
# ----------------------------
REPRESENTATIVES_CSV = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Telecom_utils/telecom_fault_patterns_representatives.csv"
METRIC_DIR = Path("/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Telecom/metric")
OUTPUT_ROOT_A = Path("./telecom_metric_A_case_set")
OUTPUT_ROOT_B = Path("./telecom_metric_B_case_set")
BUCKET_SEC = 60
DURATION_BUCKETS = 30  # 30 minutes

def load_metadata(date_str, entity_type):
    meta_path = METRIC_DIR / f"metadata_{entity_type}_{date_str}_60s.json"
    with open(meta_path, 'r') as f:
        return json.load(f)

def timestamp_to_index(ts, meta):
    start = meta["global_start_sec"]
    bucket_sec = meta["bucket_sec"]
    idx = int((ts - start) // bucket_sec)
    if not (0 <= idx < meta["num_buckets"]):
        raise ValueError(f"Timestamp {ts} out of range for {meta['date_dir']} entity_{meta['entity_type']}")
    return idx

def align_entities_and_data(meta_f, data_f, meta_n, data_n):
    entities_f = meta_f["entities"]
    entities_n = meta_n["entities"]
    
    common_entities = sorted(set(entities_f) & set(entities_n))
    if not common_entities:
        raise ValueError("No common entities between two days!")
    
    idx_f = [entities_f.index(e) for e in common_entities]
    idx_n = [entities_n.index(e) for e in common_entities]
    
    aligned_data_f = data_f[idx_f]
    aligned_data_n = data_n[idx_n]
    
    meta_f_aligned = {k: v for k, v in meta_f.items()}
    meta_n_aligned = {k: v for k, v in meta_n.items()}
    meta_f_aligned["entities"] = common_entities
    meta_n_aligned["entities"] = common_entities
    meta_f_aligned["num_entities"] = len(common_entities)
    meta_n_aligned["num_entities"] = len(common_entities)
    
    return aligned_data_f, meta_f_aligned, aligned_data_n, meta_n_aligned

def process_entity_type(entity_type, representatives_df):
    output_root = OUTPUT_ROOT_A if entity_type == "A" else OUTPUT_ROOT_B
    output_root.mkdir(exist_ok=True)

    for _, row in representatives_df.iterrows():
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
        output_dir = output_root / dir_name
        output_dir.mkdir(exist_ok=True)

        try:
            date_f_str = fault_date.replace('-', '_')
            date_n_str = normal_date.replace('-', '_')

            # 加载 metadata 和数据
            meta_f = load_metadata(date_f_str, entity_type)
            meta_n = load_metadata(date_n_str, entity_type)
            data_f = np.load(METRIC_DIR / f"entity_{entity_type}_{date_f_str}_60s.npy")
            data_n = np.load(METRIC_DIR / f"entity_{entity_type}_{date_n_str}_60s.npy")

            # 对齐实体
            aligned_f, meta_f_a, aligned_n, meta_n_a = align_entities_and_data(
                meta_f, data_f, meta_n, data_n
            )

            # 时间索引
            start_idx_f = timestamp_to_index(fault_start_ts, meta_f_a)
            start_idx_n = timestamp_to_index(normal_start_ts, meta_n_a)
            end_idx_f = start_idx_f + DURATION_BUCKETS
            end_idx_n = start_idx_n + DURATION_BUCKETS

            # 截取 30 分钟
            online = aligned_f[:, start_idx_f:end_idx_f, :]
            offline = aligned_n[:, start_idx_n:end_idx_n, :]

            # 保存 .npy
            np.save(output_dir / "online_data.npy", online)
            np.save(output_dir / "offline_data.npy", offline)

            # 构建 info.json
            info = {
                "entity_type": f"entity_{entity_type}",
                "level": level,
                "reason": reason,
                "component": component,
                "fault_date": fault_date,
                "normal_date": normal_date,
                "fault_start_time": row['fault_start_time'],
                "fault_end_time": row['fault_end_time'],
                "normal_start_time": row['normal_start_time'],
                "normal_end_time": row['normal_end_time'],
                "bucket_sec": BUCKET_SEC,
                "duration_minutes": DURATION_BUCKETS,
                "online_shape": online.shape,
                "offline_shape": offline.shape,
                "common_entity_count": len(meta_f_a["entities"]),
                "sample_entities": meta_f_a["entities"],
                "features_sample": meta_f_a["features"],
                "feature_count": len(meta_f_a["features"]),
                "note": "Data aligned by common entities across fault and normal days."
            }

            with open(output_dir / "info.json", 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            print(f"✅ [{entity_type}] {dir_name} → online={online.shape}, offline={offline.shape}")

        except Exception as e:
            print(f"❌ [{entity_type}] {dir_name} 失败: {e}")

def main():
    df = pd.read_csv(REPRESENTATIVES_CSV)
    df = df.reset_index(drop=False)
    df.rename(columns={'index': 'case_id'}, inplace=True)
    df['case_id'] = df['case_id'] + 1

    print("🚀 开始处理 entity A...")
    process_entity_type("A", df)

    print("\n🚀 开始处理 entity B...")
    process_entity_type("B", df)

    print("\n🎉 Telecom 指标数据已按 entity 类型分离保存完成！")

if __name__ == "__main__":
    main()
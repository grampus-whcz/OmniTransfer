import pandas as pd
import numpy as np
import json
from pathlib import Path

# ----------------------------
# 配置
# ----------------------------
RECORD_CSV = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Market_utils/cloudbed-2_record_with_normal_intervals.csv"
LOG_BASE_DIR = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Market/log/temp_data/raw_data/train_valid"
BUCKET_SEC = 60  # 60s per bucket

# 输出根目录
OUTPUT_ROOTS = {
    "service_patterns": Path("./log_service_patterns_case_set"),
    "istio_patterns": Path("./log_istio_patterns_case_set"),
    "istio_words": Path("./log_istio_word_case_set")
}

# 日志文件名映射
LOG_FILES = {
    "service_patterns": "service_log_patterns_count.npy",
    "istio_patterns": "istio_log_patterns_count.npy",
    "istio_words": "istio_word_count.npy"
}
META_FILES = {
    "service_patterns": "service_log_patterns_count_meta.npy",
    "istio_patterns": "istio_log_patterns_count_meta.npy",
    "istio_words": "istio_word_count_meta.npy"
}

# ----------------------------
# 加载日志元数据
# ----------------------------
def load_log_meta(date_str, cloudbed="cloudbed-2", log_type="service_patterns"):
    date_dir = date_str.replace('-', '_')
    meta_path = Path(LOG_BASE_DIR) / date_dir / cloudbed / "raw_log" / META_FILES[log_type]
    meta = np.load(meta_path, allow_pickle=True).item()
    return meta

def timestamp_to_bucket_index(ts, meta):
    timestamps = meta["timestamps"]
    if ts < timestamps[0] or ts >= timestamps[-1] + BUCKET_SEC:
        raise ValueError(f"Timestamp {ts} out of log range [{timestamps[0]}, {timestamps[-1] + BUCKET_SEC})")
    idx = int((ts - timestamps[0]) // BUCKET_SEC)
    return idx

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
    typical_patterns = pattern_counts[pattern_counts >= 1].index  # THRESHOLD=1

    print(f"🔍 共 {len(typical_patterns)} 种典型故障模式\n")

    for i, pattern in enumerate(typical_patterns, 1):
        row = df[df['fault_pattern'] == pattern].iloc[0]
        fault_date = row['date_str']
        normal_date = date_b if fault_date == date_a else date_a
        row_num = int(row['csv_row_number'])
        component = row['component']

        print(f"[{i}] 模式: {pattern}")
        print(f"    组件: {component}, CSV行号: {row_num}")

        # 时间戳
        fault_ts = int(row['fault_slot_start'].timestamp())
        normal_ts = row['normal_start_timestamp']

        # 创建输出目录（三个）
        comp_safe = component.replace('-', '_').replace('.', '_')
        case_name = f"fault_case_{i:02d}_row{row_num}_{row['level']}_{comp_safe}_{fault_date}"

        output_dirs = {
            k: OUTPUT_ROOTS[k] / case_name for k in OUTPUT_ROOTS
        }
        for d in output_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # 处理三种日志类型
        for log_type in ["service_patterns", "istio_patterns", "istio_words"]:
            try:
                # 加载 meta（用于获取 timestamps）
                meta_f = load_log_meta(fault_date, "cloudbed-2", log_type)
                meta_n = load_log_meta(normal_date, "cloudbed-2", log_type)

                # 计算索引
                start_idx_f = timestamp_to_bucket_index(fault_ts, meta_f)
                start_idx_n = timestamp_to_bucket_index(normal_ts, meta_n)
                end_idx_f = start_idx_f + 30
                end_idx_n = start_idx_n + 30

                # 构建路径
                date_f_str = fault_date.replace('-', '_')
                date_n_str = normal_date.replace('-', '_')
                log_path_f = Path(LOG_BASE_DIR) / date_f_str / "cloudbed-2" / "raw_log" / LOG_FILES[log_type]
                log_path_n = Path(LOG_BASE_DIR) / date_n_str / "cloudbed-2" / "raw_log" / LOG_FILES[log_type]

                # 加载完整日志
                log_data_f = np.load(log_path_f)
                log_data_n = np.load(log_path_n)

                # 截取 30 个时间桶
                online_log = log_data_f[:, start_idx_f:end_idx_f, :]
                offline_log = log_data_n[:, start_idx_n:end_idx_n, :]

                # 保存
                np.save(output_dirs[log_type] / "online_data.npy", online_log)
                np.save(output_dirs[log_type] / "offline_data.npy", offline_log)

                print(f"    ✅ {log_type}: online={online_log.shape}, offline={offline_log.shape}")

            except Exception as e:
                print(f"    ❌ {log_type} 提取失败: {e}")
                continue

        # 保存 info.json（以 service_patterns 的 meta 为准）
        try:
            meta_info = load_log_meta(fault_date, "cloudbed-2", "service_patterns")
            info = {
                "fault_exact_time_stamp": str(row['timestamp']),
                "fault_exact_time": str(row['datetime']),
                "fault_slot_start": str(row['fault_slot_start']),
                "csv_row_number": row_num,
                "fault_date": fault_date,
                "normal_date": normal_date,
                "log_pods": meta_info["pods"],
                "log_pod_count": len(meta_info["pods"]),
                "log_bucket_sec": BUCKET_SEC,
                "log_online_shape_service": np.load(output_dirs["service_patterns"] / "online_data.npy").shape,
                "log_offline_shape_service": np.load(output_dirs["service_patterns"] / "offline_data.npy").shape,
                "log_online_shape_istio_patterns": np.load(output_dirs["istio_patterns"] / "online_data.npy").shape,
                "log_offline_shape_istio_patterns": np.load(output_dirs["istio_patterns"] / "offline_data.npy").shape,
                "log_online_shape_istio_words": np.load(output_dirs["istio_words"] / "online_data.npy").shape,
                "log_offline_shape_istio_words": np.load(output_dirs["istio_words"] / "offline_data.npy").shape,
                "note": "Each log feature array has shape (num_pods, 30, num_features). Pods order is consistent across all files."
            }

            with open(output_dirs["service_patterns"] / "info.json", 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            # 可选：复制到其他目录
            for k in ["istio_patterns", "istio_words"]:
                (output_dirs[k] / "info.json").write_text((output_dirs["service_patterns"] / "info.json").read_text())

        except Exception as e:
            print(f"    ⚠️ info.json 生成失败: {e}")

        print("-" * 80)

    print("\n🎉 所有日志数据提取完成！")

if __name__ == "__main__":
    main()
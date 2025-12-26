import pandas as pd
import numpy as np
import json
from pathlib import Path

# ----------------------------
# 配置
# ----------------------------
RECORD_CSV = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Market_utils/cloudbed-2_record_with_normal_intervals.csv"
BASE_DIR = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Market/metric/market_processed/"
THRESHOLD = 1

# ----------------------------
# 辅助函数 mesh / container
# ----------------------------
def load_metadata(date_str):
    meta_path = BASE_DIR + f"metadata_mesh_cloudbed-2_{date_str}_60s.json"
    with open(meta_path, 'r') as f:
        return json.load(f)

def timestamp_to_index(ts, meta):
    start_ts = meta["time_range_sec"][0]
    bucket_sec = meta["bucket_sec"]
    idx = int((ts - start_ts) // bucket_sec)
    if not (0 <= idx < meta["num_buckets"]):
        raise ValueError(
            f"Timestamp {ts} ({pd.to_datetime(ts, unit='s', utc=True).tz_convert('Asia/Shanghai')}) "
            f"is out of range for date {meta['date_dir']}"
        )
    return idx

# ----------------------------
# 主程序
# ----------------------------
def main():
    # 读取 CSV，并保留原始行号（从 1 开始）
    df = pd.read_csv(RECORD_CSV)
    df = df.reset_index(drop=False)  # 保留原始索引为 'index'
    df.rename(columns={'index': 'csv_row_index'}, inplace=True)
    df['csv_row_number'] = df['csv_row_index'] + 1  # 行号从 1 开始

    # 解析时间列
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['fault_slot_start'] = pd.to_datetime(df['fault_slot_start'])
    df['date_str'] = df['datetime'].dt.strftime('%Y-%m-%d')

    # 获取两天日期
    unique_dates = sorted(df['date_str'].unique())
    if len(unique_dates) != 2:
        raise ValueError(f"Expected exactly 2 dates, got: {unique_dates}")
    date_a, date_b = unique_dates
    print(f"✅ 检测到两天数据: {date_a} 和 {date_b}\n")

    # 构造故障模式
    df['fault_pattern'] = df['level'].astype(str) + " | " + df['reason'].astype(str)
    pattern_counts = df['fault_pattern'].value_counts()
    typical_patterns = pattern_counts[pattern_counts >= THRESHOLD].index

    print(f"🔍 共发现 {len(typical_patterns)} 种典型故障模式（出现次数 ≥ {THRESHOLD}）\n")

    for i, pattern in enumerate(typical_patterns, 1):
        # 取第一个匹配行（保留其原始行号）
        row = df[df['fault_pattern'] == pattern].iloc[0]

        fault_date = row['date_str']
        normal_date = date_b if fault_date == date_a else date_a
        row_num = int(row['csv_row_number'])

        print(f"[{i}] 模式: {pattern}")
        print(f"    组件: {row['component']}")
        print(f"    故障确切时间: {row['datetime']}")
        print(f"    CSV 行号: {row_num}, 故障日: {fault_date}, 正常日: {normal_date}")

        # === 故障数据（online）===
        fault_slot_start_ts = int(row['fault_slot_start'].timestamp())
        fault_date_us = fault_date.replace('-', '_')
        meta_f = load_metadata(fault_date_us)
        start_idx_f = timestamp_to_index(fault_slot_start_ts, meta_f)
        end_idx_f = start_idx_f + 30

        data_f = np.load(BASE_DIR + f"mesh_cloudbed-2_{fault_date_us}_60s.npy")
        online_data = data_f[:, start_idx_f:end_idx_f, :]

        # === 正常数据（offline）===
        normal_start_ts = row['normal_start_timestamp']
        normal_date_us = normal_date.replace('-', '_')
        meta_n = load_metadata(normal_date_us)
        start_idx_n = timestamp_to_index(normal_start_ts, meta_n)
        end_idx_n = start_idx_n + 30

        data_n = np.load(BASE_DIR + f"mesh_cloudbed-2_{normal_date_us}_60s.npy")
        offline_data = data_n[:, start_idx_n:end_idx_n, :]

        # === 构建输出目录名（含行号）===
        comp_safe = row['component'].replace('-', '_').replace('.', '_')
        output_dir = Path(f"./mesh_fault_case_set/fault_case_{i:02d}_row{row_num}_{row['level']}_{comp_safe}_{fault_date}")
        output_dir.mkdir(exist_ok=True)

        # 保存数据
        np.save(output_dir / "online_data.npy", online_data)
        np.save(output_dir / "offline_data.npy", offline_data)

        # 保存详细信息
        info = {
            "fault_pattern": pattern,
            "level": row['level'],
            "component": row['component'],
            "fault_exact_time_stamp": str(row['timestamp']),
            "fault_exact_time": str(row['datetime']),
            "fault_slot_start": str(row['fault_slot_start']),
            "csv_row_number": row_num,
            "fault_date": fault_date,
            "normal_date": normal_date,
            "normal_interval": f"{row['normal_start_time']} -> {row['normal_end_time']}",
            "online_shape": online_data.shape,
            "offline_shape": offline_data.shape,
            "entity_list_sample": meta_f["entity_list"][:5],
            "kpi_count": len(meta_f["kpi_list"])
        }
        with open(output_dir / "info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"    ✅ 已保存到: {output_dir.absolute()}")
        print("-" * 80)

    print("\n🎉 所有典型故障案例提取完成！")

if __name__ == "__main__":
    main()
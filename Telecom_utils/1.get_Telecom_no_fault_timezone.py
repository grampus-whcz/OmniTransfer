import pandas as pd
from datetime import datetime, timezone, timedelta

# 输入输出路径
INPUT_CSV = "/root/shared-nvme/work/agent/OpenRCA/dataset/Telecom/record.csv"
OUTPUT_CSV = "./record_with_intervals.csv"

# 定义 UTC+8 时区
UTC8 = timezone(timedelta(hours=8))

def align_to_half_hour_utc8(ts):
    """
    将 Unix 时间戳（秒）按 UTC+8 本地时间向下对齐到最近的 30 分钟边界，
    返回对齐后的 UTC+8 datetime 和对应的 Unix 时间戳。
    """
    # 先转为 UTC 时间，再转为 UTC+8
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_utc8 = dt_utc.astimezone(UTC8)
    
    # 对齐到 30 分钟（在 UTC+8 下操作）
    minute = (dt_utc8.minute // 30) * 30
    aligned_utc8 = dt_utc8.replace(minute=minute, second=0, microsecond=0)
    
    # 转回 Unix 时间戳（注意：timestamp() 自动处理时区）
    aligned_ts = int(aligned_utc8.timestamp())
    return aligned_utc8, aligned_ts

def main():
    df = pd.read_csv(INPUT_CSV)
    df['timestamp'] = df['timestamp'].astype(int)

    # 新列容器
    fault_start_ts_list = []
    fault_end_ts_list = []
    normal_start_ts_list = []
    normal_end_ts_list = []

    for _, row in df.iterrows():
        ts = row['timestamp']
        
        # === 故障时段对齐（UTC+8）===
        fault_aligned_dt, fault_start_ts = align_to_half_hour_utc8(ts)
        fault_end_ts = fault_start_ts + 30 * 60

        # === 正常时段：固定日期 2020-04-20（UTC+8），相同小时和分钟 ===
        normal_dt = fault_aligned_dt.replace(year=2020, month=4, day=20)
        normal_start_ts = int(normal_dt.timestamp())
        normal_end_ts = normal_start_ts + 30 * 60

        fault_start_ts_list.append(fault_start_ts)
        fault_end_ts_list.append(fault_end_ts)
        normal_start_ts_list.append(normal_start_ts)
        normal_end_ts_list.append(normal_end_ts)

    # 添加时间戳列
    df['fault_start_timestamp'] = fault_start_ts_list
    df['fault_end_timestamp'] = fault_end_ts_list
    df['normal_start_timestamp'] = normal_start_ts_list
    df['normal_end_timestamp'] = normal_end_ts_list

    # 转换为 UTC+8 的可读时间字符串（格式 YYYY-MM-DD HH:MM:SS）
    def ts_to_utc8_str(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(UTC8).strftime('%Y-%m-%d %H:%M:%S')

    df['fault_start_time'] = df['fault_start_timestamp'].apply(ts_to_utc8_str)
    df['fault_end_time'] = df['fault_end_timestamp'].apply(ts_to_utc8_str)
    df['normal_start_time'] = df['normal_start_timestamp'].apply(ts_to_utc8_str)
    df['normal_end_time'] = df['normal_end_timestamp'].apply(ts_to_utc8_str)

    # 列顺序
    cols = [
        'level', 'reason', 'component', 'timestamp', 'datetime',
        'fault_start_timestamp', 'fault_start_time',
        'fault_end_timestamp', 'fault_end_time',
        'normal_start_timestamp', 'normal_start_time',
        'normal_end_timestamp', 'normal_end_time'
    ]
    df = df[cols]

    # 保存
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 已生成新文件（正确处理 UTC+8）: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
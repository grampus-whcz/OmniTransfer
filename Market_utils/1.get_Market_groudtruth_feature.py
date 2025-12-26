import pandas as pd
from datetime import datetime

# 文件路径
file_path = "/root/shared-nvme/work/agent/OpenRCA/dataset/Market/cloudbed-2/record.csv"

# 读取 CSV 文件
df = pd.read_csv(file_path)

# 将 datetime 列转换为 datetime 类型（确保正确解析）
df['datetime'] = pd.to_datetime(df['datetime'])

# 按 datetime 升序排序
df = df.sort_values(by='datetime').reset_index(drop=True)

# 计算相邻行之间的时间差（单位：秒）
df['time_diff_sec'] = df['datetime'].diff().dt.total_seconds()

# 第一行没有前驱，time_diff_sec 为 NaN，丢弃
df_diff = df.dropna(subset=['time_diff_sec']).copy()

# 按时间间隔降序排序，取前10个最大间隔
top_10 = df_diff.nlargest(10, 'time_diff_sec')

# 输出结果
print("cloudbed-2 Top 10 largest time intervals between consecutive faults:\n")
for idx, row in top_10.iterrows():
    prev_time = df.loc[idx - 1, 'datetime']
    curr_time = row['datetime']
    diff_hours = row['time_diff_sec'] / 3600  # 转换为小时便于阅读
    print(f"From: {prev_time}  →  To: {curr_time}")
    print(f"  Interval: {row['time_diff_sec']:.1f} seconds ({diff_hours:.2f} hours)")
    print(f"  Previous fault: {df.loc[idx - 1, 'component']} ({df.loc[idx - 1, 'reason']})")
    print(f"  Current fault:  {row['component']} ({row['reason']})")
    print("-" * 80)
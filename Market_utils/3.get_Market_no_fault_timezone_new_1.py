import pandas as pd
from datetime import timedelta, date

# ----------------------------
# 配置
# ----------------------------
INPUT_CSV = "/root/shared-nvme/work/agent/OpenRCA/dataset/Market/cloudbed-1/record.csv"
OUTPUT_CSV = "./cloudbed-1_record_with_normal_intervals_new_new.csv"

# ----------------------------
# 加载数据
# ----------------------------
df = pd.read_csv(INPUT_CSV)
# 转换datetime并添加时区，保留原始格式便于后续处理
df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('Asia/Shanghai')
df['date'] = df['datetime'].dt.date
df = df.sort_values('datetime').reset_index(drop=True)

unique_dates = sorted(df['date'].unique())
if len(unique_dates) != 2:
    raise ValueError(f"期望恰好两天的数据，但检测到 {len(unique_dates)} 天")
date1, date2 = unique_dates
print(f"检测到两天数据：{date1} 和 {date2}")

# ----------------------------
# 辅助函数
# ----------------------------
def assign_to_slot(dt):
    """将时间向下取整到最近的30分钟槽位起点"""
    return dt.floor('30T')

def format_time(dt):
    """格式化时间为字符串（不含时区后缀，保持结果整洁）"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# ----------------------------
# 步骤1: 标记故障 slot 并计算故障窗口
# ----------------------------
df['fault_slot_start'] = df['datetime'].apply(assign_to_slot)
# 计算故障窗口：[fault_slot_start - 30min, fault_slot_start + 30min)
df['fault_window_start'] = df['fault_slot_start'] - timedelta(minutes=30)
df['fault_window_end'] = df['fault_slot_start'] + timedelta(minutes=30)

# ----------------------------
# 步骤2: 直接匹配相邻日期的对应正常窗口
# ----------------------------
results = []

for _, row in df.iterrows():
    fault_win_start = row['fault_window_start']
    fault_win_end = row['fault_window_end']
    current_date = fault_win_start.date()

    # 确定目标日期（切换为另一天）
    if current_date == date1:
        target_date = date2
    else:
        target_date = date1

    # 构造对应时间的正常窗口（仅替换日期，时间部分完全一致）
    # 先提取时间部分，再与目标日期组合，并保留时区
    normal_win_start = pd.Timestamp.combine(
        target_date,
        fault_win_start.tz_localize(None).time()  # 临时去除时区提取时间
    ).tz_localize('Asia/Shanghai')
    
    normal_win_end = pd.Timestamp.combine(
        target_date,
        fault_win_end.tz_localize(None).time()
    ).tz_localize('Asia/Shanghai')

    # 修复：计算两个Timestamp的中点（正常窗口中心），避免直接使用+和/
    # 方法1：通过timedelta计算（推荐，保持Timestamp类型，兼容时区）
    normal_window_mid = normal_win_start + (normal_win_end - normal_win_start) / 2

    # 构建结果行
    new_row = {
        **row.to_dict(),
        # 故障窗口格式化输出
        'fault_window_start_time': format_time(fault_win_start),
        'fault_window_end_time': format_time(fault_win_end),
        'fault_window_start_timestamp': int(fault_win_start.timestamp()),
        'fault_window_end_timestamp': int(fault_win_end.timestamp()),
        # 正常窗口格式化输出（对应另一天的相同时间段）
        'normal_window_start_time': format_time(normal_win_start),
        'normal_window_end_time': format_time(normal_win_end),
        'normal_window_start_timestamp': int(normal_win_start.timestamp()),
        'normal_window_end_timestamp': int(normal_win_end.timestamp()),
        # 修复：使用计算好的中点
        'matched_normal_center': format_time(normal_window_mid)
    }
    results.append(new_row)

# ----------------------------
# 保存结果
# ----------------------------
result_df = pd.DataFrame(results)
# 保留原始列 + 新增列，确保格式与原输出一致
original_columns = list(df.columns)
new_columns = [
    'fault_window_start_time', 'fault_window_end_time',
    'fault_window_start_timestamp', 'fault_window_end_timestamp',
    'normal_window_start_time', 'normal_window_end_time',
    'normal_window_start_timestamp', 'normal_window_end_timestamp',
    'matched_normal_center'
]
final_columns = original_columns + new_columns
result_df = result_df[final_columns]
result_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ 已成功保存结果到: {OUTPUT_CSV}")
print("📌 逻辑修改完成：正常窗口直接取相邻日期的对应时间段，已修复Timestamp相加错误")
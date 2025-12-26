import pandas as pd
from datetime import timedelta

# ----------------------------
# 配置
# ----------------------------
INPUT_CSV = "/root/shared-nvme/work/agent/OpenRCA/dataset/Market/cloudbed-1/record.csv"
OUTPUT_CSV = "./record_with_normal_intervals.csv"

# ----------------------------
# 步骤1：加载数据并设置为 UTC+8 (Asia/Shanghai)
# ----------------------------
df = pd.read_csv(INPUT_CSV)

# 解析 datetime 并显式指定为 Asia/Shanghai（UTC+8）
df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('Asia/Shanghai')

# 提取日期和时间（时区感知）
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.time

# 按时间排序
df = df.sort_values('datetime').reset_index(drop=True)

# 获取唯一日期（应为两天）
unique_dates = sorted(df['date'].unique())
if len(unique_dates) != 2:
    raise ValueError(f"期望恰好两天的数据，但检测到 {len(unique_dates)} 天: {unique_dates}")
date1, date2 = unique_dates

print(f"检测到两天数据：{date1} 和 {date2}")

# ----------------------------
# 辅助函数
# ----------------------------
def get_all_half_hour_slots():
    """返回一天中 48 个半小时起始时间（time 对象）"""
    return pd.date_range("00:00", periods=48, freq="30T").time.tolist()

def assign_to_slot(dt):
    """将带时区的 datetime 向下取整到最近的半小时（保留时区）"""
    return dt.floor('30T')

def find_nearest_normal_slot(target_slot, normal_slots):
    """在 normal_slots 中找离 target_slot 最近的窗口（按秒差）"""
    if not normal_slots:
        return None
    target_ts = target_slot.timestamp()
    min_diff = float('inf')
    best_slot = None
    for slot in normal_slots:
        diff = abs(slot.timestamp() - target_ts)
        if diff < min_diff:
            min_diff = diff
            best_slot = slot
    return best_slot

# ----------------------------
# 步骤2：构建每天的无故障半小时窗口（带时区）
# ----------------------------
df['fault_slot_start'] = df['datetime'].apply(assign_to_slot)
fault_slots_set = set(df['fault_slot_start'])

# 构建全天所有半小时窗口（带 Asia/Shanghai 时区）
all_slots_day1 = [
    pd.Timestamp.combine(date1, t).tz_localize('Asia/Shanghai')
    for t in get_all_half_hour_slots()
]
all_slots_day2 = [
    pd.Timestamp.combine(date2, t).tz_localize('Asia/Shanghai')
    for t in get_all_half_hour_slots()
]

# 故障窗口集合
faulty_slots_day1 = {s for s in all_slots_day1 if s in fault_slots_set}
faulty_slots_day2 = {s for s in all_slots_day2 if s in fault_slots_set}

# 无故障窗口（已排序）
normal_slots_day1 = sorted(set(all_slots_day1) - faulty_slots_day1)
normal_slots_day2 = sorted(set(all_slots_day2) - faulty_slots_day2)

normal_set_day1 = set(normal_slots_day1)
normal_set_day2 = set(normal_slots_day2)

print(f"第1天无故障窗口数: {len(normal_slots_day1)}")
print(f"第2天无故障窗口数: {len(normal_slots_day2)}")

# ----------------------------
# 步骤3：为每个故障匹配无故障窗口
# ----------------------------
results = []

for _, row in df.iterrows():
    slot_start = row['fault_slot_start']
    current_date = slot_start.date()

    # 确定“另一天”及其无故障窗口
    if current_date == date1:
        other_date = date2
        other_normal_slots = normal_slots_day2
        other_normal_set = normal_set_day2
    else:
        other_date = date1
        other_normal_slots = normal_slots_day1
        other_normal_set = normal_set_day1

    # 构造另一天同一时间段（必须带时区！）
    same_time_slot = pd.Timestamp.combine(other_date, slot_start.time()).tz_localize('Asia/Shanghai')

    # 原则1：优先选另一天同一时段（若无故障）
    if same_time_slot in other_normal_set:
        normal_start = same_time_slot
    else:
        # 原则2：找最近的无故障窗口
        normal_start = find_nearest_normal_slot(slot_start, other_normal_slots)
        if normal_start is None:
            raise RuntimeError(f"无法为故障 {row['datetime']} 找到无故障窗口！")

    normal_end = normal_start + timedelta(minutes=30)

    # 格式化时间为字符串（UTC+8）
    def format_time(dt):
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    # 构建新行
    new_row = {
        **row.to_dict(),
        'normal_start_timestamp': int(normal_start.timestamp()),
        'normal_start_time': format_time(normal_start),
        'normal_end_timestamp': int(normal_end.timestamp()),
        'normal_end_time': format_time(normal_end)
    }
    results.append(new_row)

# ----------------------------
# 步骤4：保存结果
# ----------------------------
result_df = pd.DataFrame(results)

# 列顺序：原始列 + 新增4列
original_columns = list(df.columns)
new_columns = ['normal_start_timestamp', 'normal_start_time', 'normal_end_timestamp', 'normal_end_time']
final_columns = original_columns + new_columns

result_df = result_df[final_columns]
result_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ 已成功保存结果到: {OUTPUT_CSV}")
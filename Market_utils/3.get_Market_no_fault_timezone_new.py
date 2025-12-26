import pandas as pd
from datetime import timedelta

# ----------------------------
# 配置
# ----------------------------
INPUT_CSV = "/root/shared-nvme/work/agent/OpenRCA/dataset/Market/cloudbed-2/record.csv"
OUTPUT_CSV = "./cloudbed-2_record_with_normal_intervals_new.csv"

# ----------------------------
# 加载数据
# ----------------------------
df = pd.read_csv(INPUT_CSV)
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
def get_all_half_hour_slots():
    return pd.date_range("00:00", periods=48, freq="30T").time.tolist()

def assign_to_slot(dt):
    return dt.floor('30T')

def find_nearest_center(target_center, candidate_centers):
    if not candidate_centers:
        return None
    target_ts = target_center.timestamp()
    best = min(candidate_centers, key=lambda c: abs(c.timestamp() - target_ts))
    return best

def format_time(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# ----------------------------
# 步骤1: 标记故障 slot
# ----------------------------
df['fault_slot_start'] = df['datetime'].apply(assign_to_slot)
fault_slots_set = set(df['fault_slot_start'])

# 构建全天所有 slot（带时区）
all_slots_day1 = [
    pd.Timestamp.combine(date1, t).tz_localize('Asia/Shanghai')
    for t in get_all_half_hour_slots()
]
all_slots_day2 = [
    pd.Timestamp.combine(date2, t).tz_localize('Asia/Shanghai')
    for t in get_all_half_hour_slots()
]

# ----------------------------
# 步骤2: 为每天构建“有效60分钟窗口中心” M
# 条件：M 和 M-30min 都是无故障 slot
# ----------------------------
def build_valid_centers(all_slots, fault_set):
    valid_centers = []
    slot_set = set(all_slots)
    for slot in all_slots:
        prev_slot = slot - timedelta(minutes=30)
        # 只有从 00:30 开始才有前一个 slot
        if prev_slot in slot_set:
            if prev_slot not in fault_set and slot not in fault_set:
                # M = slot （即后一个 slot 的起点，作为60分钟窗口的中心）
                valid_centers.append(slot)
    return sorted(valid_centers)

valid_centers_day1 = build_valid_centers(all_slots_day1, fault_slots_set)
valid_centers_day2 = build_valid_centers(all_slots_day2, fault_slots_set)

valid_center_set_day1 = set(valid_centers_day1)
valid_center_set_day2 = set(valid_centers_day2)

print(f"第1天有效60分钟正常窗口中心数: {len(valid_centers_day1)}")
print(f"第2天有效60分钟正常窗口中心数: {len(valid_centers_day2)}")

# ----------------------------
# 步骤3: 为每个故障匹配正常窗口
# ----------------------------
results = []

for _, row in df.iterrows():
    fault_dt = row['datetime']
    T = row['fault_slot_start']  # 故障 slot 起点
    current_date = T.date()

    # 故障窗口：[T - 30min, T + 30min)
    fault_win_start = T - timedelta(minutes=30)
    fault_win_end = T + timedelta(minutes=30)

    # 选择另一天的候选中心
    if current_date == date1:
        other_date = date2
        other_centers = valid_centers_day2
        other_center_set = valid_center_set_day2
    else:
        other_date = date1
        other_centers = valid_centers_day1
        other_center_set = valid_center_set_day1

    # 构造 same-time center on other day
    same_time_center = pd.Timestamp.combine(other_date, T.time()).tz_localize('Asia/Shanghai')

    # 候选 M 必须是 valid center（即本身和前一个 slot 都干净）
    if same_time_center in other_center_set:
        M = same_time_center
    else:
        M = find_nearest_center(T, other_centers)
        if M is None:
            raise RuntimeError(f"无法为故障 {fault_dt} 找到有效60分钟正常窗口！")

    # 正常窗口：[M - 30min, M + 30min)
    normal_win_start = M - timedelta(minutes=30)
    normal_win_end = M + timedelta(minutes=30)

    # 构建结果行
    new_row = {
        **row.to_dict(),
        # 故障窗口（60分钟，以故障slot起点为中心）
        'fault_window_start_time': format_time(fault_win_start),
        'fault_window_end_time': format_time(fault_win_end),
        'fault_window_start_timestamp': int(fault_win_start.timestamp()),
        'fault_window_end_timestamp': int(fault_win_end.timestamp()),
        # 正常窗口（60分钟，以匹配中心M为中心）
        'normal_window_start_time': format_time(normal_win_start),
        'normal_window_end_time': format_time(normal_win_end),
        'normal_window_start_timestamp': int(normal_win_start.timestamp()),
        'normal_window_end_timestamp': int(normal_win_end.timestamp()),
        # 匹配的中心时间（便于调试）
        'matched_normal_center': format_time(M)
    }
    results.append(new_row)

# ----------------------------
# 保存结果
# ----------------------------
result_df = pd.DataFrame(results)
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
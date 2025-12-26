import pandas as pd
from collections import defaultdict

# 文件路径
file_path = "/root/shared-nvme/work/agent/OpenRCA/dataset/Market/cloudbed-1/record.csv"

# 读取数据
df = pd.read_csv(file_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.time

# 获取所有唯一日期（应为两天）
unique_dates = sorted(df['date'].unique())
if len(unique_dates) < 1:
    raise ValueError("未检测到有效日期数据")
print(f"cloudbed-1 检测到 {len(unique_dates)} 天的数据: {unique_dates}")

# 自动生成一天中 48 个半小时时间段：00:00, 00:30, ..., 23:30
half_hour_slots = pd.date_range("00:00", periods=48, freq="30T").time.tolist()

# 初始化嵌套字典：slot_time -> date -> list of rows
slot_dict = defaultdict(lambda: defaultdict(list))

# 将每条故障分配到对应“当天”的半小时时间段（如 09:37 → 09:30）
for _, row in df.iterrows():
    dt = row['datetime']
    slot_start = dt.floor('30T').time()  # 向下取整到最近的30分钟边界
    slot_dict[slot_start][row['date']].append(row)

# 输出对齐结果
print("\n🔍 故障按每日半小时时段对齐对比（时间段格式：HH:MM - HH:MM）\n")
print("=" * 100)

for slot in half_hour_slots:
    # 计算时间段结束时间（用于显示）
    slot_dt = pd.Timestamp.combine(pd.Timestamp.today().date(), slot)
    end_dt = slot_dt + pd.Timedelta(minutes=30)
    end_str = end_dt.time().strftime("%H:%M")
    interval_label = f"{slot.strftime('%H:%M')} - {end_str}"

    # 获取这两天在该时段的故障
    day1_faults = slot_dict[slot].get(unique_dates[0], [])
    day2_faults = slot_dict[slot].get(unique_dates[1], []) if len(unique_dates) > 1 else []

    # 仅当至少有一天有故障时才输出
    if day1_faults or day2_faults:
        print(f"\n🕒 {interval_label}")
        print("-" * 90)

        # 第一天
        print(f"📅 {unique_dates[0]} ({len(day1_faults)} fault(s)):")
        if day1_faults:
            for r in day1_faults:
                print(f"    • {r['component']} | {r['reason']}")
        else:
            print("    (无故障)")

        # 第二天（如果存在）
        if len(unique_dates) > 1:
            print(f"📅 {unique_dates[1]} ({len(day2_faults)} fault(s)):")
            if day2_faults:
                for r in day2_faults:
                    print(f"    • {r['component']} | {r['reason']}")
            else:
                print("    (无故障)")

print("\n✅ 对齐分析完成。")
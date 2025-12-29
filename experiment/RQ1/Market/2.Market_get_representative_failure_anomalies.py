import csv
import os
from datetime import datetime, timezone, timedelta

# 输入和输出文件路径
input_file = "/root/shared-nvme/work/agent/OpenRCA/dataset/Market/cloudbed-1/record.csv"
output_file = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/Market/groundtruth.csv"

def get_half_hour_span(unix_ts):
    """根据 Unix 时间戳返回 'HHMM_HHMM' 格式的半小时区间字符串"""
    dt = datetime.fromtimestamp(unix_ts)
    hour = dt.hour
    minute = dt.minute

    # 确定起始分钟：0~29 → 00；30~59 → 30
    if minute < 30:
        start_min = 0
    else:
        start_min = 30

    start_dt = dt.replace(minute=start_min, second=0, microsecond=0)
    end_dt = start_dt + timedelta(minutes=30)

    start_str = start_dt.strftime("%H%M")
    end_str = end_dt.strftime("%H%M")
    return f"{start_str}_{end_str}"

def extract_datestr(datetime_str):
    """从 '2022-03-20 09:09:06' 提取 '2022_03_20'"""
    try:
        date_part = datetime_str.split()[0]  # "2022-03-20"
        y, m, d = date_part.split('-')
        return f"{y}_{m}_{d}"
    except Exception:
        return "UNKNOWN_DATE"

# 用于记录已见过的 (level, reason) 组合
seen_combinations = set()
output_rows = []

# 表头
new_header = ["level", "component", "timestamp", "datetime", "reason", "span", "datestr"]

with open(input_file, mode='r', encoding='utf-8') as f_in:
    reader = csv.reader(f_in)
    header = next(reader)  # 跳过原始表头

    output_rows.append(new_header)

    for row in reader:
        if len(row) < 5:
            continue
        timestamp_str, level, component, reason, datetime_str = row[:5]
        try:
            timestamp = int(timestamp_str)
        except ValueError:
            continue  # 跳过无效时间戳

        key = (level.strip(), reason.strip())
        if key not in seen_combinations:
            seen_combinations.add(key)

            span = get_half_hour_span(timestamp)
            datestr = extract_datestr(datetime_str)

            new_row = [
                level.strip(),
                component.strip(),
                str(timestamp),
                datetime_str.strip(),
                reason.strip(),
                span,
                datestr
            ]
            output_rows.append(new_row)

# 写入新 CSV
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, mode='w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerows(output_rows)

print(f"✅ 已选出 {len(output_rows) - 1} 种唯一的 (level, reason) 故障组合。")
print(f"📄 结果已保存至: {output_file}")
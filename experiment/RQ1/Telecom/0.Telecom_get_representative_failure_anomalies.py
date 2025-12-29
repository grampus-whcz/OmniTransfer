import csv
import os
from datetime import datetime, timedelta

# 输入和输出文件路径
input_file = "/root/shared-nvme/work/agent/OpenRCA/dataset/Telecom/record.csv"
output_file = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/Telecom/groundtruth.csv"

def get_half_hour_span(unix_ts):
    dt = datetime.fromtimestamp(unix_ts)
    start_min = 0 if dt.minute < 30 else 30
    start_dt = dt.replace(minute=start_min, second=0, microsecond=0)
    end_dt = start_dt + timedelta(minutes=30)
    return f"{start_dt.strftime('%H%M')}_{end_dt.strftime('%H%M')}"

def extract_datestr(datetime_str):
    try:
        date_part = datetime_str.split()[0]  # "2020-04-11"
        y, m, d = date_part.split('-')
        return f"{y}_{m}_{d}"
    except Exception:
        return "UNKNOWN_DATE"

seen_combinations = set()
output_rows = []
new_header = ["level", "component", "timestamp", "datetime", "reason", "span", "datestr"]
output_rows.append(new_header)

# 读取文件
with open(input_file, mode='r', encoding='utf-8') as f_in:
    reader = csv.reader(f_in)
    header = next(reader)  # 跳过: ['level','reason','component','timestamp','datetime']

    for row in reader:
        if len(row) < 5:
            continue

        # ✅ 正确按 Telecom 顺序解析
        level = row[0].strip()
        reason = row[1].strip()
        component = row[2].strip()
        timestamp_str = row[3].strip()
        datetime_str = row[4].strip()

        # 转换时间戳
        try:
            timestamp = int(timestamp_str)
        except ValueError:
            print(f"⚠️ 跳过无效时间戳行: {row}")
            continue

        key = (level, reason)
        if key not in seen_combinations:
            seen_combinations.add(key)

            span = get_half_hour_span(timestamp)
            datestr = extract_datestr(datetime_str)

            output_rows.append([
                level,
                component,
                str(timestamp),
                datetime_str,
                reason,
                span,
                datestr
            ])

# 写入输出
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, mode='w', newline='', encoding='utf-8') as f_out:
    writer = csv.writer(f_out)
    writer.writerows(output_rows)

print(f"✅ 已选出 {len(output_rows) - 1} 种唯一的 (level, reason) 故障组合。")
print(f"📄 结果已保存至: {output_file}")
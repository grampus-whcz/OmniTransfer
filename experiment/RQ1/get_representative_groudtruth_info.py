import csv
import os
import re
import ast

input_log = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/get_representative_failure_anomalies.log"
output_md = "groudtruth.md"
output_csv = "groundtruth.csv"

csv_header = ["level", "component", "timestamp", "datetime", "reason", "span", "datestr"]

# 读取日志文件
with open(input_log, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# 初始化 CSV（写表头）
csv_exists = os.path.exists(output_csv)
if not csv_exists:
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_header)

# 正则用于解析下一行的时间窗口信息
time_window_pattern = re.compile(r"^\s*→ Time window: ([\d_]+) on ([\d_]+)")

# 写入 Markdown 片段并解析 CSV 数据
with open(output_md, 'w', encoding='utf-8') as md_file:
    for i, line in enumerate(lines):
        if "Original line in groudtruth: " in line:
            # --- 1. 解析当前行的 groundtruth 数据 ---
            try:
                prefix = "Original line in groudtruth: "
                data_str = line.strip()[len(prefix):]
                data_list = ast.literal_eval(data_str)

                if len(data_list) != 5:
                    print(f"⚠️ 跳过格式异常行 {i+1}（字段数≠5）")
                    continue

                level, component, ts_raw, dt_str, reason = data_list
                try:
                    timestamp_clean = str(int(float(ts_raw)))
                except:
                    timestamp_clean = str(ts_raw)

            except Exception as e:
                print(f"❌ 当前行解析失败 (行 {i+1}): {e}")
                continue

            # --- 2. 尝试解析下一行的时间窗口信息 ---
            span = ""
            datestr = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                match = time_window_pattern.match(next_line)
                if match:
                    span = match.group(1)      # e.g., "0330_0400"
                    datestr = match.group(2)   # e.g., "2021_03_07"

            # --- 3. 写入 CSV（追加）---
            with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([level, component, timestamp_clean, dt_str, reason, span, datestr])

            # --- 4. 写入上下文到 .md（5行）---
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            snippet = lines[start:end]
            for l in snippet:
                md_file.write(l.rstrip('\n') + '\n')
            md_file.write('\n')  # 空行分隔

print(f"✅ 处理完成！\n - Markdown: {output_md}\n - CSV: {output_csv}")
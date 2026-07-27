import re
import argparse
import random
from datetime import datetime, timedelta

# random.seed(42) # 开启此行可固定随机结果，方便复现

def parse_log_timestamp(line: str):
    """匹配行首时间戳 YYYY-MM-DD HH:MM:SS.fff"""
    pat = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})')
    m = pat.match(line)
    if not m:
        return None
    ts_str = m.group(1)
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt, m.group(1)


def random_stretch_timestamps(input_path: str, output_path: str, new_start_str: str):
    """
    日志时间随机扰动：
    - 首条带时间行强制设为 new_start_str
    - 每一段相邻时间间隔 Δt，随机取 [0.9Δt, 1.1Δt]
    - 保证时序不变、满足 ±10% 间隔约束
    """
    new_start = datetime.strptime(new_start_str, "%Y-%m-%d %H:%M:%S.%f")

    all_lines = []
    ts_records = []  # [(line_index, original_datetime, original_text)]

    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            all_lines.append(line)
            res = parse_log_timestamp(line)
            if res is not None:
                dt_old, text_old = res
                ts_records.append((idx, dt_old, text_old))

    if not ts_records:
        print("未找到任何行首时间戳，直接复制原文件")
        with open(output_path, "w", encoding="utf-8") as fw:
            fw.writelines(all_lines)
        return

    # 提取原始时间序列
    orig_times = [item[1] for item in ts_records]
    new_times = [new_start]

    for i in range(1, len(orig_times)):
        delta_original = orig_times[i] - orig_times[i-1]
        delta_sec = delta_original.total_seconds()

        # 随机采样 0.9 ~ 1.1 倍原间隔
        scale = random.uniform(0.9, 1.1)
        delta_new_sec = delta_sec * scale
        delta_new = timedelta(seconds=delta_new_sec)

        t_next = new_times[-1] + delta_new
        new_times.append(t_next)

    # 逐行替换时间字符串
    for i, (line_idx, _, old_text) in enumerate(ts_records):
        t_new = new_times[i]
        new_text = t_new.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        old_line = all_lines[line_idx]
        # 修复：去掉count=关键字，使用位置传参
        new_line = old_line.replace(old_text, new_text, 1)
        all_lines[line_idx] = new_line

    # 写出文件
    with open(output_path, "w", encoding="utf-8") as fw:
        fw.writelines(all_lines)

    print(f"处理完成！")
    print(f"目标起始时间：{new_start_str}")
    print(f"共处理 {len(ts_records)} 条带时间日志行")
    print(f"输出路径：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="日志时间扰动工具：相邻间隔随机 ±10%，首时间固定，时序不变"
    )
    parser.add_argument("--input", required=True, help="原始日志路径")
    parser.add_argument("--output", required=True, help="输出新日志路径")
    parser.add_argument("--new-start", required=True, help="第一条时间戳，格式 2026-07-21 10:00:00.000")
    args = parser.parse_args()
    random_stretch_timestamps(args.input, args.output, args.new_start)
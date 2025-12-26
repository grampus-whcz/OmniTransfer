import csv
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta, date
import subprocess
import sys
import os
import time

def extract_non_digit_prefix(component):
    match = re.match(r'^[a-zA-Z]+', component)
    return match.group(0) if match else component

def get_half_hour_window(timestamp_sec):
    dt = datetime.fromtimestamp(timestamp_sec, tz=timezone(timedelta(hours=8)))  # GMT+8
    minute = dt.minute
    window_start_minute = 0 if minute < 30 else 30
    window_start = dt.replace(minute=window_start_minute, second=0, microsecond=0)
    start_ts = int(window_start.timestamp())
    end_ts = start_ts + 1800
    start_str = window_start.strftime("%H%M")
    end_str = (window_start + timedelta(minutes=30)).strftime("%H%M")
    output_suffix = f"{start_str}_{end_str}"
    date_online = window_start.strftime("%Y_%m_%d")
    return start_ts, end_ts, date_online, output_suffix

def main():
    file_path = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/groundtruth.csv"
    script_dir = "/root/shared-nvme/work/timeSeries/OmniTransfer_new"

    pipelines = [
        ("Bank_metric_container", "run_pipline_Bank_metric_container.py"),
        # ("Bank_metric_app",      "run_pipline_Bank_metric_app.py"),
        # ("Bank_trace",           "run_pipline_Bank_trace.py"),
        # ("Bank_log",             "run_pipline_Bank_log.py"),
    ]

    for name, script in pipelines:
        full_path = os.path.join(script_dir, script)
        if not os.path.isfile(full_path):
            print(f"[FATAL] Script not found: {full_path}", flush=True)
            sys.exit(1)

    groups = defaultdict(list)
    all_lines = []

    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        all_lines.append(header)

        for i, row in enumerate(reader, start=2):
            all_lines.append(row)
            if len(row) < 5:
                continue
            level, component, timestamp_str, datetime_str, reason = row[:5]
            try:
                timestamp = float(timestamp_str)
            except ValueError:
                continue
            prefix = extract_non_digit_prefix(component)
            key = (reason.strip(), prefix.strip())
            groups[key].append((i, timestamp, row))

    representative_entries = []
    for key, entries in groups.items():
        line_num, timestamp, row = entries[0]
        representative_entries.append((key, line_num, timestamp, row))

    representative_entries.sort(key=lambda x: (x[0][0], x[0][1]))
    today_str = date.today().strftime("%m%d")

    print(f"[INFO] Total representative entries: {len(representative_entries)}", flush=True)
    print(f"[INFO] Output folder name: {today_str}", flush=True)
    print(f"[INFO] Script directory: {script_dir}", flush=True)
    print("=" * 90, flush=True)

    success_count = 0
    total_commands = 0
    total_elapsed = 0.0

    overall_start = time.time()

    # index = 0
    for idx, ((reason, comp_prefix), line_num, timestamp, row) in enumerate(representative_entries, 1):
        # if index > 0:
        #     break
        try:
            start_ts, end_ts, date_online, output_suffix = get_half_hour_window(timestamp)
        except Exception as e:
            print(f"[ERROR] Skip entry {idx} (line {line_num}): failed to parse timestamp {timestamp}: {e}", flush=True)
            continue

        print(f"\n[ENTRY {idx}] Reason: '{reason}', Component prefix: '{comp_prefix}' (line {line_num})", flush=True)
        print(f"\nOriginal line in groudtruth: {row}", flush=True)
        print(f"  → Time window: {output_suffix} on {date_online}", flush=True)

        entry_start = time.time()
        entry_cmd_count = 0

        for name, script in pipelines:
            total_commands += 1
            entry_cmd_count += 1
            script_path = os.path.join(script_dir, script)
            cmd = [
                sys.executable, script_path,
                "--date_offline", "2021_03_05",
                "--date_online", date_online,
                "--start_ts", str(start_ts),
                "--end_ts", str(end_ts),
                "--method", "TranAD",
                "--output_folder_name", today_str,
                "--output_suffix", output_suffix
            ]

            print(f"\n[CMD {total_commands}] [{name}] Start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            print("Command:", " ".join(cmd), flush=True)

            cmd_start = time.time()
            try:
                result = subprocess.run(cmd, check=True, text=True, capture_output=False)
                cmd_end = time.time()
                cmd_elapsed = cmd_end - cmd_start
                print(f"[SUCCESS] {name} completed in {cmd_elapsed:.2f} seconds.", flush=True)
                success_count += 1
            except subprocess.CalledProcessError as e:
                cmd_end = time.time()
                cmd_elapsed = cmd_end - cmd_start
                print(f"[FAILED] {name} exited with code {e.returncode} after {cmd_elapsed:.2f} seconds. Continue to next.", flush=True)
            except Exception as e:
                cmd_end = time.time()
                cmd_elapsed = cmd_end - cmd_start
                print(f"[EXCEPTION] Failed to run {name} after {cmd_elapsed:.2f} seconds: {e}", flush=True)

        entry_end = time.time()
        entry_elapsed = entry_end - entry_start
        total_elapsed += entry_elapsed
        print(f"\n[ENTRY {idx} SUMMARY] Total time for this fault case: {entry_elapsed:.2f} seconds ({entry_cmd_count} commands)", flush=True)
        
        # index = index +1

    overall_end = time.time()
    overall_elapsed = overall_end - overall_start

    print("\n" + "=" * 90, flush=True)
    print(f"[FINAL SUMMARY]", flush=True)
    print(f"Total representative fault cases: {len(representative_entries)}", flush=True)
    print(f"Total commands executed: {total_commands}", flush=True)
    print(f"Successful commands: {success_count}", flush=True)
    print(f"Failed commands: {total_commands - success_count}", flush=True)
    print(f"Total execution time (all commands): {total_elapsed:.2f} seconds", flush=True)
    print(f"Overall wall-clock time (including overhead): {overall_elapsed:.2f} seconds", flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    main()
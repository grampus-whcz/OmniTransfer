import csv
import subprocess
import os

# 配置
GROUNDTRUTH_FILE = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/Market/groundtruth.csv"
SCRIPT_NAME = "0.Market_get_representative_anomaly_report_new.py"
OUTPUT_DIR = "./report"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取 CSV 并逐行执行命令
with open(GROUNDTRUTH_FILE, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        datestr = row['datestr']          # e.g., "2021_03_04"
        window = row['span']              # e.g., "2200_2230"
        ts = row['timestamp']             # e.g., "1614867900"

        output_file = os.path.join(OUTPUT_DIR, f"{i}_report_{datestr}_{window}_{ts}.txt")

        cmd = [
            "python", SCRIPT_NAME,
            "--date", datestr,
            "--window", window,
            "--ts", ts,
            "--output", output_file
        ]

        print(f"[{i}] Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            print(f"✅ Success: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed for ts={ts}, window={window}")
            print("stderr:", e.stderr)
        except FileNotFoundError:
            print(f"❌ Script not found: {SCRIPT_NAME}")
            break
import os
import subprocess
import re
from pathlib import Path

# 配置路径和参数
BASE_DIR = Path("/root/shared-nvme/work/timeSeries/OmniTransfer_new/1204")
PYTHON_EXEC = "/root/shared-nvme/.conda/envs/faiss-env/bin/python"
SCRIPT_PATH = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Bank_cluster_window_analyze_anomalies.py"  # ✅ 修正为绝对路径
FIXED_ARGS = ["--eps", "60", "--min_samples", "2", "--output_folder_name", "1204"]

# 正则表达式匹配文件名
pattern = re.compile(r'Bank_trace_anomaly_report_(\d{4}_\d{2}_\d{2})_(\d{4}_\d{4})\.txt')

def main():
    if not BASE_DIR.exists():
        print(f"目录不存在: {BASE_DIR}")
        return

    files = list(BASE_DIR.glob("Bank_trace_anomaly_report_*.txt"))
    if not files:
        print("未找到任何匹配的文件。")
        return

    print(f"共找到 {len(files)} 个文件，开始处理...")

    for file_path in sorted(files):
        filename = file_path.name
        match = pattern.match(filename)
        if not match:
            print(f"跳过无法解析的文件名: {filename}")
            continue

        date_online = match.group(1)      # e.g., 2021_03_25
        output_suffix = match.group(2)    # e.g., 2300_2330

        cmd = [
            PYTHON_EXEC,
            SCRIPT_PATH,
            "--date_online", date_online,
            "--output_suffix", output_suffix,
        ] + FIXED_ARGS

        print(f"\n[运行] {' '.join(cmd)}")

        try:
            # 注意：cwd 设为 BASE_DIR 或脚本所在目录？这里保持 cwd=BASE_DIR（因为输入文件在那里）
            # 如果脚本内部依赖相对路径读取数据，cwd 应为 BASE_DIR
            result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ 成功: {filename}")
            else:
                print(f"❌ 失败: {filename}")
                print("错误信息:", result.stderr)
        except Exception as e:
            print(f"⚠️ 异常: {filename} | 错误: {e}")

    print("\n所有任务完成。")

if __name__ == "__main__":
    main()
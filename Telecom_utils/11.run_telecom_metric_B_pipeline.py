#!/usr/bin/env python3
import subprocess
import os
import argparse

def run_cmd(cmd, cwd=None):
    """运行 shell 命令，打印并等待完成"""
    print(f"\n[CMD] {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, text=True)
    return result

def main(args):
    base_dir = "/root/shared-nvme/work/timeSeries/OmniTransfer_new"
    
    # 构造 Telecom Metric B 原始文件路径（注意：这些是原始单实体文件）
    offline_src = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Telecom/metric/entity_B_{args.date_offline}_{args.bucket_sec}s.npy"
    )
    online_src = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Telecom/metric/entity_B_{args.date_online}_{args.bucket_sec}s.npy"
    )
    metric_meta_file = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Telecom/metric/metadata_B_{args.date_online}_{args.bucket_sec}s.json"
    )

    # 目标路径（OmniTransfer 输入）
    dst_offline = os.path.join(base_dir, f"dataset/{args.data_dir}/offline_data.npy")
    dst_online = os.path.join(base_dir, f"dataset/{args.data_dir}/online_data.npy")

    # Step 1 & 2: 复制原始单实体数据（后续由预处理脚本或手动扩展为12份）
    # ⚠️ 重要：假设你已经有一个脚本或机制将 (1, T, F) 扩展为 (12, T, F)
    # 这里我们直接复制原始文件，然后依赖一个“扩展脚本”来生成符合 OmniTransfer 的输入
    run_cmd(f"cp {offline_src} {dst_offline}")
    run_cmd(f"cp {online_src} {dst_online}")

    # Step 2.5: 扩展单实体为12副本（必须在聚类前完成！）
    expand_script = os.path.join(base_dir, "Telecom_utils/6.supply_0d_for_single_entity_data.py")
    if not os.path.exists(expand_script):
        raise FileNotFoundError(
            f"Required expansion script not found: {expand_script}\n"
            "This script should load offline_data.npy/online_data.npy and save back as (12, T, F)."
        )
    run_cmd(f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {expand_script}", cwd=base_dir)

    # Step 3: 聚类（现在数据已是 (12, T, F)，可被 cluster_Bank_v3.py 处理）
    cluster_script = os.path.join(base_dir, "code/cluster/cluster_Telecom.py")
    run_cmd(f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {cluster_script}", cwd=base_dir)

    # Step 4: 运行异常检测方法（如 TranAD）
    data_dir = os.path.join(base_dir, f"code/test_dataset/{args.data_dir}")
    run_cmd(f"bash run_new.sh {args.method} {args.output_folder_name} {data_dir}", cwd=base_dir)

    # Step 5: 分析 Telecom Metric B 异常（仅解析第0个实体）
    analyze_script = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Telecom_utils/8.Telecom_metric_B_analyze_anomalies.py"
    output_file = f"{base_dir}/{args.output_folder_name}/Telecom_metric_B_anomalies_{args.date_online}_{args.output_suffix}.npy"

    cmd_analyze = (
        f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {analyze_script} "
        f"--meta {metric_meta_file} "
        f"--pred_dir {args.output_folder_name} "
        f"--start {args.start_ts} "
        f"--end {args.end_ts} "
        f"--output {output_file} "
        f"--anomaly_report {args.date_online}_{args.output_suffix}"
    )
    run_cmd(cmd_analyze, cwd=base_dir)
    
    print(f"\n✅ Telecom Metric B Pipeline completed. Anomaly results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full OmniTransfer anomaly detection pipeline for Telecom Metric B (single-entity case).")

    parser.add_argument("--date_offline", type=str, required=True,
                        help="Date string for offline (normal) data, e.g., '2020_05_31'")
    parser.add_argument("--date_online", type=str, required=True,
                        help="Date string for online (faulty) data, e.g., '2020_04_11'")
    parser.add_argument("--bucket_sec", type=int, default=60,
                        help="Bucket granularity in seconds (e.g., 60, 120, 300). Default: 60")
    parser.add_argument("--start_ts", type=int, required=True,
                        help="Start timestamp (Unix epoch), e.g., 1586534400")
    parser.add_argument("--end_ts", type=int, required=True,
                        help="End timestamp (Unix epoch), e.g., 1586556000")
    parser.add_argument("--method", type=str, default="TranAD",
                        help="Anomaly detection method name used in run.sh")
    parser.add_argument("--output_folder_name", type=str, default="1120",
                        help="Output folder name passed to run.sh (e.g., experiment ID)")
    parser.add_argument("--output_suffix", type=str, default="result",
                        help="Suffix for output anomalies file")
    parser.add_argument("--data_dir", type=str, default="data3",
                        help="Directory name for test data, e.g., 'data1', 'data2', 'data3'")

    args = parser.parse_args()
    try:
        main(args)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed: {e.cmd}")
        print(f"Exit code: {e.returncode}")
        exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        exit(1)
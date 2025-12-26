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
    
    # 构造文件路径
    offline_src = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Bank/trace/{args.date_offline}_trace_edge_bucket_60.npy"
    )
    online_src = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Bank/trace/{args.date_online}_trace_edge_bucket_60.npy"
    )
    trace_meta_file = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Bank/trace/{args.date_online}_trace_edge_bucket_60.meta.json"  
    )

    dst_offline = os.path.join(base_dir, "dataset/data1/offline_data.npy")
    dst_online = os.path.join(base_dir, "dataset/data1/online_data.npy")

    # Step 1 & 2: 复制数据文件
    run_cmd(f"cp {offline_src} {dst_offline}")
    run_cmd(f"cp {online_src} {dst_online}")

    # Step 3: 聚类
    cluster_script = os.path.join(base_dir, "code/cluster/cluster_Bank_v3.py")
    run_cmd(f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {cluster_script}", cwd=base_dir)

    # Step 4: 运行异常检测方法（如 TranAD）
    run_cmd(f"bash run.sh {args.method} {args.output_folder_name}", cwd=base_dir)
   
    # Step 7: trace 分析异常
    pred_dir = os.path.join(
        base_dir,
        f"{args.output_folder_name}/{args.method}/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"
    )
    analyze_script = os.path.join(base_dir, "Bank_utils/Bank_trace_analyze_anomalies.py")
    output_file = f"{base_dir}/{args.output_folder_name}/Bank_trace_anomalies_{args.date_online}_{args.output_suffix}.npy"

    cmd_analyze = (
        f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {analyze_script} "
        f"--meta {trace_meta_file} "
        f"--pred_dir {args.output_folder_name} "
        f"--start {args.start_ts} "
        f"--end {args.end_ts} "
        f"--output {output_file} "
        f"--anomaly_report {args.date_online}_{args.output_suffix}"
    )
    run_cmd(cmd_analyze, cwd=base_dir)

    print(f"\n✅ Pipeline completed. Anomaly results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full OmniTransfer anomaly detection pipeline for Bank dataset.")

    parser.add_argument("--date_offline", type=str, required=True,
                        help="Date string for offline (normal) data, e.g., '2021_03_05'")
    parser.add_argument("--date_online", type=str, required=True,
                        help="Date string for online (faulty) data, e.g., '2021_03_06'")
    parser.add_argument("--start_ts", type=int, required=True,
                        help="Start timestamp (Unix epoch), e.g., 1614972600")
    parser.add_argument("--end_ts", type=int, required=True,
                        help="End timestamp (Unix epoch), e.g., 1614974400")
    parser.add_argument("--method", type=str, default="TranAD",
                        help="Anomaly detection method name used in run.sh, e.g., 'TranAD', 'OmniAnomaly', 'InterFusion', 'SDFVAE', and 'USAD'")
    parser.add_argument("--output_folder_name", type=str, default="1116",
                        help="Output folder name passed to run.sh (e.g., experiment ID)")
    parser.add_argument("--output_suffix", type=str, default="result",
                        help="Suffix for output anomalies file, e.g., '14_to_15' → anomalies_14_to_15.npy")

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
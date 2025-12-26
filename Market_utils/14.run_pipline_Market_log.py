#!/usr/bin/env python3
import subprocess
import os
import argparse

import numpy as np
import json

def extract_log_time_range_from_npy(
    data_npy_path: str,
    meta_npy_path: str,
    start_ts: int,
    end_ts: int,
    output_npy_path: str
):
    """
    从原始 LOG .npy 文件中提取 [start_ts, end_ts] 时间范围内的数据。
    Meta 是 .npy 文件，加载后是 dict，包含 'timestamps' 列表（每个桶的 Unix 时间戳）。
    """
    # Load meta (it's a .npy file containing a dict)
    meta = np.load(meta_npy_path, allow_pickle=True).item()
    timestamps = np.array(meta['timestamps'])  # shape (T,)
    num_buckets_total = len(timestamps)

    # Find indices where timestamp is in [start_ts, end_ts]
    mask = (timestamps >= start_ts) & (timestamps <= end_ts)
    selected_indices = np.where(mask)[0]

    if len(selected_indices) == 0:
        raise ValueError(f"No timestamps found in range [{start_ts}, {end_ts}]")

    i_start = selected_indices[0]
    i_end_excl = selected_indices[-1] + 1  # inclusive end → exclusive slice

    actual_start_ts = timestamps[i_start]
    actual_end_ts = timestamps[i_end_excl - 1]

    print(f"[LOG] Extracting buckets [{i_start}, {i_end_excl}) → time [{actual_start_ts}, {actual_end_ts}] "
          f"({len(selected_indices)} buckets)")

    # Load data and slice
    data = np.load(data_npy_path)  # shape (entities, T, features)
    if data.shape[1] != num_buckets_total:
        print(f"⚠️ Warning: data has {data.shape[1]} time steps, but meta has {num_buckets_total}")

    extracted = data[:, i_start:i_end_excl, :]
    np.save(output_npy_path, extracted)
    print(f"✅ LOG extracted shape: {extracted.shape} → {output_npy_path}")

def run_cmd(cmd, cwd=None):
    """运行 shell 命令，打印并等待完成"""
    print(f"\n[CMD] {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, text=True)
    return result

def main(args):
    base_dir = "/root/shared-nvme/work/timeSeries/OmniTransfer_new"

    offline_src = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Market/log/temp_data/raw_data/train_valid/{args.date_offline}/{args.cloudbed}/raw_log/service_log_patterns_count.npy"
    )
    offline_log_app_meta_file = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Market/log/temp_data/raw_data/train_valid/{args.date_offline}/{args.cloudbed}/raw_log/service_log_patterns_count_meta.npy"
    )
    
    online_src = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Market/log/temp_data/raw_data/train_valid/{args.date_online}/{args.cloudbed}/raw_log/service_log_patterns_count.npy"
    )
    online_log_app_meta_file = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Market/log/temp_data/raw_data/train_valid/{args.date_online}/{args.cloudbed}/raw_log/service_log_patterns_count_meta.npy"
    )
    log_template_file = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Market/log/analysis/log_istio_patterns.json"  
    )

    dst_offline = os.path.join(base_dir, "dataset/data2/offline_data.npy")
    dst_online = os.path.join(base_dir, "dataset/data2/online_data.npy")
    
    # Step 1: 复制数据文件
    extract_log_time_range_from_npy(offline_src, offline_log_app_meta_file,
                                args.date_offline_start_ts, args.date_offline_end_ts,
                                dst_offline)
    extract_log_time_range_from_npy(online_src, online_log_app_meta_file,
                                args.date_online_start_ts, args.date_online_end_ts,
                                dst_online)

    # Step 3: 聚类（使用 Market 专用聚类脚本）
    cluster_script = os.path.join(base_dir, "code/cluster/cluster_Market.py")
    run_cmd(f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {cluster_script}", cwd=base_dir)

    # Step 4: 运行异常检测方法（如 TranAD）
    data_dir = os.path.join(base_dir, f"code/test_dataset/{args.data_dir}")
    run_cmd(f"bash run_new.sh {args.method} {args.output_folder_name} {data_dir}", cwd=base_dir)

    # Step 5: 分析 log 异常
    pred_dir = os.path.join(
        base_dir,
        f"{args.output_folder_name}/{args.method}/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"
    )
    analyze_script = os.path.join(base_dir, "Market_utils/13.Market_log_analyze_anomalies.py")
    output_file = f"{base_dir}/{args.output_folder_name}/Market_log_anomalies_{args.date_online}_{args.output_suffix}.npy"

    cmd_analyze = (
        f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {analyze_script} "
        f"--log_meta {online_log_app_meta_file} "
        f"--log_patterns {log_template_file} "
        f"--pred_dir {args.output_folder_name} "
        f"--pred_start {args.date_online_start_ts} "
        f"--pred_end {args.date_online_end_ts} "
        f"--query_start {args.date_online_start_ts} "
        f"--query_end {args.date_online_end_ts} "
        f"--output {output_file} "
        f"--anomaly_report {args.date_online}_{args.output_suffix}"
    )
    run_cmd(cmd_analyze, cwd=base_dir)

    print(f"\n✅ Market Pipeline completed. Anomaly results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full OmniTransfer anomaly detection pipeline for Market dataset.")

    parser.add_argument("--date_offline", type=str, required=True,
                        help="Date string for offline (normal) data, e.g., '2022_03_21'")
    parser.add_argument("--date_online", type=str, required=True,
                        help="Date string for online (faulty) data, e.g., '2022_03_21'")
    parser.add_argument("--cloudbed", type=str, required=True,
                        help="Cloudbed name, e.g., 'cloudbed-1' or 'cloudbed-2'")
    parser.add_argument("--date_offline_start_ts", type=int, required=True,
                        help="Start timestamp (Unix epoch), e.g., 1614972600")
    parser.add_argument("--date_offline_end_ts", type=int, required=True,
                        help="End timestamp (Unix epoch), e.g., 1614974400")
    parser.add_argument("--date_online_start_ts", type=int, required=True,
                        help="Start timestamp (Unix epoch), e.g., 1614972600")
    parser.add_argument("--date_online_end_ts", type=int, required=True,
                        help="End timestamp (Unix epoch), e.g., 1614974400")
    parser.add_argument("--method", type=str, default="TranAD",
                        help="Anomaly detection method name used in run.sh, e.g., 'TranAD', 'OmniAnomaly', 'InterFusion', 'SDFVAE', and 'USAD'")
    parser.add_argument("--output_folder_name", type=str, default="1116",
                        help="Output folder name passed to run.sh (e.g., experiment ID)")
    parser.add_argument("--output_suffix", type=str, default="result",
                        help="Suffix for output anomalies file, e.g., '14_to_15' → anomalies_14_to_15.npy")
    parser.add_argument("--data_dir", type=str, default="data2",
                        help="Directory name for test data, e.g., 'data1', 'data2'")

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
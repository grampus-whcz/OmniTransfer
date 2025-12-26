#!/usr/bin/env python3
import subprocess
import os
import argparse

import numpy as np
import json

def extract_time_range_from_npy(
    data_npy_path: str,
    meta_json_path: str,
    start_ts: int,
    end_ts: int,
    output_npy_path: str
):
    """
    从原始 trace .npy 文件中提取 [start_ts, end_ts] 时间范围内的数据。

    Parameters:
    - data_npy_path: 原始 .npy 文件路径，如 container_cloudbed-1_2022_03_20_60s.npy
    - meta_json_path: 对应的 metadata JSON 文件路径
    - start_ts: 起始 Unix 时间戳（包含）
    - end_ts: 结束 Unix 时间戳（包含）
    - output_npy_path: 输出 .npy 文件路径
    """
    # Step 1: Load metadata
    with open(meta_json_path, 'r') as f:
        meta = json.load(f)
    
    bucket_sec = meta["bucket_sec"]
    global_start_ts = meta["global_start_sec"]  # 第一个桶的时间戳
    num_buckets_total = meta["num_buckets"]

    # Step 2: Compute bucket indices
    # 每个桶 i 对应时间: global_start_ts + i * bucket_sec
    # 我们要找最小的 i 满足 time >= start_ts → i_start = ceil((start_ts - global_start_ts) / bucket_sec)
    # 但为了安全，我们用 floor 并 clamp

    if start_ts < global_start_ts:
        raise ValueError(f"start_ts ({start_ts}) is before data start time ({global_start_ts})")
    if end_ts > meta["global_end_sec"]:
        print(f"⚠️ Warning: end_ts ({end_ts}) exceeds data end time ({meta['global_end_sec']})")

    # Compute inclusive bucket indices
    i_start = (start_ts - global_start_ts) // bucket_sec
    i_end_excl = (end_ts - global_start_ts) // bucket_sec + 1  # 因为 end_ts 是包含的

    # Clamp to valid range
    i_start = max(0, min(i_start, num_buckets_total - 1))
    i_end_excl = max(i_start + 1, min(i_end_excl, num_buckets_total))

    # Final slice: [i_start, i_end_excl)
    actual_start_ts = global_start_ts + i_start * bucket_sec
    actual_end_ts = global_start_ts + (i_end_excl - 1) * bucket_sec

    print(f"File: {data_npy_path}, Data covers [{actual_start_ts}, {actual_end_ts}] (Unix), "
          f"buckets [{i_start}, {i_end_excl}), duration: {(i_end_excl - i_start) * bucket_sec}s")

    # Step 3: Load and slice data
    data = np.load(data_npy_path)  # shape (entities, time, features)
    if data.shape[1] != num_buckets_total:
        print(f"⚠️ Warning: data has {data.shape[1]} buckets, but metadata says {num_buckets_total}")

    extracted = data[:, i_start:i_end_excl, :]  # shape (73, L, 64)

    # Step 4: Save
    np.save(output_npy_path, extracted)
    print(f"✅ Extracted shape: {extracted.shape} → saved to {output_npy_path}")

def run_cmd(cmd, cwd=None):
    """运行 shell 命令，打印并等待完成"""
    print(f"\n[CMD] {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, text=True)
    return result

def main(args):
    base_dir = "/root/shared-nvme/work/timeSeries/OmniTransfer_new"

    offline_src = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Market/trace/{args.cloudbed}_{args.date_offline}_trace_edge_bucket_60.npy"
    )
    offline_trace_app_meta_file = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Market/trace/{args.cloudbed}_{args.date_offline}_trace_edge_bucket_60.meta.json"
    )
    
    online_src = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Market/trace/{args.cloudbed}_{args.date_online}_trace_edge_bucket_60.npy"
    )
    online_trace_app_meta_file = os.path.join(
        base_dir,
        f"OpenRCA_preprocess_dataset/Market/trace/{args.cloudbed}_{args.date_online}_trace_edge_bucket_60.meta.json"
    )

    dst_offline = os.path.join(base_dir, "dataset/data2/offline_data.npy")
    dst_online = os.path.join(base_dir, "dataset/data2/online_data.npy")
    
    # Step 1: 复制数据文件
    extract_time_range_from_npy(offline_src, offline_trace_app_meta_file,
                                args.date_offline_start_ts, args.date_offline_end_ts,
                                dst_offline)
    extract_time_range_from_npy(online_src, online_trace_app_meta_file,
                                args.date_online_start_ts, args.date_online_end_ts,
                                dst_online)

    # Step 3: 聚类（使用 Market 专用聚类脚本）
    cluster_script = os.path.join(base_dir, "code/cluster/cluster_Market.py")
    run_cmd(f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {cluster_script}", cwd=base_dir)

    # Step 4: 运行异常检测方法（如 TranAD）
    data_dir = os.path.join(base_dir, f"code/test_dataset/{args.data_dir}")
    run_cmd(f"bash run_new.sh {args.method} {args.output_folder_name} {data_dir}", cwd=base_dir)

    # Step 5: 分析 trace 异常
    pred_dir = os.path.join(
        base_dir,
        f"{args.output_folder_name}/{args.method}/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"
    )
    analyze_script = os.path.join(base_dir, "Market_utils/11.Market_trace_analyze_anomalies.py")
    output_file = f"{base_dir}/{args.output_folder_name}/Market_trace_anomalies_{args.date_online}_{args.output_suffix}.npy"

    cmd_analyze = (
        f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {analyze_script} "
        f"--cloudbed {args.cloudbed} "
        f"--date {args.date_online} "
        f"--pred_dir {pred_dir} "
        f"--output_folder_name {args.output_folder_name} "
        f"--start {args.date_online_start_ts} "
        f"--end {args.date_online_end_ts} "
        f"--output {output_file} "
        f"--report_suffix {args.date_online}_{args.output_suffix} "
        f"--bucket_sec 60"
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
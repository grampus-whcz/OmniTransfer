#!/usr/bin/env python3
import subprocess
import os
import argparse

def run_cmd(cmd, cwd=None):
    print(f"\n[CMD] {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, text=True)
    return result

def main(args):
    base_dir = "/root/shared-nvme/work/timeSeries/OmniTransfer_new"
    trace_dir = os.path.join(base_dir, "OpenRCA_preprocess_dataset/Telecom/trace")

    # 构造原始文件路径
    offline_npy = os.path.join(trace_dir, f"telecom_{args.date_offline}_trace_edge_bucket_{args.bucket_sec}.npy")
    offline_meta = os.path.join(trace_dir, f"telecom_{args.date_offline}_trace_edge_bucket_{args.bucket_sec}.meta.json")
    online_npy = os.path.join(trace_dir, f"telecom_{args.date_online}_trace_edge_bucket_{args.bucket_sec}.npy")
    online_meta = os.path.join(trace_dir, f"telecom_{args.date_online}_trace_edge_bucket_{args.bucket_sec}.meta.json")

    if not all(os.path.exists(f) for f in [offline_npy, offline_meta, online_npy, online_meta]):
        raise FileNotFoundError("One or more trace files not found.")

    # Step 1: 对齐 trace 数据（生成 aligned offline/online + info.json）
    align_script = os.path.join(base_dir, "Telecom_utils/12.align_telecom_trace.py")
    alignment_info = os.path.join(base_dir, f"dataset/{args.data_dir}/trace_alignment_info.json")
    dst_offline = os.path.join(base_dir, f"dataset/{args.data_dir}/offline_data.npy")
    dst_online = os.path.join(base_dir, f"dataset/{args.data_dir}/online_data.npy")

    cmd_align = (
        f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {align_script} "
        f"--offline_npy {offline_npy} "
        f"--offline_meta {offline_meta} "
        f"--online_npy {online_npy} "
        f"--online_meta {online_meta} "
        f"--output_offline {dst_offline} "
        f"--output_online {dst_online} "
        f"--output_info {alignment_info}"
    )
    run_cmd(cmd_align, cwd=base_dir)

    # Step 2: 聚类（输入已是 (N, T, F)，N = len(common_edges)）
    cluster_script = os.path.join(base_dir, "code/cluster/cluster_Telecom.py")
    run_cmd(f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {cluster_script}", cwd=base_dir)

    # Step 3: 运行异常检测
    data_dir = os.path.join(base_dir, f"code/test_dataset/{args.data_dir}")
    run_cmd(f"bash run_new.sh {args.method} {args.output_folder_name} {data_dir}", cwd=base_dir)

    # Step 4: 分析异常（使用对齐后的 info.json）
    analyze_script = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Telecom_utils/9.Telecom_trace_analyze_anomalies.py"
    output_file = f"{base_dir}/{args.output_folder_name}/Telecom_trace_anomalies_{args.date_online}_{args.output_suffix}.npy"

    cmd_analyze = (
        f"/root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python {analyze_script} "
        f"--info_json {alignment_info} "
        f"--pred_dir {args.output_folder_name} "
        f"--start {args.start_ts} "
        f"--end {args.end_ts} "
        f"--output {output_file} "
        f"--anomaly_report {args.date_online}_{args.output_suffix}"
    )
    run_cmd(cmd_analyze, cwd=base_dir)

    print(f"\n✅ Telecom Trace Pipeline completed. Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OmniTransfer pipeline for Telecom Trace data with entity alignment.")

    parser.add_argument("--date_offline", type=str, required=True,
                        help="Normal date, e.g., '2020_04_20'")
    parser.add_argument("--date_online", type=str, required=True,
                        help="Fault date, e.g., '2020_04_11'")
    parser.add_argument("--bucket_sec", type=int, default=60,
                        help="Bucket granularity, e.g., 60")
    parser.add_argument("--start_ts", type=int, required=True,
                        help="Query start Unix timestamp")
    parser.add_argument("--end_ts", type=int, required=True,
                        help="Query end Unix timestamp")
    parser.add_argument("--method", type=str, default="TranAD",
                        help="Anomaly detection method")
    parser.add_argument("--output_folder_name", type=str, default="1120",
                        help="Experiment output folder")
    parser.add_argument("--output_suffix", type=str, default="result",
                        help="Output file suffix")
    parser.add_argument("--data_dir", type=str, default="data3",
                        help="Directory name for test data, e.g., 'data1', 'data2', 'data3'")

    args = parser.parse_args()
    try:
        main(args)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed: {e.cmd}")
        exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        exit(1)
#！/bin/bash
# nohup bash anomaly.sh > anomaly_new.log 2>&1 &

# Convert to Unix timestamp (UTC)
# 2021-03-06 23:30:00 UTC → 1615044600
# 2021-03-07 24:00:00 UTC → 1615046400

# python Bank_metric_analyze_anomalies.py \
#   --meta /root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/metric/metadata_container_2021_03_06_60s.json \
#   --pred_dir /root/shared-nvme/work/timeSeries/OmniTransfer_new/1116/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g \
#   --start 1614972600  \
#   --end 1614974400 \
#   --output ./anomalies_14_to_15.npy

# Example time range: 2021-03-04 10:00:00 to 12:00:00 UTC
# START_TS=1615044600
# END_TS=1615046400

# python Bank_trace_analyze_anomalies.py \
#   --meta /root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/trace/2021_03_05_trace_edge_bucket_60.meta.json \
#   --pred_dir /root/shared-nvme/work/timeSeries/OmniTransfer_new/1116/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g \
#   --start $START_TS \
#   --end $END_TS \
#   --output ./trace_anomalies_10_to_12.npy

# /root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python /root/shared-nvme/work/timeSeries/OmniTransfer_new/run_pipline_Bank_metric_container.py \
# --date_offline 2021_03_05 --date_online 2021_03_06 --start_ts 1614960000 --end_ts 1614967200 --method TranAD --output_folder_name 1117 --output_suffix 0000_0200

python 2.Bank_cluster_window_analyze_anomalies_old_ablation.py \
--date_online 2021_03_04 \
--output_suffix 0100_0130 \
--eps 60 \
--min_samples 3 \
--output_folder_name 1204

# # 拓扑感知聚类，时序分段5分钟
# python 2.Bank_cluster_window_analyze_anomalies_old_topo_aware.py \
# --date_online 2021_03_04 \
# --output_suffix 0100_0130 \
# --output_folder_name 1204

# # 示例：5分钟等宽分段
# python 2.Bank_cluster_window_analyze_anomalies_old_equal_width.py \
# --date_online 2021_03_04 \
# --output_suffix 0100_0130 \
# --seg_width 300 \
# --output_folder_name 1204
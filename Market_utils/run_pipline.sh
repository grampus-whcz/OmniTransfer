#！/bin/bash
# nohup bash run_pipline.sh >> run_pipline_log_8.log 2>&1 &

# python 9.run_pipline_Market_metric.py \
#   --date_offline 2022_03_20 \
#   --date_offline_start_ts 1647734400 \
#   --date_offline_end_ts 1647739800 \
#   --date_online 2022_03_20 \
#   --date_online_start_ts 1647739800 \
#   --date_online_end_ts 1647741600 \
#   --cloudbed cloudbed-1 \
#   --method TranAD \
#   --output_folder_name 1215 \
#   --output_suffix 1200_1230 \
#   --data_dir data2 \
#   --metric_type node

# # 注意LLM调用工具时，参数传递不依从问题。时间戳和日期根本对不上
# /root/shared-nvme/.conda/envs/OmniTransfer_py3.7/bin/python /root/shared-nvme/work/timeSeries/OmniTransfer_new/Market_utils/9.run_pipline_Market_metric.py \
# --date_offline 2022_03_21 \
# --date_online 2022_03_20 \
# --date_offline_start_ts 1647813600 \
# --date_offline_end_ts 1647817200 \
# --date_online_start_ts 1647738000 \
# --date_online_end_ts 1647741600 \
# --cloudbed cloudbed-1 \
# --method TranAD \
# --output_folder_name 1215 \
# --output_suffix 0930_1000 \
# --metric_type node \
# --data_dir data2

python -u 16.run_all_commands.py






















## node runtime entity太少 已增补数据，解决问题
## mesh 有超大值 cluster报错 cluster_Market.py有除0bug
# echo "Market_metric_container"
# python 9.run_pipline_Market_metric.py \
#   --date_offline 2022_03_21 \
#   --date_online 2022_03_20 \
#   --cloudbed cloudbed-1 \
#   --date_offline_start_ts 1647792000 \
#   --date_offline_end_ts 1647793800 \
#   --date_online_start_ts 1647736200 \
#   --date_online_end_ts 1647738000 \
#   --method TranAD \
#   --output_folder_name 1215 \
#   --output_suffix 0830_to_0900 \
#   --data_dir data2 \
#   --metric_type runtime

# echo "Market_trace"
# python 12.run_pipline_Market_trace.py \
#   --date_offline 2022_03_21 \
#   --date_online 2022_03_20 \
#   --cloudbed cloudbed-1 \
#   --date_offline_start_ts 1647792000 \
#   --date_offline_end_ts 1647793800 \
#   --date_online_start_ts 1647736200 \
#   --date_online_end_ts 1647738000 \
#   --method TranAD \
#   --output_folder_name 1215 \
#   --output_suffix 0830_to_0900 \
#   --data_dir data2

# python 11.Market_trace_analyze_anomalies.py \
#   --date 2022_03_20 \
#   --cloudbed cloudbed-1 \
#   --pred_dir /root/shared-nvme/work/timeSeries/OmniTransfer_new/1215/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g \
#   --start 1647736200 \
#   --end 1647738000 \
#   --output /root/shared-nvme/work/timeSeries/OmniTransfer_new/1215/Market_trace_anomalies_2022_03_20_0830_to_0900.npy \
#   --report_suffix 0830_to_0900 \
#   --bucket_sec 60

# echo "Market_log"
# python 14.run_pipline_Market_log.py \
#   --date_offline 2022_03_21 \
#   --date_online 2022_03_20 \
#   --cloudbed cloudbed-1 \
#   --date_offline_start_ts 1647792000 \
#   --date_offline_end_ts 1647793800 \
#   --date_online_start_ts 1647736200 \
#   --date_online_end_ts 1647738000 \
#   --method TranAD \
#   --output_folder_name 1215 \
#   --output_suffix 0830_to_0900 \
#   --data_dir data2

# python 15.Market_cluster_window_analyze_anomalies.py \
#   --date_online 2022_03_20 \
#   --output_suffix 0830_to_0900 \
#   --eps 60 \
#   --min_samples 2 \
#   --output_folder_name 1215
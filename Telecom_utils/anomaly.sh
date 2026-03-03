#！/bin/bash
# nohup bash anomaly.sh > anomaly_5.log 2>&1 &

# Convert to Unix timestamp (UTC)
# 2021-03-06 23:30:00 UTC → 1615044600
# 2021-03-07 24:00:00 UTC → 1615046400

# python 7.Telecom_metric_A_analyze_anomalies.py \
#     --pred_dir 1120 \
#     --start 1586554200 \
#     --end 1586556000 \
#     --anomaly_report 2020_04_11_full

# python 8.Telecom_metric_B_analyze_anomalies.py \
#     --pred_dir 1120 \
#     --start 1586554200 \
#     --end 1586556000 \
#     --anomaly_report 2020_04_11_osb001

# python 9.Telecom_trace_analyze_anomalies.py \
#     --pred_dir 1120 \
#     --start 1586534400 \
#     --end 1586556000 \
#     --anomaly_report 2020_04_11_trace

# python 10.run_telecom_metric_A_pipeline.py \
#     --date_offline 2020_04_20 \
#     --date_online 2020_04_11 \
#     --bucket_sec 60 \
#     --start_ts 1586534400 \
#     --end_ts 1586536200 \
#     --method TranAD \
#     --output_folder_name 1216 \
#     --data_dir data3 \
#     --output_suffix 0000_0030

# python 11.run_telecom_metric_B_pipeline.py \
#     --date_offline 2020_04_20 \
#     --date_online 2020_04_11 \
#     --bucket_sec 60 \
#     --start_ts 1586534400 \
#     --end_ts 1586536200 \
#     --method TranAD \
#     --output_folder_name 1216 \
#     --data_dir data3 \
#     --output_suffix 0000_0030

# python 13.run_telecom_trace_pipeline.py \
#     --date_offline 2020_04_20 \
#     --date_online 2020_04_11 \
#     --bucket_sec 60 \
#     --start_ts 1586534400 \
#     --end_ts 1586536200 \
#     --method TranAD \
#     --output_folder_name 1216 \
#     --data_dir data3 \
#     --output_suffix 0000_0030

# python 14.Telecom_cluster_window_analyze_anomalies.py \
#   --date_online 2020_04_11 \
#   --output_suffix 0000_0030 \
#   --eps 60 \
#   --min_samples 2 \
#   --output_folder_name 1216

python 14.Telecom_cluster_window_analyze_anomalies.3.3.py \
  --date_online 2020_04_11 \
  --output_suffix 0000_0030 \
  --eps 60 \
  --min_samples 2 \
  --output_folder_name 1216
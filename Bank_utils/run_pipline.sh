#！/bin/bash
# nohup bash run_pipline.sh > run_pipline_log6.log 2>&1 &

# echo "Bank_metric_container"
# python run_pipline_Bank_metric_container.py \
#   --date_offline 2021_03_05 \
#   --date_online 2021_03_06 \
#   --start_ts 1614972600 \
#   --end_ts 1614974400 \
#   --method TranAD \
#   --output_folder_name 1116 \
#   --output_suffix 14_to_15

# echo "Bank_metric_app"
# python run_pipline_Bank_metric_app.py \
#   --date_offline 2021_03_05 \
#   --date_online 2021_03_06 \
#   --start_ts 1614972600 \
#   --end_ts 1614974400 \
#   --method TranAD \
#   --output_folder_name 1116 \
#   --output_suffix 14_to_15

# echo "Bank_trace"
# python run_pipline_Bank_trace.py \
#   --date_offline 2021_03_05 \
#   --date_online 2021_03_06 \
#   --start_ts 1614972600 \
#   --end_ts 1614974400 \
#   --method TranAD \
#   --output_folder_name 1116 \
#   --output_suffix 14_to_15

# echo "Bank_log"
# python run_pipline_Bank_log.py \
#   --date_offline 2021_03_05 \
#   --date_online 2021_03_06 \
#   --start_ts 1614972600 \
#   --end_ts 1614974400 \
#   --method TranAD \
#   --output_folder_name 1116 \
#   --output_suffix 14_to_15

# python -u 4.run_all_analyses.py


# python Bank_cluster_window_analyze_anomalies_2.7.py \
#   --date_online 2021_03_04 \
#   --eps 60 \
#   --min_samples 3 \
#   --output_folder_name 1204 \
#   --output_suffix 0230_0300

# python Bank_cluster_window_analyze_anomalies_3.13.py \
#   --date_online 2021_03_04 \
#   --eps 60 \
#   --min_samples 3 \
#   --output_folder_name 1204 \
#   --output_suffix 0230_0300

python Bank_knowledge_graph_RCA.3.20.py \
  --date_online 2021_03_04 \
  --output_folder_name 1204 \
  --output_suffix 0230_0300
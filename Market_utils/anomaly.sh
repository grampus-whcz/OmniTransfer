#！/bin/bash
# nohup bash anomaly.sh > anomaly.log 2>&1 &

# Convert to Unix timestamp (UTC)
# 2021-03-06 23:30:00 UTC → 1615044600
# 2021-03-07 24:00:00 UTC → 1615046400

python 8.Market_metric_container_analyze_anomalies.py \
    --meta /root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Market/metric/market_processed/metadata_container_cloudbed-1_2022_03_20_60s.json \
    --pred_dir 1121 \
    --start 1647772200 \
    --end 1647774000 \
    --output /root/shared-nvme/work/timeSeries/OmniTransfer_new/1121/Market_container_anomalies_2022_03_20.npy \
    --anomaly_report cloudbed1_20220320_cpu_spike
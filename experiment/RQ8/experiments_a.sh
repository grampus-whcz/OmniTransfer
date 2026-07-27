#! /bin/bash

# nohup bash experiments_a.sh >> experiments_a.log 2>&1 &

# 示例：将日志第一个时间戳改为 2026-07-20 09:00:00.000
python change_t.py \
--input /root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ8/Bank_all_RAG_k1_glm-4.7.log \
--output /root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ8/Bank_all_RAG_k1_glm-4.7_new.log \
--new-start "2026-07-20 09:00:00.000"



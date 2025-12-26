

## get_representative_groudtruth.py 根据groundtruth.csv的内容生成四种类型的报告。当groudtruth内容较多时，耗时较长。

## according to npy, generate the anomalies' report of 1614867900' (-1, +5) 
## get_representative_anomaly_report.py 为每个groudtruth生成基于故障点前后(-1,5)时间区域内的异常报告
```
python get_representative_anomaly_report.py \
  --date 2021_03_04 \
  --window 2200_2230 \
  --ts 1614867900 \
  --output ./report/report_20210304_1614867900.txt
```

## report生成
0.get_all_representative_anomaly_reports.py 读取groundtruth.csv，为每个故障生成故障点附近(-1, +5)的故障报告, 该程序调用0.get_representative_anomaly_report.py
由于log模板有大量的干扰信息，我们对log模板进行了裁剪，删除了无关字符,如:<:NUM:><:*:>。一些无意义的模板，我们直接把它置为空字符串。同时我们在get_representative_anomaly_report.py中调整的不同模态异常的顺序：
```
# 写入文件
with open(output_file, 'w', encoding='utf-8') as f:        
    write_metric_app_section(f, metric_app_groups)
    write_metric_container_section(f, metric_container_groups)
    write_trace_section(f, trace_groups)
    write_log_section(f, log_groups, LOG_TEMPLATES)

    f.write("💡 Note: 'CST' = China Standard Time (UTC+8).\n")
```
/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/groundtruth.csv

## 1.Bank_generate_root_cause_reports_all30td.py 使用LLM为典型的故障生成可进行RAG的postmortem chunk
目前的方法基于所有类型的报告中的所有30分钟时间跨度中的多实体多属性异常值生成报告，存在异常多，不能聚焦于故障点处的异常的情况。

## 2.Bank_generate_anomalies_data_pic.py 为某个30分钟时间内的异常数据生成异常图形化报告

## 2.Bank_generate_anomalies_data_pic_new.py 为某个30分钟时间内的异常数据生成异常图形化报告. 调整了图的位置, 删除重复图.

## 3.Bank_generate_root_cause_reports_6mtd.py 为故障发生的(-1, +5)时域构建根因分析报告

## 4.clean_root_cause_records.py 对3.Bank_generate_root_cause_reports_6mtd.py生成的jsonl进行进一步优化去掉干扰信息，添加关于log pattern ID对应的文字描述

## 5.generate_rag_ready_jsonl.py 生成符合RAG corpus构建规范的jsonl

## get_cluster_bank_window.py 对具体日期，某半小时跨度内的多实体特征异常检测数据进行聚类分析。
eps 60是一个理想的超参数
```
python get_cluster_bank_window.py \
  --date 2021_03_04 \
  --window 0030_0100 \
  --output ./cluster/clusters_20210304_0030_0100.txt \
  --eps 300 \
  --min_samples 2
```

```
python get_cluster_bank_window.py \
  --date 2021_03_04 \
  --window 2200_2230 \
  --output ./cluster/clusters_20210304_2200_2230.txt \
  --eps 60 \
  --min_samples 2
```

```
python Bank_cluster_window_analyze_anomalies.py \
--date_online 2021_03_06 \
--output_suffix 1830_1900 \
--output_folder_name 1117
```


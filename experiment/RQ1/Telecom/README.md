# 0.Telecom_get_representative_failure_anomalies.py
生成代表性故障的groudtruth.csv文件

# 1.Telecom_get_representative_anomaly_report.py
为一个故障生成报告

# 1.Telecom_get_representative_anomaly_report_new.py
由于一个故障报告中含有的异常可能过多。因此，只关注那些“实体+属性”异常大于等于2的。

# 2.Telecom_get_all_representative_anomaly_reports.py
为所有代表性故障生成故障报告，保存在report中

# 3.Telecom_generate_rag_ready_jsonl.py
使用llm生成RAG所需的jsonl源文件
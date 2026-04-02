# 14.Telecom_cluster_window_analyze_anomalies.py
原始cluster，不对异常做任何处理

# 14.Telecom_cluster_window_analyze_anomalies.3.3_old.py
只 load metric_A中3个以上的异常
第一步：对 metric_A 按 (entity, attribute) 分组计数
第二步：只保留出现 >=3 次的 metric_A 异常
第三步：合并 filtered_metric_a + 其他类型（metric_B, trace）
去重：避免同一 (type, entity, attr, ts) 重复

# 14.Telecom_cluster_window_analyze_anomalies.3.3.py
ClusTopoRCA 版本

# 14.Telecom_cluster_window_analyze_anomalies.3.12.py
# 15.Telecom_knowledge_graph_RCA.3.19.py
知识图谱 版本
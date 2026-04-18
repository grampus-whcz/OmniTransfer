import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 1. 兼容处理：修复字体警告+Matplotlib样式问题（Ubuntu专用）--------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# -------------------------- 2. 数据加载与预处理 --------------------------
trace_path = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Bank_time_line/Bank_trace_anomalies_2021_03_06_1830_1900.npy"
metric_container_path = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Bank_time_line/Bank_metric_container_anomalies_2021_03_06_1830_1900.npy"
metric_app_path = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Bank_time_line/Bank_metric_app_anomalies_2021_03_06_1830_1900.npy"
log_path = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Bank_time_line/Bank_log_anomalies_2021_03_06_1830_1900.npy"

def load_anomaly_data(file_path, data_type):
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"{data_type}数据加载成功，共{len(data)}条异常")
        return data
    except Exception as e:
        print(f"加载{data_type}数据失败，使用示例数据: {e}")
        if data_type == "trace":
            return np.array([
                ('IG01->Tomcat02', 'duration', 1615026655),
                ('IG01->Tomcat02', 'duration', 1615026835),
                ('IG01->Tomcat02', 'frequency', 1615026775),
                ('IG01->Tomcat02', 'frequency', 1615026835),
                ('IG02->Tomcat02', 'duration', 1615026655),
                ('IG02->Tomcat02', 'duration', 1615026835),
                ('IG02->Tomcat02', 'frequency', 1615026835),
                ('Tomcat02->MG01', 'frequency', 1615026655),
                ('Tomcat02->MG01', 'frequency', 1615026835),
                ('Tomcat02->MG02', 'frequency', 1615026835),
                ('Tomcat02->Tomcat02', 'duration', 1615026655),
                ('Tomcat02->Tomcat02', 'duration', 1615026835),
                ('Tomcat02->Tomcat02', 'frequency', 1615026835)
            ], dtype=object)
        elif data_type == "metric_container":
            sample_data = [
                ('IG01', 'OSLinux-', 1615027920), ('IG01', 'OSLinux-', 1615028160),
                ('IG01', 'OSLinux-', 1615027860), ('IG01', 'OSLinux-', 1615028100),
                ('IG01', 'OSLinux-', 1615027860), ('IG01', 'OSLinux-', 1615028100),
                ('IG02', 'OSLinux-', 1615027440), ('IG02', 'OSLinux-', 1615027440),
                ('MG01', 'OSLinux-', 1615027860), ('MG01', 'OSLinux-', 1615027860),
                ('MG02', 'OSLinux-', 1615026720), ('MG02', 'OSLinux-', 1615027080),
                ('Mysql01', 'Containe', 1615027560), ('Mysql01', 'Mysql-My', 1615027560)
            ]
            repeat_times = 208 // len(sample_data) + 1
            full_data = sample_data * repeat_times
            return np.array(full_data[:208], dtype=object)
        elif data_type == "metric_app":
            return np.array([
                ('ServiceTest1', 'mrt', 1615026600), ('ServiceTest1', 'mrt', 1615026780),
                ('ServiceTest1', 'mrt', 1615026840), ('ServiceTest10', 'mrt', 1615026600),
                ('ServiceTest10', 'mrt', 1615026660), ('ServiceTest10', 'mrt', 1615026780),
                ('ServiceTest11', 'mrt', 1615026780), ('ServiceTest2', 'cnt', 1615026840),
                ('ServiceTest2', 'mrt', 1615026780), ('ServiceTest2', 'mrt', 1615026840),
                ('ServiceTest3', 'mrt', 1615026780), ('ServiceTest3', 'mrt', 1615026840),
                ('ServiceTest5', 'mrt', 1615026660), ('ServiceTest5', 'mrt', 1615026780),
                ('ServiceTest6', 'mrt', 1615026780), ('ServiceTest7', 'mrt', 1615026780),
                ('ServiceTest8', 'mrt', 1615026780), ('ServiceTest9', 'mrt', 1615026660),
                ('ServiceTest9', 'mrt', 1615026780)
            ], dtype=object)
        elif data_type == "log":
            sample_log = [
                ('IG01', 2, 'CMS-concurrent-mark', 1615027380),
                ('IG01', 3, '<:*:>', 1615027860),
                ('IG01', 5, 'GC (CMS Final Remark)', 1615027860),
                ('IG01', 7, 'GC (CMS Final Remark)', 1615027020),
                ('IG01', 8, 'GC (Allocation Failure)', 1615027380)
            ]
            repeat_times = 66 // len(sample_log) + 1
            full_log = sample_log * repeat_times
            return np.array(full_log[:66], dtype=object)

# 加载四类数据
trace_data = load_anomaly_data(trace_path, "trace")
metric_container_data = load_anomaly_data(metric_container_path, "metric_container")
metric_app_data = load_anomaly_data(metric_app_path, "metric_app")
log_data = load_anomaly_data(log_path, "log")

# 数据格式标准化
def normalize_data(data, data_type):
    df = pd.DataFrame(data)
    if data_type == "trace":
        df.columns = ["entity", "attribute", "timestamp"]
    elif data_type == "metric_container":
        df.columns = ["entity", "attribute", "timestamp"]
    elif data_type == "metric_app":
        df.columns = ["entity", "attribute", "timestamp"]
    elif data_type == "log":
        df.columns = ["entity", "log_id", "log_content", "timestamp"]
        df["attribute"] = "log_" + df["log_id"].astype(str)
    
    df["anomaly_type"] = data_type
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    return df[["entity", "attribute", "timestamp", "anomaly_type"]]

trace_df = normalize_data(trace_data, "trace")
metric_container_df = normalize_data(metric_container_data, "metric_container")
metric_app_df = normalize_data(metric_app_data, "metric_app")
log_df = normalize_data(log_data, "log")

all_anomalies_df = pd.concat([trace_df, metric_container_df, metric_app_df, log_df], ignore_index=True)
print(f"\n数据预处理完成，共{len(all_anomalies_df)}条异常记录")
print(f"异常类型分布：\n{all_anomalies_df['anomaly_type'].value_counts()}")

# -------------------------- 3. 异常统计分析 --------------------------
entity_anomaly_count = all_anomalies_df.groupby(["entity", "anomaly_type"]).size().unstack(fill_value=0)
entity_anomaly_count["total"] = entity_anomaly_count.sum(axis=1)
entity_anomaly_count_sorted = entity_anomaly_count.sort_values("total", ascending=False)

attribute_anomaly_count = all_anomalies_df.groupby(["attribute", "anomaly_type"]).size().unstack(fill_value=0)
attribute_anomaly_count["total"] = attribute_anomaly_count.sum(axis=1)
attribute_top10 = attribute_anomaly_count.sort_values("total", ascending=False).head(10)

all_anomalies_df["minute"] = all_anomalies_df["timestamp"].dt.floor("min")
time_distribution = all_anomalies_df.groupby(["minute", "anomaly_type"]).size().unstack(fill_value=0)

# -------------------------- 4. 可视化图表（新布局：堆叠柱在左上，饼图在右上，时间图在下方）--------------------------
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

# 上半：左 = 堆叠柱状图，右 = 饼图
ax_stack = fig.add_subplot(gs[0, 0])   # 左上：堆叠柱状图
ax_pie = fig.add_subplot(gs[0, 1])     # 右上：饼图

# 下半：时间分布（跨整行）
ax_time = fig.add_subplot(gs[1, :])

# === 堆叠柱状图：Top10实体的异常类型分布（现在在左上）===
top10_entities_stack = entity_anomaly_count_sorted.head(10).drop("total", axis=1, errors='ignore')
valid_cols = [col for col in top10_entities_stack.columns if col in all_anomalies_df["anomaly_type"].unique()]
top10_entities_stack = top10_entities_stack[valid_cols]

top10_entities_stack.plot(
    kind='bar', stacked=True, ax=ax_stack,
    color=colors[:len(valid_cols)], alpha=0.8,
    width=0.5
)
ax_stack.set_xlabel('Entity', fontsize=11)
ax_stack.set_ylabel('Anomaly Count', fontsize=11)
ax_stack.set_title('① Anomaly type distribution of top10 entities', fontsize=15, fontweight='bold')
ax_stack.legend(
    fontsize=8,
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98)
)
ax_stack.grid(axis='y', alpha=0.3)
ax_stack.tick_params(axis='x', rotation=60)

# === 饼图：异常类型分布（现在在右上）===
anomaly_type_count = all_anomalies_df["anomaly_type"].value_counts()
wedges, texts, autotexts = ax_pie.pie(
    anomaly_type_count.values,
    labels=anomaly_type_count.index,
    autopct='%1.1f%%',
    colors=colors[:len(anomaly_type_count)],
    startangle=90
)
ax_pie.set_title('② Anomaly type distribution', fontsize=15, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)

# === 时间分布折线图（放在下方）===
time_distribution_plot = time_distribution.reset_index()
# time_distribution_plot["minute_str"] = time_distribution_plot["minute"].dt.strftime('%H:%M')
import pandas as pd

# 直接给时间列加上8小时，再格式化
time_distribution_plot["minute_str"] = (time_distribution_plot["minute"] + pd.Timedelta(hours=8)).dt.strftime('%H:%M')

x = list(range(len(time_distribution_plot)))

for i, col in enumerate(time_distribution.columns):
    if col in time_distribution_plot.columns:
        ax_time.plot(
            x, time_distribution_plot[col],
            marker='o', linewidth=2, markersize=4,
            label=col, color=colors[i % len(colors)], alpha=0.8
        )

ax_time.set_xlabel('Time (Minute)', fontsize=11)
ax_time.set_ylabel('Anomaly Count', fontsize=11)
ax_time.set_title('③ Anomaly time distribution (per minute)', fontsize=15, fontweight='bold')

# 图例向右移动，避免遮挡线条
ax_time.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.8, 0.98), borderaxespad=0.)

ax_time.grid(alpha=0.3)

# 确保 x 轴包含首尾时间点
n_ticks = min(7, len(x))
if len(x) <= n_ticks:
    tick_positions = x
else:
    tick_positions = np.linspace(0, len(x) - 1, num=n_ticks, dtype=int)
    tick_positions[-1] = len(x) - 1
    tick_positions = np.unique(tick_positions)

ax_time.set_xticks(tick_positions)
ax_time.set_xticklabels(
    time_distribution_plot["minute_str"].iloc[tick_positions],
    rotation=45, ha='right'
)

# 关键：为右侧图例预留空间（右边留白至 0.88）
plt.tight_layout(rect=[0, 0.03, 0.88, 0.95])

# 保存图像
plt.savefig('2.bank_fault_anomaly_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('2.bank_fault_anomaly_analysis.pdf', dpi=300, bbox_inches='tight')
print("\nCharts saved as:")
print("- 2.bank_fault_anomaly_analysis.png (High-resolution PNG)")
print("- 2.bank_fault_anomaly_analysis.pdf (Vector PDF)")

# -------------------------- 5. 统计结果输出 --------------------------
entity_anomaly_count_sorted.to_csv('2.entity_anomaly_count.csv', encoding='utf-8-sig')
attribute_top10.to_csv('2.attribute_anomaly_top10.csv', encoding='utf-8-sig')
time_distribution.to_csv('2.time_anomaly_distribution.csv', encoding='utf-8-sig')
print("\nStatistical results saved to CSV files:")
print("- 2.entity_anomaly_count.csv (Entity anomaly statistics)")
print("- 2.attribute_anomaly_top10.csv (Top10 attribute anomaly statistics)")
print("- 2.time_anomaly_distribution.csv (Time distribution statistics)")
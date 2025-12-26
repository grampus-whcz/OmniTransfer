import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)

# 示例数据生成（模拟原始数据）
np.random.seed(42)
categories = ['Bank', 'Market', 'Telecom', 'Bank OpenRCA', 'Market OpenRCA', 'Telecom OpenRCA']
n_samples = 30  # 每个类别的样本数

# 模拟运行时间和成本数据
runtime_data = {
    'Category': [],
    'Runtime': []
}
cost_data = {
    'Category': [],
    'Cost': []
}

for cat in categories:
    # 模拟运行时间（单位：秒）
    rt_mean = {'Bank': 50, 'Market': 80, 'Telecom': 120, 'Bank OpenRCA': 90, 'Market OpenRCA': 150, 'Telecom OpenRCA': 100}[cat]
    rt_std = rt_mean * 0.3
    runtime_vals = np.clip(np.random.normal(rt_mean, rt_std, n_samples), 0, 700)
    
    # 模拟成本（单位：美元）
    cost_mean = {'Bank': 0.08, 'Market': 0.12, 'Telecom': 0.15, 'Bank OpenRCA': 0.10, 'Market OpenRCA': 0.18, 'Telecom OpenRCA': 0.12}[cat]
    cost_std = cost_mean * 0.5
    cost_vals = np.clip(np.random.normal(cost_mean, cost_std, n_samples), 0, 0.55)
    
    # 添加到数据中
    runtime_data['Category'].extend([cat] * n_samples)
    runtime_data['Runtime'].extend(runtime_vals)
    
    cost_data['Category'].extend([cat] * n_samples)
    cost_data['Cost'].extend(cost_vals)

# 转换为 DataFrame
df_runtime = pd.DataFrame(runtime_data)
df_cost = pd.DataFrame(cost_data)

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# 左图：Runtime
sns.boxplot(data=df_runtime, x='Runtime', y='Category', ax=ax1, palette='Set3', width=0.5)
sns.stripplot(data=df_runtime, x='Runtime', y='Category', ax=ax1, color='orange', size=4, alpha=0.8, jitter=True, linewidth=0.5)
ax1.set_xlabel('Runtime(s)')
ax1.set_title('Runtime(s)', fontsize=14, pad=20)
ax1.grid(True, axis='x', alpha=0.3)

# 右图：Cost
sns.boxplot(data=df_cost, x='Cost', y='Category', ax=ax2, palette='Set3', width=0.5)
sns.stripplot(data=df_cost, x='Cost', y='Category', ax=ax2, color='orange', size=4, alpha=0.8, jitter=True, linewidth=0.5)
ax2.set_xlabel('Cost($)') 
ax2.set_title('Cost($)', fontsize=14, pad=20)
ax2.grid(True, axis='x', alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图像到文件
plt.savefig("output_figure.pdf", bbox_inches='tight')  # 保存为PDF
plt.savefig("output_figure.png", dpi=300, bbox_inches='tight')  # 保存为PNG

# 显示图像
plt.show() 
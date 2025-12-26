import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# 示例数据生成（与原代码一致）
np.random.seed(42)
categories_original = ['Bank', 'Market', 'Telecom']
categories_openrca = ['Bank OpenRCA', 'Market OpenRCA', 'Telecom OpenRCA']
all_categories = categories_original + categories_openrca
n_samples = 30

runtime_data = {'Category': [], 'Runtime': [], 'Method': []}
cost_data = {'Category': [], 'Cost': [], 'Method': []}

for cat in all_categories:
    method = 'OpenRCA' if 'OpenRCA' in cat else 'Original'
    
    rt_mean = {'Bank': 50, 'Market': 80, 'Telecom': 120, 
               'Bank OpenRCA': 90, 'Market OpenRCA': 150, 'Telecom OpenRCA': 100}[cat]
    rt_std = rt_mean * 0.3
    runtime_vals = np.clip(np.random.normal(rt_mean, rt_std, n_samples), 0, 700)
    
    cost_mean = {'Bank': 0.08, 'Market': 0.12, 'Telecom': 0.15,
                 'Bank OpenRCA': 0.10, 'Market OpenRCA': 0.18, 'Telecom OpenRCA': 0.12}[cat]
    cost_std = cost_mean * 0.5
    cost_vals = np.clip(np.random.normal(cost_mean, cost_std, n_samples), 0, 0.55)
    
    runtime_data['Category'].extend([cat] * n_samples)
    runtime_data['Runtime'].extend(runtime_vals)
    runtime_data['Method'].extend([method] * n_samples)
    
    cost_data['Category'].extend([cat] * n_samples)
    cost_data['Cost'].extend(cost_vals)
    cost_data['Method'].extend([method] * n_samples)

df_runtime = pd.DataFrame(runtime_data)
df_cost = pd.DataFrame(cost_data)

# 拆分数据
df_runtime_original = df_runtime[df_runtime['Method'] == 'Original'].copy()
df_cost_original = df_cost[df_cost['Method'] == 'Original'].copy()
df_runtime_openrca = df_runtime[df_runtime['Method'] == 'OpenRCA'].copy()
df_cost_openrca = df_cost[df_cost['Method'] == 'OpenRCA'].copy()

# 类别顺序
original_order = ['Bank', 'Market', 'Telecom']
openrca_order = ['Bank OpenRCA', 'Market OpenRCA', 'Telecom OpenRCA']
palette = {"Original": "#f5f5bd", "OpenRCA": "#c1bed6"}

# 创建2行2列子图
fig, ((ax1_original, ax2_original), (ax1_openrca, ax2_openrca)) = plt.subplots(
    2, 2, figsize=(16, 10), sharex='col'
)

# ---------------------- 第1行：Original组 ----------------------
# 左：Original Runtime（保留y轴刻度标签）
sns.boxplot(data=df_runtime_original, x='Runtime', y='Category', hue='Method', 
            order=original_order, palette=palette, width=0.5, ax=ax1_original, legend=False)
sns.stripplot(data=df_runtime_original, x='Runtime', y='Category', hue='Method',
              order=original_order, palette=palette, size=4, alpha=0.7, jitter=True, 
              linewidth=0.5, ax=ax1_original, legend=False)
ax1_original.set_xlabel('Runtime (s)', fontsize=12)
ax1_original.set_ylabel('')  # 移除Category文字标签
ax1_original.grid(True, axis='x', alpha=0.3)
ax1_original.set_yticklabels(original_order)  # 保留Bank等刻度标签

# 右：Original Cost（隐藏y轴刻度标签，核心修改1）
sns.boxplot(data=df_cost_original, x='Cost', y='Category', hue='Method', 
            order=original_order, palette=palette, width=0.5, ax=ax2_original, legend=False)
sns.stripplot(data=df_cost_original, x='Cost', y='Category', hue='Method',
              order=original_order, palette=palette, size=4, alpha=0.7, jitter=True, 
              linewidth=0.5, ax=ax2_original, legend=False)
ax2_original.set_xlabel('Cost ($)', fontsize=12)
ax2_original.set_ylabel('')  # 移除Category文字标签
ax2_original.grid(True, axis='x', alpha=0.3)
ax2_original.set_yticklabels([])  # 隐藏Bank等所有y轴刻度标签

# ---------------------- 第2行：OpenRCA组 ----------------------
# 左：OpenRCA Runtime（保留y轴刻度标签）
sns.boxplot(data=df_runtime_openrca, x='Runtime', y='Category', hue='Method', 
            order=openrca_order, palette=palette, width=0.5, ax=ax1_openrca, legend=False)
sns.stripplot(data=df_runtime_openrca, x='Runtime', y='Category', hue='Method',
              order=openrca_order, palette=palette, size=4, alpha=0.7, jitter=True, 
              linewidth=0.5, ax=ax1_openrca, legend=False)
ax1_openrca.set_xlabel('Runtime (s)', fontsize=12)
ax1_openrca.set_ylabel('')  # 移除Category文字标签
ax1_openrca.grid(True, axis='x', alpha=0.3)
ax1_openrca.set_yticklabels(openrca_order)  # 保留Bank OpenRCA等刻度标签

# 右：OpenRCA Cost（隐藏y轴刻度标签，核心修改2）
sns.boxplot(data=df_cost_openrca, x='Cost', y='Category', hue='Method', 
            order=openrca_order, palette=palette, width=0.5, ax=ax2_openrca, legend=False)
sns.stripplot(data=df_cost_openrca, x='Cost', y='Category', hue='Method',
              order=openrca_order, palette=palette, size=4, alpha=0.7, jitter=True, 
              linewidth=0.5, ax=ax2_openrca, legend=False)
ax2_openrca.set_xlabel('Cost ($)', fontsize=12)
ax2_openrca.set_ylabel('')  # 移除Category文字标签
ax2_openrca.grid(True, axis='x', alpha=0.3)
ax2_openrca.set_yticklabels([])  # 隐藏Bank OpenRCA等所有y轴刻度标签

# 全局图例
handles = [
    plt.Line2D([0], [0], color=palette['Original'], lw=4, label='Original'),
    plt.Line2D([0], [0], color=palette['OpenRCA'], lw=4, label='OpenRCA')
]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.94, 0.98), title='Method', fontsize=11)

# 调整布局
plt.tight_layout(rect=[0, 0, 0.95, 1])

# 保存+显示
plt.savefig("final_clean_figure.pdf", bbox_inches='tight')
plt.savefig("final_clean_figure.png", dpi=300, bbox_inches='tight')
plt.show()
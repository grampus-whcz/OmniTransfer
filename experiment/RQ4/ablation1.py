import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 数据准备 ----------------------
groups = [r'$Easy$', r'$Mid$', r'$Hard$']
models = ['M', 'M1', 'M2']
new_legend_names = ['$ClusterRCA$', '$ClusterRCA_{w/o\ LLM-inferred\ log\ context}$', '$ClusterRCA_{w\ raw\ log\ context}$']

deepseek_data = {
    'M': [0.2609, 0.4583, 0.6667],
    'M1': [0.2309, 0.4167, 0.5833],
    'M2': [0.1773, 0.3583, 0.5133]
}

gpt_data = {
    'M': [0.2876, 0.4779, 0.6993],
    'M1': [0.2609, 0.4167, 0.5833],
    'M2': [0.1973, 0.3383, 0.4633]
}

styles = {
    'M': {'hatch': '///', 'color': 'white', 'edgecolor': 'black'},
    'M1': {'hatch': '...', 'color': 'white', 'edgecolor': 'black'},
    'M2': {'hatch': 'xxx', 'color': 'white', 'edgecolor': 'black'}
}

bar_width = 0.15
group_gap = 0.75
x = np.arange(len(groups)) * group_gap

# ---------------------- 2. 全局字号设置 ----------------------
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 20,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

show_yticks = [0.2, 0.5, 0.7]
y_lim = (0.1, 0.75)

# ---------------------- 3. 绘图设置 ----------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

# ---------------------- 4. 绘制 deepseek 子图 ----------------------
ax1.set_ylabel('deepseek', rotation=0, labelpad=20, ha='center', va='center', fontsize=18)
ax1.set_ylim(y_lim)
ax1.set_yticks(show_yticks)
ax1.tick_params(axis='y', labelsize=12)
for i, model in enumerate(models):
    ax1.bar(x + i * bar_width, deepseek_data[model], bar_width, 
            label=new_legend_names[i], **styles[model])

# ---------------------- 5. 绘制 gpt-4o 子图 ----------------------
ax2.set_ylabel('gpt-4o', rotation=0, labelpad=20, ha='center', va='center', fontsize=18)
ax2.set_ylim(y_lim)
ax2.set_yticks(show_yticks)
ax2.tick_params(axis='y', labelsize=12)
ax2.set_xticks(x + (len(models)-1)*bar_width/2)
ax2.set_xticklabels(groups)
ax2.tick_params(axis='x', labelsize=20)
for i, model in enumerate(models):
    ax2.bar(x + i * bar_width, gpt_data[model], bar_width, 
            **styles[model])

# ---------------------- 6. 添加数据标签（核心修改） ----------------------
def add_labels(ax, data):
    # 去掉所有条件判断，为所有柱子添加标签
    for i, model in enumerate(models):
        for j, val in enumerate(data[model]):
            ax.text(x[j] + i * bar_width, val + 0.005, f'{val:.2f}', 
                    ha='center', va='bottom', fontsize=10)

add_labels(ax1, deepseek_data)
add_labels(ax2, gpt_data)

# ---------------------- 7. 图例 ----------------------
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5), 
           ncol=1, frameon=False, fontsize=12)

# ---------------------- 8. 保存与显示 ----------------------
plt.tight_layout(rect=[0, 0, 0.85, 0.95])  
plt.savefig('ablation1.pdf', dpi=300, bbox_inches='tight')
plt.show()
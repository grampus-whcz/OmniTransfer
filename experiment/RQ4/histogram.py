import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 数据准备 ----------------------
groups = [r'$\mathcal{A}_s$', r'$\mathcal{A}_p$', r'$\mathcal{A}_n$']
models = ['M', 'M1', 'M2', 'M3', 'M4', 'M5']

mif1_data = {
    'M': [0.72, 0.74, 0.78],
    'M1': [0.71, 0.75, 0.77],
    'M2': [0.70, 0.70, 0.76],
    'M3': [0.25, 0.31, 0.37],
    'M4': [0.76, 0.77, 0.78],
    'M5': [0.71, 0.70, 0.71]
}

maf1_data = {
    'M': [0.79, 0.79, 0.79],
    'M1': [0.72, 0.78, 0.72],
    'M2': [0.70, 0.74, 0.76],
    'M3': [0.66, 0.54, 0.54],
    'M4': [0.78, 0.80, 0.79],
    'M5': [0.72, 0.75, 0.71]
}

styles = {
    'M': {'hatch': '///', 'color': 'white', 'edgecolor': 'black'},
    'M1': {'hatch': '...', 'color': 'white', 'edgecolor': 'black'},
    'M2': {'hatch': 'xxx', 'color': 'white', 'edgecolor': 'black'},
    'M3': {'hatch': '+++', 'color': 'white', 'edgecolor': 'black'},
    'M4': {'hatch': 'ooo', 'color': 'white', 'edgecolor': 'black'},
    'M5': {'hatch': '\\\\\\', 'color': 'white', 'edgecolor': 'black'}
}

bar_width = 0.15
x = np.arange(len(groups))

# ---------------------- 2. 全局字号设置（可选，统一调整） ----------------------
plt.rcParams.update({
    'font.size': 12,  # 全局基础字号
    'axes.labelsize': 14,  # 轴标签字号
    'axes.titlesize': 16,  # 轴标题字号
    'xtick.labelsize': 20,  # x轴刻度字号
    'ytick.labelsize': 12,  # y轴刻度字号
    'legend.fontsize': 12,  # 图例字号
    'figure.titlesize': 16  # 图表总标题字号
})

# ---------------------- 3. 绘图设置（预留图例空间） ----------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
# 总标题字号调大
# fig.suptitle('Figure 6: Contributions of the multimodal data', fontsize=16, y=0.95)

# ---------------------- 4. 绘制 MiF1 子图 ----------------------
# y轴标签字号调大（14号）
ax1.set_ylabel('MiF1', rotation=0, labelpad=20, ha='center', va='center', fontsize=18)
ax1.set_ylim(0.2, 0.95)
ax1.set_yticks([0.25, 0.75, 0.9])
# y轴刻度字号调大（12号）
ax1.tick_params(axis='y', labelsize=12)
for i, model in enumerate(models):
    ax1.bar(x + i * bar_width, mif1_data[model], bar_width, 
            label=model, **styles[model])

# ---------------------- 5. 绘制 MaF1 子图 ----------------------
ax2.set_ylabel('MaF1', rotation=0, labelpad=20, ha='center', va='center', fontsize=18)
ax2.set_ylim(0.2, 0.95)
ax2.set_yticks([0.25, 0.75, 0.9])
ax2.tick_params(axis='y', labelsize=12)
ax2.set_xticks(x + bar_width * 2.5)
ax2.set_xticklabels(groups)
# x轴刻度字号调大（12号）
ax2.tick_params(axis='x', labelsize=20)
for i, model in enumerate(models):
    ax2.bar(x + i * bar_width, maf1_data[model], bar_width, 
            **styles[model])

# ---------------------- 6. 添加数据标签（调大字号到10） ----------------------
def add_labels(ax, data):
    for i, model in enumerate(models):
        for j, val in enumerate(data[model]):
            if val >= 0.6:
                # 数据标签字号调大到10
                ax.text(x[j] + i * bar_width, val + 0.005, f'{val:.2f}', 
                        ha='center', va='bottom', fontsize=10)

add_labels(ax1, mif1_data)
add_labels(ax2, maf1_data)

# ---------------------- 7. 图例：整个图右侧垂直中间（字号12） ----------------------
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5), 
           ncol=1, frameon=False, fontsize=12)

# ---------------------- 8. 保存与显示 ----------------------
plt.tight_layout(rect=[0, 0, 0.85, 0.95])  
plt.savefig('multimodal_contributions.pdf', dpi=300, bbox_inches='tight')
plt.show()
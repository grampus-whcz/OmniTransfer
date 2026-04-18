import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 数据准备 ----------------------
groups = [r'$Easy$', r'$Mid$', r'$Hard$']
models = ['M', 'M1', 'M2', 'M3']
new_legend_names = ['$ClusTopoRCA$', '$ClusTopoRCA_{w/o\ topo}$', '$ClusTopoRCA_{w/o\ clustering}$', '$RCA\\text{-}agent$']

deepseek_data = {
    'M': [0.3225, 0.5087, 0.6470],
    'M1': [0.2958, 0.4385, 0.5294],
    'M2': [0.1290, 0.2631, 0.3529],
    'M3': [0.2258, 0.3859, 0.5294]
}

gpt_data = {
    'M': [0.3548, 0.5263, 0.7058],
    'M1': [0.3225, 0.5087, 0.7058],
    'M2': [0.1290, 0.1929, 0.3529],
    'M3': [0.1774, 0.5438, 0.4705]
}

# color schemes
# styles = {
#     'M': {'hatch': '///', 'color': 'white', 'edgecolor': 'black'},
#     'M1': {'hatch': '...', 'color': 'white', 'edgecolor': 'black'},
#     'M2': {'hatch': 'xxx', 'color': 'white', 'edgecolor': 'black'},
#     'M3': {'hatch': '|||', 'color': 'white', 'edgecolor': 'black'}
# }
# styles = {
#     'M': {'hatch': '///', 'color': '#AED6F1', 'edgecolor': 'black'},
#     'M1': {'hatch': '...', 'color': '#D6EAF8', 'edgecolor': 'black'},
#     'M2': {'hatch': 'xxx', 'color': '#F5B7B1', 'edgecolor': 'black'},
#     'M3': {'hatch': '|||', 'color': '#E5E8E8', 'edgecolor': 'black'}
# }

# styles = {
#     'M': {'hatch': '///', 'color': '#DDDDDD', 'edgecolor': 'black'},
#     'M1': {'hatch': '...', 'color': '#BBBBBB', 'edgecolor': 'black'},
#     'M2': {'hatch': 'xxx', 'color': '#888888', 'edgecolor': 'black'},
#     'M3': {'hatch': '|||', 'color': '#EEEEEE', 'edgecolor': 'black'}
# }

styles = {
    'M': {'hatch': '///', 'color': "#BBD9E4", 'edgecolor': 'black'},
    'M1': {'hatch': '...', 'color': "#FEF8DB", 'edgecolor': 'black'},
    'M2': {'hatch': 'xxx', 'color': "#C0E8C0", 'edgecolor': 'black'},
    'M3': {'hatch': '|||', 'color': "#FFDBE1", 'edgecolor': 'black'}
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
ax1.set_ylabel('deepseek-\nr1-0528', rotation=0, labelpad=20, ha='center', va='center', fontsize=13)
ax1.set_ylim(y_lim)
ax1.set_yticks(show_yticks)
ax1.tick_params(axis='y', labelsize=12)
for i, model in enumerate(models):
    ax1.bar(x + i * bar_width, deepseek_data[model], bar_width, 
            label=new_legend_names[i], **styles[model])

# ---------------------- 5. 绘制 gpt-4o 子图 ----------------------
ax2.set_ylabel('Gemini\n2.5 Pro', rotation=0, labelpad=20, ha='center', va='center', fontsize=13)
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
plt.savefig('ablation_ClusTopo.pdf', dpi=300, bbox_inches='tight')
plt.show()
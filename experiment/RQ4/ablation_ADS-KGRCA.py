import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 数据准备 ----------------------
groups = [r'$Easy$', r'$Mid$', r'$Hard$']
models = ['M', 'M1', 'M2', 'M3']
new_legend_names = ['$\\text{ADS-KGRCA}$', '$\\text{ADS-KGRCA}_{w/o\ KG}$', '$\\text{ADS-KGRCA}_{w/o\ cl}$', '$\\text{RCA-agent}$']

deepseek_data = {
    'M': [0.2903, 0.4912, 0.5294], # /root/shared-nvme/work/agent/ADS-KGRCA/experiments/Bank/deepseek-r1-0528/Bank_adskg_tog_llm_deepseek-r1-0528_difficulty_summary.csv
    'M1': [0.2951, 0.4386, 0.5294], # /root/shared-nvme/work/agent/OpenRCA/experiments/Bank/deepseek-r1-0528/Bank_c3_deepseek-r1-0528_difficulty_summary.csv
    'M2': [0.1452, 0.1930, 0.2353], # /root/shared-nvme/work/agent/ADS-KGRCA/experiments/Bank/deepseek-r1-0528/Bank_adskg_tog_llm_ablation_no_clustering_deepseek-r1-0528_difficulty_summary.csv
    'M3': [0.2258, 0.3860, 0.5294] # /root/shared-nvme/work/agent/RCA-agent/RCA-agent/experiments_original/Bank/deepseek-r1-250528/Bank_original_deepseek-r1-250528_difficulty_summary.csv
}

# gemini
# gpt_data = {
#     'M': [0.3226, 0.5614, 0.5882], # /root/shared-nvme/work/agent/ADS-KGRCA/experiments/Bank/gemini-2.5-pro/Bank_adskg_tog_llm_gemini-2.5-pro_difficulty_summary.csv
#     'M1': [0.3225, 0.5087, 0.7058], # /root/shared-nvme/work/agent/OpenRCA/experiments/Bank/gemini-2.5-pro-preview-p/Bank_c3_gemini-2.5-pro-preview-p_difficulty_summary.csv
#     'M2': [0.1290, 0.1929, 0.3529], # no cash
#     'M3': [0.1774, 0.5438, 0.4705] # /root/shared-nvme/work/agent/RCA-agent/RCA-agent/experiments_original/Bank/gemini-2.5-pro-preview-p/Bank_original_gemini-2.5-pro-preview-p_difficulty_summary.csv
# }

# glm-4.5
gpt_data = {
    'M': [0.2903, 0.5088, 0.5294], #/root/shared-nvme/work/agent/ADS-KGRCA/experiments/Bank/glm-4.5/Bank_adskg_tog_llm_glm-4.5_difficulty_summary.csv
    'M1': [0.2951, 0.4464, 0.5294], # /root/shared-nvme/work/agent/ADS-KGRCA/experiments/Bank/glm-4.5/Bank_adskg_tog_llm_ablation_clustering_3_no_ToG-EE_glm-4.5_difficulty_summary.csv
    'M2': [0.1579, 0.1800, 0.5882], # /root/shared-nvme/work/agent/ADS-KGRCA/experiments/Bank/glm-4.5/Bank_adskg_tog_llm_ablation_no_clustering_glm-4.5_difficulty_summary.csv
    'M3': [0.0323, 0.2632, 0.5294] # /root/shared-nvme/work/agent/RCA-agent/RCA-agent/experiments_original/Bank/glm-4.5/Bank_original_glm-4.5_difficulty_summary.csv
}

styles = {
    'M':  {'hatch': '///', 'color': "#D4E6F1", 'edgecolor': 'black'},
    'M1': {'hatch': '...', 'color': "#FCF3CF", 'edgecolor': 'black'},
    'M2': {'hatch': 'xxx', 'color': "#D5F5E3", 'edgecolor': 'black'},
    'M3': {'hatch': '|||', 'color': "#FADBD8", 'edgecolor': 'black'}
}

bar_width = 0.15
group_gap = 0.75
x = np.arange(len(groups)) * group_gap

# ---------------------- 2. 全局字号设置 ----------------------
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 20,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

show_yticks = [0.2, 0.5, 0.7]
y_lim = (0.1, 0.75) # Y轴下限设为0.1，这会导致0.03的数据看起来像是“悬空”或截断，需要特别注意

# ---------------------- 3. 绘图设置 ----------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

# ---------------------- 4. 绘制 deepseek 子图 ----------------------
ax1.set_ylabel('deepseek-\nr1-0528', rotation=0, labelpad=20, ha='center', va='center', fontsize=13)
ax1.set_ylim(y_lim)
ax1.set_yticks(show_yticks)
for i, model in enumerate(models):
    ax1.bar(x + i * bar_width, deepseek_data[model], bar_width,
            label=new_legend_names[i], **styles[model])

# ---------------------- 5. 绘制 glm-4.5 子图 ----------------------
ax2.set_ylabel('GLM-4.5', rotation=0, labelpad=20, ha='center', va='center', fontsize=13)
ax2.set_ylim(y_lim)
ax2.set_yticks(show_yticks)
ax2.set_xticks(x + (len(models)-1)*bar_width/2)
ax2.set_xticklabels(groups)
for i, model in enumerate(models):
    ax2.bar(x + i * bar_width, gpt_data[model], bar_width,
            **styles[model])

# ---------------------- 6. 添加数据标签（核心修正部分） ----------------------
def add_labels(ax, data, threshold=0.12):
    """
    ax: 坐标轴对象
    data: 数据字典
    threshold: 判定为“过小”的阈值。由于Y轴从0.1开始，低于0.12的值都需要特殊处理以防重叠
    """
    for i, model in enumerate(models):
        for j, val in enumerate(data[model]):
            x_pos = x[j] + i * bar_width

            # 逻辑判断：如果数值很小（接近或低于Y轴下限），强制抬高标签位置
            if val < threshold:
                # 对于极小值，固定显示在 Y=0.12 的位置（即Y轴起始线上方一点点）
                # 或者使用 val + 固定偏移量，确保它浮在空中
                y_pos = 0.12
                vertical_align = 'bottom'
            else:
                # 正常数值，显示在柱子顶端上方
                y_pos = val + 0.015
                vertical_align = 'bottom'

            ax.text(x_pos, y_pos, f'{val:.2f}',
                    ha='center', va=vertical_align, fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8)) # 增加白色背景框防止看不清

add_labels(ax1, deepseek_data)
add_labels(ax2, gpt_data)

# ---------------------- 7. 图例 ----------------------
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
           ncol=4, frameon=False, fontsize=12)

# ---------------------- 8. 保存与显示 ----------------------
# rect调整布局，给顶部的图例留出空间
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('0.ablation_ADS-KGRCA.pdf', dpi=300, bbox_inches='tight')
plt.show()
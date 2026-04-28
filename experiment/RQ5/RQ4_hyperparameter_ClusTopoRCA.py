import matplotlib.pyplot as plt
import numpy as np

# 使用默认字体，避免 Arial 报错
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# 原始 reduction values (用于标签)
reduction_vals = [1, 2, 3, 4, 5, 6]
x = np.arange(len(reduction_vals)) 

# 数据
y_blue_n = np.array([0.2742, 0.2833, 0.2581, 0.3065, 0.3065, 0.3065])
y_red_n = np.array([0.0323, 0.0333, 0.0323, 0.0323, 0.0323, 0.0323])
y_green_n = np.array([0.3065, 0.3167, 0.2903, 0.3387, 0.3387, 0.3387])

y_blue_s = np.array([0.1053, 0.1091, 0.1404, 0.1578, 0.1228, 0.1404])
y_red_s = np.array([0.3860, 0.4364, 0.3860, 0.4211, 0.4035, 0.3684])
y_green_s = np.array([0.4913, 0.5455, 0.5264, 0.5789, 0.5263, 0.5088])

y_blue_p = np.array([0, 0, 0.0588, 0, 0, 0])
y_red_p = np.array([0.7059, 0.7059, 0.5294, 0.7059, 0.7059, 0.5882])
y_green_p = np.array([0.7059, 0.7059, 0.5882, 0.7059, 0.7059, 0.5882])


label_fontsize = 20  # 设置字体大小

# 创建图形
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)  # 增加宽度以便为图例留出空间

# ********** 关键修改1：定义要显示的刻度（仅0、0.5、0.75）**********
show_yticks = [0, 0.5, 0.75]
# ********** 关键修改2：定义y轴范围（上下端点-0.1到0.85）**********
y_lim = (-0.1, 0.85)

# 定义颜色
color_blue = '#2E75B6'
color_red = "#C27566"
color_green = "#66c274"

# 子图 3: An
ax1.plot(x, y_blue_n, color=color_blue, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax1.plot(x, y_red_n, color=color_red, linestyle=':', marker='^', markersize=15, linewidth=1.5)
ax1.plot(x, y_green_n, color=color_green, linestyle='-.', marker='.', markersize=15, linewidth=1.5)
ax1.set_ylabel(r'$Easy$', rotation=0, labelpad=25, va='center', fontsize=label_fontsize, fontweight='bold')
ax1.set_ylim(y_lim)
ax1.set_yticks(show_yticks)
ax1.tick_params(axis='y', labelsize=12)
# ax3.grid(True, alpha=0.3)

# 子图 1: As
ax2.plot(x, y_blue_s, color=color_blue, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax2.plot(x, y_red_s, color=color_red, linestyle=':', marker='^', markersize=15, linewidth=1.5)
ax2.plot(x, y_green_s, color=color_green, linestyle='-.', marker='.', markersize=15, linewidth=1.5)
ax2.set_ylabel(r'$Mid$', rotation=0, labelpad=25, va='center', fontsize=label_fontsize, fontweight='bold')
ax2.set_ylim(y_lim)
ax2.set_yticks(show_yticks)
ax2.tick_params(axis='y', labelsize=12)
# ax1.grid(True, alpha=0.3)

# 子图 2: Ap
ax3.plot(x, y_blue_p, color=color_blue, linestyle='--', marker='*', markersize=15, linewidth=1.5)
ax3.plot(x, y_red_p, color=color_red, linestyle=':', marker='^', markersize=15, linewidth=1.5)
ax3.plot(x, y_green_p, color=color_green, linestyle='-.', marker='.', markersize=15, linewidth=1.5)
ax3.set_ylabel(r'$Hard$', rotation=0, labelpad=25, va='center', fontsize=label_fontsize, fontweight='bold')
ax3.set_ylim(y_lim)
ax3.set_yticks(show_yticks)
ax3.tick_params(axis='y', labelsize=12)
# ax2.grid(True, alpha=0.3)

ax3.set_xticks(x)
ax3.set_xticklabels([str(v) for v in reduction_vals])
ax3.set_xlabel('min samples (DBSCAN Clustering)', fontsize=label_fontsize)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)

# 图例：MaF1 / MiF1
handles = [
    plt.Line2D([], [], color='#2E75B6', linestyle='--', marker='*', markersize=15, linewidth=1.5),
    plt.Line2D([], [], color='#C55A11', linestyle=':', marker='^', markersize=15, linewidth=1.5),
    plt.Line2D([], [], color='#66c274', linestyle='-.', marker='.', markersize=15, linewidth=1.5)
]
# labels = ['CORRECT', 'PARTIAL', 'TOTAL']

labels = [
    r'$\mathbf{C}$ORRECT',
    r'$\mathbf{P}$ARTIAL',
    r'$\mathbf{T}$OTAL'
]

fig.legend(
    handles, labels,
    loc='center left',
    bbox_to_anchor=(0.86, 0.5),
    frameon=False,
    handlelength=2,
    fontsize=15
)

plt.tight_layout()
plt.subplots_adjust(right=0.86)  # 留出刚好容纳图例的空间
plt.savefig('hyperparameter.pdf', format='pdf', bbox_inches='tight')
plt.show()
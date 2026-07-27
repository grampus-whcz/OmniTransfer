import matplotlib.pyplot as plt
import numpy as np

# 使用默认字体，避免 Arial 报错
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# 原始 reduction values (用于标签)
reduction_vals = [3, 4, 5, 6, 7, 8]
x = np.arange(len(reduction_vals)) 

# 数据
y_blue_n = np.array([0.2581, 0.2581, 0.2581, 0.2418, 0.2742, 0.3064])
y_red_n = np.array([0.0323, 0.0323, 0.0323, 0.0323, 0.0323, 0.0323])
y_green_n = np.array([0.2904, 0.2904, 0.2904, 0.2742, 0.3065, 0.3387])

y_blue_s = np.array([0.0877, 0.1053, 0.0877, 0.0526, 0.0526, 0.0877])
y_red_s = np.array([0.4211, 0.3860, 0.4561, 0.4035, 0.4386, 0.4211])
y_green_s = np.array([0.5088, 0.4913, 0.5438, 0.4561, 0.4912, 0.5088])

y_blue_p = np.array([0, 0.0588, 0.1176, 0, 0, 0.0588])
y_red_p = np.array([0.5294, 0.4706, 0.4706, 0.5882, 0.5294, 0.4706])
y_green_p = np.array([0.5294, 0.5294, 0.5882, 0.5882, 0.5294, 0.5294])


label_fontsize = 20  # 设置字体大小

# 创建图形
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)  # 增加宽度以便为图例留出空间

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
ax3.set_xlabel('ToG-EE Search depth and width (depth=width)', fontsize=label_fontsize)
ax3.tick_params(axis='x', labelsize=12)
ax3.tick_params(axis='y', labelsize=12)

# 图例：MaF1 / MiF1
handles = [
    plt.Line2D([], [], color='#2E75B6', linestyle='--', marker='*', markersize=15, linewidth=1.5),
    plt.Line2D([], [], color='#C55A11', linestyle=':', marker='^', markersize=15, linewidth=1.5),
    plt.Line2D([], [], color='#66c274', linestyle='-.', marker='.', markersize=15, linewidth=1.5)
]

labels = [
    r'$\mathbf{C}$ORRECT',
    r'$\mathbf{P}$ARTIAL',
    r'$\mathbf{T}$OTAL'
]

# --- 修改点 1：调整图例位置 ---
fig.legend(
    handles, labels,
    loc='upper center',      # 保持居中
    ncol=3,                  # 横向排列
    frameon=False,
    fontsize=15,
    columnspacing=2,
    # 关键：将 Y 轴坐标从 1.0 提到 1.05 或更高，使其移出坐标轴范围
    bbox_to_anchor=(0.5, 1.08) 
)

# --- 修改点 2：调整布局 ---
# 先执行 tight_layout 自动调整子图间距
plt.tight_layout()

# 再手动调整顶部边距 (top)，给图例留出空间
# 0.92 表示子图区域只占据画布底部 92% 的高度，顶部 8% 留给图例
plt.subplots_adjust(top=0.99) 

# 保存与显示
plt.savefig('1.hyperparamete-dw.pdf', format='pdf', bbox_inches='tight')
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 生成横坐标数据 (0 到 10 之间取 50 个点)
x = np.linspace(0, 10, 50)

# 创建画布，设置背景透明
plt.figure(figsize=(18, 4), facecolor='none')

# =======================
# 图 1: 正常波动 (无异常)
# =======================
y1_blue = np.sin(x) + 0.3 * np.random.randn(len(x))
y1_orange = np.cos(x) + 0.3 * np.random.randn(len(x))

ax1 = plt.subplot(1, 3, 1)
ax1.set_facecolor('none')

plt.plot(x, y1_blue, color='blue', linewidth=2)
plt.plot(x, y1_orange, color='orange', linewidth=2)

# 隐藏刻度标签
ax1.set_xticks([])
ax1.set_yticks([])

# 样式调整：只显示左边和下边的数轴，并加粗
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('black')
ax1.spines['bottom'].set_color('black')
# 【修改点】线宽从 1 改为 2.5
ax1.spines['left'].set_linewidth(2.5)
ax1.spines['bottom'].set_linewidth(2.5)

plt.xlim(0, 10)

# =======================
# 图 2: 带异常点 (绿色 + 橙色)
# =======================
y2_green = np.sin(x) + 0.3 * np.random.randn(len(x))
y2_orange = np.cos(x) + 0.3 * np.random.randn(len(x))

# 制造异常：在第 30 个点人为增加一个尖峰
peak_idx = 30
y2_orange[peak_idx] += 2.5 

ax2 = plt.subplot(1, 3, 2)
ax2.set_facecolor('none')

plt.plot(x, y2_green, color='green', linewidth=2)
plt.plot(x, y2_orange, color='orange', linewidth=2)

# 添加红色三角形标记异常点
plt.scatter(x[peak_idx], y2_orange[peak_idx], color='red', s=150, marker='^', zorder=5, edgecolors='black')
# 添加警告文本
# plt.text(x[peak_idx], y2_orange[peak_idx] + 0.4, '⚠️', ha='center', va='bottom', fontsize=20, color='red')

# 隐藏刻度标签
ax2.set_xticks([])
ax2.set_yticks([])

# 样式调整：加粗数轴
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('black')
ax2.spines['bottom'].set_color('black')
ax2.spines['left'].set_linewidth(2.5)
ax2.spines['bottom'].set_linewidth(2.5)

plt.xlim(0, 10)

# =======================
# 图 3: 带异常点 (黑色 + 橙色)
# =======================
y3_black = np.sin(x) + 0.3 * np.random.randn(len(x)) - 0.5 
y3_orange = np.cos(x) + 0.3 * np.random.randn(len(x))

# 同样在第 30 个点制造异常
y3_orange[peak_idx] += 2.5

ax3 = plt.subplot(1, 3, 3)
ax3.set_facecolor('none')

plt.plot(x, y3_black, color='black', linewidth=2)
plt.plot(x, y3_orange, color='orange', linewidth=2)

# 添加红色三角形标记异常点
plt.scatter(x[peak_idx], y3_orange[peak_idx], color='red', s=150, marker='^', zorder=5, edgecolors='black')
# 添加警告文本
# plt.text(x[peak_idx], y3_orange[peak_idx] + 0.4, '⚠️', ha='center', va='bottom', fontsize=20, color='red')

# 隐藏刻度标签
ax3.set_xticks([])
ax3.set_yticks([])

# 样式调整：加粗数轴
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color('black')
ax3.spines['bottom'].set_color('black')
ax3.spines['left'].set_linewidth(2.5)
ax3.spines['bottom'].set_linewidth(2.5)

plt.xlim(0, 10)

# 调整子图间距
plt.subplots_adjust(wspace=0.3, left=0.05, right=0.95, bottom=0.15, top=0.9)

# 保存为透明背景的 PNG 文件
output_filename = 'anomaly_charts_thick_axes.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', format='png', transparent=True)
print(f"✅ 粗轴透明背景图片已成功保存为: {output_filename}")

# 显示图表
plt.show()
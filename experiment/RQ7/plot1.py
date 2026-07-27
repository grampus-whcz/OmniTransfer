import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# ===================== 全局配置 =====================
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# 全局刻度字号统一放大
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

FIG_SIZE = (12, 11)
CMAP = 'viridis'
SEED = 42
np.random.seed(SEED)

# ===================== 3D高斯绘制工具【标题字号放大至20】 =====================
def plot_3d_gaussian(ax, mean, cov, scale=1.0, xlabel="", ylabel="", title="", title_color="green"):
    x = np.linspace(-4, 4, 120)
    y = np.linspace(-4, 4, 120)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean, cov)
    Z = scale * rv.pdf(pos)

    surf = ax.plot_surface(X, Y, Z, cmap=CMAP, linewidth=0, antialiased=True, alpha=0.9)
    ax.contourf(X, Y, Z, zdir='z', offset=-0.03, cmap=CMAP, alpha=0.4)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_zlabel('Joint', fontsize=14)
    # 关键：增大标题字体 fontsize=20，控制 NORMAL / ANOMALY DISRUPTION 文字大小
    ax.set_title(title, fontsize=20, weight='bold', color=title_color)
    ax.set_zlim(-0.03, Z.max() * 1.1)
    ax.grid(True, alpha=0.25)
    return ax

# ===================== 散点数据生成 =====================
def gen_normal_scatter(n=400):
    mean = [0, 0]
    cov = [[0.8, 0.75], [0.75, 1.0]]
    data = multivariate_normal.rvs(mean=mean, cov=cov, size=n)
    return data[:, 0], data[:, 1]

def gen_two_separate_clusters(n_normal=400, n_anom=180):
    x_norm, y_norm = gen_normal_scatter(n_normal)
    # 红色异常簇：向右、向下调整
    mean_anom = [4.0, 3.2]
    cov_anom = [[0.5, 0.15], [0.15, 0.7]]
    data_anom = multivariate_normal.rvs(mean=mean_anom, cov=cov_anom, size=n_anom)
    x_anom, y_anom = data_anom[:, 0], data_anom[:, 1]
    return x_norm, y_norm, x_anom, y_anom

# ===================== 创建画布 =====================
fig = plt.figure(figsize=FIG_SIZE)

# fig.suptitle("JOINT PROBABILITY MODEL\n(sdb_pool_usage, sdb_latency, sapi_latency)",
#              fontsize=22, weight='bold', y=0.97,
#              bbox=dict(boxstyle="round,pad=0.4", facecolor="none", edgecolor="none"))

# ---------------------- 上排 3D曲面 ----------------------
# 左上：NORMAL 标题字体已放大
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
mean_norm_3d = [0, 0]
cov_norm_3d = [[0.8, 0.72], [0.72, 1.0]]
plot_3d_gaussian(ax1, mean_norm_3d, cov_norm_3d, scale=1.2,
                 xlabel="sdb_pool_usage", ylabel="sdb_latency",
                 title="NORMAL", title_color="#007000")

# 右上：ANOMALY DISRUPTION 标题同步放大
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
x = np.linspace(-4, 4, 120)
y = np.linspace(-4, 4, 120)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
rv_peak1 = multivariate_normal([0, 0], [[0.8, 0.72], [0.72, 1.0]])
rv_peak2 = multivariate_normal([2.2, 1.4], [[0.5, 0.12], [0.12, 0.65]])
Z_mix = 1.2 * rv_peak1.pdf(pos) + 1.05 * rv_peak2.pdf(pos)

surf2 = ax2.plot_surface(X, Y, Z_mix, cmap=CMAP, linewidth=0, alpha=0.9)
ax2.contourf(X, Y, Z_mix, zdir='z', offset=-0.03, cmap=CMAP, alpha=0.4)
ax2.set_xlabel("sapi_latency", fontsize=14)
ax2.set_ylabel("sdb_latency", fontsize=14)
ax2.set_zlabel('Joint', fontsize=14)
ax2.set_title("ANOMALY DISRUPTION", fontsize=20, weight='bold', color="#aa0000")
ax2.set_zlim(-0.03, Z_mix.max() * 1.1)
ax2.grid(True, alpha=0.25)

# ---------------------- 下排 散点图（同步放大子图标题字号至20） ----------------------
# 左下：NORMAL RELATION
ax3 = fig.add_subplot(2, 2, 3)
x_scat_n, y_scat_n = gen_normal_scatter(400)
ax3.scatter(x_scat_n, y_scat_n, color="#2277bb", alpha=0.75, s=35)
ax3.set_title("NORMAL RELATION", fontsize=20, weight='bold', color="#007000")
ax3.set_xlabel("sdb_latency", fontsize=14)
ax3.set_ylabel("sapi_latency", fontsize=14)
ax3.grid(True, alpha=0.3)

# 右下：ANOMALY RELATION
ax4 = fig.add_subplot(2, 2, 4)
x_clus1, y_clus1, x_clus2, y_clus2 = gen_two_separate_clusters(400, 180)
ax4.scatter(x_clus1, y_clus1, color="#2277bb", alpha=0.75, s=35)
ax4.scatter(x_clus2, y_clus2, color="#dd2222", alpha=0.9, s=48)
ax4.annotate("shifting ratio", xy=(3.6, 2.9), xytext=(0.5, 4.2),
             arrowprops=dict(arrowstyle="->", color="black", lw=2.2),
             fontsize=13)
ax4.set_title("ANOMALY RELATION\n(Failure)", fontsize=20, weight='bold', color="#aa0000")
ax4.set_xlabel("sdb_latency", fontsize=14)
ax4.set_ylabel("sapi_latency", fontsize=14)
ax4.grid(True, alpha=0.3)

# 中间水平分割虚线
# fig.lines.extend([plt.Line2D([0.04, 0.96], [0.495, 0.495], transform=fig.transFigure,
#                              linestyle="--", color="black", alpha=0.6, lw=1.2)])

plt.tight_layout(rect=[0, 0, 1, 0.94])

# 保存透明背景PDF
fig.savefig("joint_probability_model.pdf", dpi=300, bbox_inches="tight", transparent=True)
print("透明背景图像已保存为 joint_probability_model.pdf")

plt.savefig('joint_probability_model.png', dpi=300, bbox_inches='tight')
plt.savefig('joint_probability_model.svg', bbox_inches='tight', transparent=True)
plt.savefig('joint_probability_model.tif', dpi=600, bbox_inches='tight')

plt.show()
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 500)
np.random.seed(42)

# 各行右图控制线位置（均在视图内可见）
line_1r = 95
line_2r = 38
line_3r = 68

# ---------------------- 第一行左：连接池利用率，虚线标95%真实值 ----------------------
base_1l = 60
real_max_1l = 95
y1_left = base_1l + (real_max_1l - base_1l) * (x / 100)**1.5

plt.subplot(3, 2, 1)
plt.plot(x, y1_left, 'b-')
plt.axhline(y=real_max_1l, color='k', linestyle='--')
plt.axvline(x=70, color='k', linestyle='--')
plt.text(72, real_max_1l + 0.8, '95%', fontsize=10)
plt.ylim(55, 100)
plt.xlabel("Time", labelpad=2)  # labelpad控制文字离横轴距离，数值越小越近
plt.ylabel("Pool Utilization (%)")

# ---------------------- 第一行右：标注2 sigma ----------------------
base_1r = 80
peak_1r = 94
trend_1r = np.where(x < 50, base_1r, base_1r + (peak_1r - base_1r) * ((x - 50)/50)**1.2)
noise_1r = np.random.normal(0, 0.4, len(x))
y1_right = trend_1r + noise_1r

plt.subplot(3, 2, 2)
plt.plot(x, y1_right, 'r-')
plt.axhline(y=line_1r, color='k', linestyle='--')
plt.axvline(x=70, color='k', linestyle='--')
plt.text(72, line_1r + 0.8, '2 sigma', fontsize=10)
plt.ylim(75, 100)
plt.xlabel("Time", labelpad=2)
# plt.ylabel("Univariate Stat")

# ---------------------- 第二行左：DB延迟，虚线标35ms真实值 ----------------------
base_2l = 20
real_max_2l = 35
noise_2l = np.random.normal(0, 2, len(x))
trend_2l = np.where(x < 50, base_2l, base_2l + (real_max_2l - base_2l) * ((x - 50)/50)**1.5)
y2_left = trend_2l + noise_2l

plt.subplot(3, 2, 3)
plt.plot(x, y2_left, 'b-')
plt.axhline(y=real_max_2l, color='k', linestyle='--')
plt.axvline(x=70, color='k', linestyle='--')
plt.text(72, real_max_2l + 0.8, '35ms', fontsize=10)
plt.ylim(15, 40)
plt.xlabel("Time", labelpad=2)
plt.ylabel("DB Latency (ms)")

# ---------------------- 第二行右：标注1.5 sigma ----------------------
base_2r = 25
peak_2r = 34
noise_2r = np.random.normal(0, 1.5, len(x))
trend_2r = np.where(x < 50, base_2r, base_2r + (peak_2r - base_2r) * ((x - 50)/50)**1.5)
y2_right = trend_2r + noise_2r

plt.subplot(3, 2, 4)
plt.plot(x, y2_right, 'r-')
plt.axhline(y=line_2r, color='k', linestyle='--')
plt.axvline(x=70, color='k', linestyle='--')
plt.text(72, line_2r - 1, '1.5 sigma', fontsize=10)
plt.ylim(20, 40)
plt.xlabel("Time", labelpad=2)
# plt.ylabel("Univariate Stat")

# ---------------------- 第三行左：API延迟，虚线标65ms真实值 ----------------------
base_3l = 50
real_max_3l = 65
noise_3l = np.random.normal(0, 3, len(x))
trend_3l = np.where(x < 50, base_3l, base_3l + (real_max_3l - base_3l) * ((x - 50)/50)**1.5)
y3_left = trend_3l + noise_3l

plt.subplot(3, 2, 5)
plt.plot(x, y3_left, 'b-')
plt.axhline(y=real_max_3l, color='k', linestyle='--')
plt.axvline(x=70, color='k', linestyle='--')
plt.text(72, real_max_3l + 0.8, '65ms', fontsize=10)
plt.ylim(45, 70)
plt.xlabel("Time", labelpad=2)
plt.ylabel("API Latency (ms)")

# ---------------------- 第三行右：标注1.2 sigma ----------------------
base_3r = 55
peak_3r = 64
noise_3r = np.random.normal(0, 1, len(x))
trend_3r = np.where(x < 50, base_3r, base_3r + (peak_3r - base_3r) * ((x - 50)/50)**1.5)
y3_right = trend_3r + noise_3r

plt.subplot(3, 2, 6)
plt.plot(x, y3_right, 'r-')
plt.axhline(y=line_3r, color='k', linestyle='--')
plt.axvline(x=70, color='k', linestyle='--')
plt.text(72, line_3r - 1, '1.2 sigma', fontsize=10)
plt.ylim(50, 70)
plt.xlabel("Time", labelpad=2)
# plt.ylabel("Univariate Stat")

# ========== 调整子图间距：增大行间距hspace，缩小xlabel距离 ==========
fig = plt.gcf()
# hspace：行与行垂直间距，数值越大空隙越大；wspace左右子图间距
plt.subplots_adjust(hspace=0.45, wspace=0.22, top=0.94, bottom=0.06, left=0.08, right=0.96)

# 透明背景
fig.patch.set_alpha(0)
for ax in fig.axes:
    ax.patch.set_alpha(0)

# 多格式保存
plt.savefig('motivating_example.pdf', format='pdf', bbox_inches='tight')
plt.savefig('motivating_example.png', dpi=300, bbox_inches='tight')
plt.savefig('motivating_example.svg', bbox_inches='tight')
plt.savefig('motivating_example.tif', dpi=600, bbox_inches='tight')

plt.show()
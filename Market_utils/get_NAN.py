import numpy as np

# 加载数据
path = '/root/shared-nvme/work/timeSeries/OmniTransfer_new/dataset/data2/online_data.npy'
data = np.load(path)

# 确保是浮点类型（如果不是，转换为 float64 以安全检查）
if not np.issubdtype(data.dtype, np.floating):
    print(f"Warning: Data dtype is {data.dtype}. Converting to float64 for safe checking.")
    data = data.astype(np.float64)

# 检查 NaN
has_nan = np.isnan(data).any()

# 检查无穷大（+inf 或 -inf）
has_inf = np.isinf(data).any()

# 检查是否包含过大值（超过 float64 可表示范围）
# float64 最大有限值约为 1.8e308
# 但很多算法（如 sklearn）在内部会因数值不稳定而报错，即使未达 inf
# 所以也可以检查是否有绝对值异常大的数（可选）
max_abs = np.abs(data).max() if data.size > 0 else 0
too_large = max_abs > 1e300  # 阈值可根据实际情况调整

# 输出结果
print(f"Shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Contains NaN: {has_nan}")
print(f"Contains Inf/-Inf: {has_inf}")
print(f"Max absolute value: {max_abs:.3e}")
print(f"Potentially too large (>|1e300|): {too_large}")

if has_nan or has_inf or too_large:
    print("\n⚠️  数据包含可能导致 'ValueError: Input contains NaN, infinity or a value too large...' 的问题！")
else:
    print("\n✅ 数据看起来是数值安全的。")
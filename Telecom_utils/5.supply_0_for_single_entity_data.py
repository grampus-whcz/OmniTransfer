# 维metric_B补充数据全0

import numpy as np

file_path = '/root/shared-nvme/work/timeSeries/OmniTransfer_new/dataset/data3/offline_data.npy'

# 加载数据
data = np.load(file_path)

# 获取原始形状
orig_shape = data.shape
if len(orig_shape) != 3:
    raise ValueError(f"Expected 3D array, but got shape {orig_shape}")

n_samples, dim1, dim2 = orig_shape

# 如果第0维小于6，则扩展到6；否则不处理
if n_samples < 12:
    # 自动使用原始的后两维尺寸
    new_shape = (12, dim1, dim2)
    new_data = np.zeros(new_shape, dtype=data.dtype)
    new_data[:n_samples] = data
    np.save(file_path, new_data)
    print(f"Expanded from {orig_shape} to {new_data.shape}")
else:
    print(f"First dimension ({n_samples}) >= 12. No change made.")
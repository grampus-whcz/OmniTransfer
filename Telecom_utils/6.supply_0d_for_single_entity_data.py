# /root/shared-nvme/work/timeSeries/OmniTransfer_new/Telecom_utils/expand_entityB_to_12.py
import numpy as np
import os

base_dir = "/root/shared-nvme/work/timeSeries/OmniTransfer_new"
offline_path = os.path.join(base_dir, "dataset/data3/offline_data.npy")
online_path = os.path.join(base_dir, "dataset/data3/online_data.npy")

for path in [offline_path, online_path]:
    data = np.load(path)  # shape (1, T, F)
    assert data.shape[0] == 1, f"Expected single entity, got {data.shape[0]}"
    expanded = np.repeat(data, 12, axis=0)  # (12, T, F)
    np.save(path, expanded)
    print(f"Expanded {path} from {data.shape} to {expanded.shape}")
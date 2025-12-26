import os
import numpy as np
import glob
import re

# 数据集根目录
base_dir = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/SMD/ServerMachineDataset"

def parse_machine_id(filename):
    """
    从文件名如 'machine-1-2.txt' 中提取 (1, 2)
    """
    basename = os.path.basename(filename)
    match = re.match(r"machine-(\d+)-(\d+)\.txt", basename)
    if not match:
        raise ValueError(f"Unexpected filename format: {basename}")
    group = int(match.group(1))
    machine_id = int(match.group(2))
    return (group, machine_id)

def natural_sort_key(filename):
    return parse_machine_id(filename)

# 获取所有文件并按 (group, id) 自然排序
train_files = sorted(glob.glob(os.path.join(base_dir, "train", "machine-*.txt")), key=natural_sort_key)
test_files = sorted(glob.glob(os.path.join(base_dir, "test", "machine-*.txt")), key=natural_sort_key)
label_files = sorted(glob.glob(os.path.join(base_dir, "test_label", "machine-*.txt")), key=natural_sort_key)

# 验证数量和顺序
assert len(train_files) == len(test_files) == len(label_files) == 28, \
    f"Expected 28 files per split, got train:{len(train_files)}, test:{len(test_files)}, label:{len(label_files)}"

# 可选：打印前几个文件名确认顺序
print("First few files (should be in natural order):")
for f in train_files[:28]:
    print("  ", os.path.basename(f))
print("...")

def load_files_and_truncate(files, is_label=False):
    data_list = []
    lengths = []

    for f in files:
        if is_label:
            arr = np.loadtxt(f, dtype=np.int32)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            assert arr.ndim == 1, f"Label file {f} should be 1D"
            data_list.append(arr)
            lengths.append(arr.shape[0])
        else:
            arr = np.loadtxt(f, delimiter=',', dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            assert arr.shape[1] == 38, f"Feature file {f} has {arr.shape[1]} columns, expected 38"
            data_list.append(arr)
            lengths.append(arr.shape[0])

    min_len = min(lengths)
    # min_len = 1344
    print(f"  Min length among {len(files)} files: {min_len}")

    if is_label:
        truncated = [d[:min_len] for d in data_list]
        stacked = np.stack(truncated, axis=0)  # (28, T)
        return stacked
    else:
        truncated = [d[:min_len] for d in data_list]
        stacked = np.stack(truncated, axis=0)  # (28, T, 38)
        return stacked  # <-- 不再转置！

# --- 处理数据 ---
print("Processing offline_data (train)...")
offline_data = load_files_and_truncate(train_files, is_label=False)
np.save("offline_data.npy", offline_data)
print("✅ offline_data.npy saved, shape:", offline_data.shape)

print("Processing online_data (test)...")
online_data = load_files_and_truncate(test_files, is_label=False)
np.save("online_data.npy", online_data)
print("✅ online_data.npy saved, shape:", online_data.shape)

print("Processing label (test_label)...")
label_data = load_files_and_truncate(label_files, is_label=True)
np.save("label.npy", label_data)
print("✅ label.npy saved, shape:", label_data.shape)

print("\n🎉 All done! Files are saved with natural machine order.")
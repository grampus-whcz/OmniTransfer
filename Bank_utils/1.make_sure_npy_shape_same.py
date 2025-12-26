import os
from pathlib import Path
import numpy as np

def scan_and_check_shapes(root_dirs):
    mismatched = []

    for root_dir in root_dirs:
        root_path = Path(root_dir)
        if not root_path.exists():
            print(f"⚠️ 路径不存在，跳过: {root_path}")
            continue

        # 递归遍历所有子目录
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirpath = Path(dirpath)
            offline_file = dirpath / "offline_log.npy"
            online_file = dirpath / "online_log.npy"

            if offline_file.exists() and online_file.exists():
                try:
                    offline_arr = np.load(offline_file)
                    online_arr = np.load(online_file)

                    offline_shape = offline_arr.shape
                    online_shape = online_arr.shape

                    if offline_shape != online_shape:
                        mismatched.append({
                            'dir': str(dirpath),
                            'offline_shape': offline_shape,
                            'online_shape': online_shape
                        })
                        print(f"❌ 不匹配: {dirpath}")
                        print(f"    offline_log.npy: {offline_shape}")
                        print(f"    online_log.npy:  {online_shape}")
                    else:
                        print(f"✅ 匹配: {dirpath} -> {offline_shape}")

                except Exception as e:
                    print(f"⚠️ 加载失败: {dirpath} | 错误: {e}")

    return mismatched

if __name__ == "__main__":
    roots = [
        "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Market_utils",
        "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Telecom_utils"
    ]

    print("🔍 开始扫描目录，检查 offline_log.npy 与 online_log.npy 的 shape 是否一致...\n")

    mismatches = scan_and_check_shapes(roots)

    print("\n" + "="*80)
    if mismatches:
        print(f"🚨 发现 {len(mismatches)} 个目录中 shape 不一致：")
        for m in mismatches:
            print(f"\n目录: {m['dir']}")
            print(f"  offline_data.npy shape: {m['offline_shape']}")
            print(f"  online_data.npy shape:  {m['online_shape']}")
    else:
        print("🎉 所有目录中的 offline_data.npy 和 online_data.npy shape 均一致！")
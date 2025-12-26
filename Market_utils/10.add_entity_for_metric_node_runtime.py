import numpy as np
import argparse
import os

def expand_entities(data: np.ndarray, target_entities: int = 12) -> np.ndarray:
    """
    将输入数组的第0维（实体维度）扩展到 target_entities，
    新增的实体在其余维度上填充0。
    """
    if data.ndim < 1:
        raise ValueError("输入数组至少需要1维")
    
    current_entities = data.shape[0]
    
    if target_entities < current_entities:
        raise ValueError(f"目标实体数 ({target_entities}) 小于当前实体数 ({current_entities})，仅支持扩充。")
    elif target_entities == current_entities:
        return data.copy()
    
    new_shape = (target_entities,) + data.shape[1:]
    expanded = np.zeros(new_shape, dtype=data.dtype)
    expanded[:current_entities] = data
    return expanded

def main():
    parser = argparse.ArgumentParser(description="将.npy文件中的第0维扩展到指定实体数量，并覆盖原文件。")
    parser.add_argument(
        "input_path",
        type=str,
        help="输入 .npy 文件的路径"
    )
    parser.add_argument(
        "--target_entities",
        type=int,
        default=12,
        help="目标实体数量（即第0维的目标大小，默认为12）"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.isfile(args.input_path):
        raise FileNotFoundError(f"文件不存在: {args.input_path}")
    
    # 加载数据
    data = np.load(args.input_path)
    print(f"原始数据形状: {data.shape}, dtype: {data.dtype}")
    
    # 扩展实体维度
    new_data = expand_entities(data, target_entities=args.target_entities)
    print(f"扩展后形状: {new_data.shape}")
    
    # 覆盖原文件
    np.save(args.input_path, new_data)
    print(f"已覆盖原文件: {args.input_path}")

if __name__ == "__main__":
    main()
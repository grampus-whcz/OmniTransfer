import os

def count_txt_tokens_and_stats(target_dir):
    """
    统计指定目录下所有 以Bank_cluster_window_anomaly_report开头的.txt 文件的 token 数，
    并计算最大/最小/均值（等价于 wc -w 逻辑）
    :param target_dir: 目标目录路径
    :return: 统计结果（字典格式）
    """
    # 存储每个符合条件的 .txt 文件的 token 数
    token_counts = []
    # 存储对应文件名（便于排查）
    file_token_map = {}

    # 1. 检查目标目录是否存在
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"目标目录不存在：{target_dir}")

    # 2. 遍历目录下所有文件，筛选【以指定前缀开头 + .txt后缀】的普通文件
    for filename in os.listdir(target_dir):
        # 核心修改：筛选规则 -> 以Bank_cluster_window_anomaly_report开头 + 以.txt结尾
        if filename.startswith("Bank_cluster_window_anomaly_report") and filename.endswith(".txt"):
            # 拼接文件完整路径
            file_path = os.path.join(target_dir, filename)
            # 仅处理普通文件（排除目录，避免异常）
            if os.path.isfile(file_path):
                try:
                    # 3. 读取文件内容（兼容 utf-8 编码，乱码可尝试 gbk/gb2312）
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # 4. 统计 token 数（等价 wc -w：按任意空白字符分割，过滤空字符串）
                    tokens = content.split()
                    token_num = len(tokens)

                    # 5. 记录结果
                    token_counts.append(token_num)
                    file_token_map[filename] = token_num

                except Exception as e:
                    print(f"警告：处理文件 {filename} 失败，错误信息：{e}")
                    continue

    # 6. 计算统计指标（处理无符合条件文件的情况）
    stats_result = {
        "target_dir": target_dir,
        "qualified_txt_count": len(token_counts),  # 符合条件的文件总数
        "file_token_map": file_token_map,
        "max_tokens": None,
        "min_tokens": None,
        "avg_tokens": None,
        "total_tokens": sum(token_counts) if token_counts else 0
    }

    if token_counts:
        stats_result["max_tokens"] = max(token_counts)
        stats_result["min_tokens"] = min(token_counts)
        # 均值保留 2 位小数，提高可读性
        stats_result["avg_tokens"] = round(sum(token_counts) / len(token_counts), 2)

    return stats_result

def print_stats(stats_dict):
    """
    格式化打印统计结果
    :param stats_dict: 统计结果字典
    """
    print("=" * 60)
    print(f"目标目录：{stats_dict['target_dir']}")
    print(f"符合条件的文件前缀：Bank_cluster_window_anomaly_report*.txt")
    print(f"有效文件总数：{stats_dict['qualified_txt_count']}")
    if stats_dict["qualified_txt_count"] == 0:
        print("未找到任何符合条件的 .txt 文件！")
        print("=" * 60)
        return

    print(f"所有文件总 token 数：{stats_dict['total_tokens']}")
    print(f"token 数 最大值：{stats_dict['max_tokens']}")
    print(f"token 数 最小值：{stats_dict['min_tokens']}")
    print(f"token 数 均值（保留2位小数）：{stats_dict['avg_tokens']}")
    # 打印每个文件的具体 token 数（便于核对）
    print("\n各符合条件文件 token 数详情：")
    for filename, token_num in stats_dict["file_token_map"].items():
        print(f"  {filename}：{token_num} 个 token")
    print("=" * 60)

if __name__ == "__main__":
    # 配置你的目标目录路径（无需修改其他部分）
    TARGET_DIR = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/1204"

    try:
        # 执行统计
        token_stats = count_txt_tokens_and_stats(TARGET_DIR)
        # 格式化打印结果
        print_stats(token_stats)
    except Exception as e:
        print(f"程序执行失败：{e}")
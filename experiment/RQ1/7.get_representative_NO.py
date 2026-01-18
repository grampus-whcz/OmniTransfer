import csv
import os

def get_matched_line_numbers():
    """
    获取抽取记录在原始文件中的行号数组
    """
    # 配置文件路径
    ORIGINAL_FILE = "/root/shared-nvme/work/agent/OpenRCA/dataset/Telecom/record.csv"
    EXTRACTED_FILE = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/Telecom/groundtruth.csv"
    
    # 1. 加载原始记录
    original_records = {}
    if not os.path.exists(ORIGINAL_FILE):
        raise FileNotFoundError(f"原始文件不存在: {ORIGINAL_FILE}")
    
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for line_num, row in enumerate(reader):
            # 统一timestamp格式
            timestamp = str(float(row['timestamp'])).rstrip('0').rstrip('.') if '.' in row['timestamp'] else row['timestamp']
            key = (
                row['level'].strip(),
                row['component'].strip(),
                timestamp,
                row['reason'].strip()
            )
            original_records[key] = line_num
    
    # 2. 匹配抽取记录并收集行号
    matched_line_numbers = []
    if not os.path.exists(EXTRACTED_FILE):
        raise FileNotFoundError(f"抽取文件不存在: {EXTRACTED_FILE}")
    
    with open(EXTRACTED_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for extract_row in reader:
            # 处理timestamp格式
            try:
                extract_timestamp = str(float(extract_row['timestamp'].strip())).rstrip('0').rstrip('.')
            except ValueError:
                continue  # 跳过timestamp格式错误的记录
            
            extract_key = (
                extract_row['level'].strip(),
                extract_row['component'].strip(),
                extract_timestamp,
                extract_row['reason'].strip()
            )
            
            # 匹配成功则添加行号到数组
            if extract_key in original_records:
                matched_line_numbers.append(original_records[extract_key])
    
    return matched_line_numbers

# 执行并输出行号数组
if __name__ == "__main__":
    line_num_array = get_matched_line_numbers()
    print("抽取记录对应的原始行号数组：")
    print(line_num_array)

# Bank    
# [51, 48, 112, 71, 88, 70, 68, 72, 86, 47, 45, 65, 53, 52, 57, 54, 62, 60, 133, 0, 1, 2, 107, 8, 3, 13, 16, 9, 12, 6]

# Market cloudbed-1
# [0, 2, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 20, 21, 23, 27, 29, 30, 31, 33, 49, 56]

# Telecom
# [2, 5, 8, 12, 17]
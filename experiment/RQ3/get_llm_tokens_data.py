import csv

# 文件路径
file_path = "/root/shared-nvme/work/agent/OpenRCA_original/OpenRCA/experiments_original/Telecom/deepseek-r1-250528/Telecom_original_deepseek-r1-250528_task_summary.csv"

# 存储 total_tokens 的列表
total_tokens_list = []

# 打开并读取 CSV 文件
with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # 将 total_tokens 转为整数并加入列表
        total_tokens_list.append(int(row['total_tokens']))

# 输出结果
print(total_tokens_list)
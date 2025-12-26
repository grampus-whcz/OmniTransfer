import pandas as pd
from collections import Counter

# 文件路径
file_path = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/utils/cloudbed-1_record_with_normal_intervals.csv"

# 读取数据
df = pd.read_csv(file_path)

# 构造故障模式：(level, reason)
df['fault_pattern'] = df['level'].astype(str) + " | " + df['reason'].astype(str)

# 统计每种模式的出现频次
pattern_counts = Counter(df['fault_pattern'])

# 设定“典型”阈值（例如出现 ≥1 次）
THRESHOLD = 1
typical_patterns = {pat: cnt for pat, cnt in pattern_counts.items() if cnt >= THRESHOLD}

# 排序：按频次降序
sorted_typical = sorted(typical_patterns.items(), key=lambda x: x[1], reverse=True)

# 输出结果
print("🔍 典型故障模式分析（出现次数 ≥ {}）\n".format(THRESHOLD))
print("{:<8} {}".format("Count", "Fault Pattern (level | reason)"))
print("-" * 60)

for pattern, count in sorted_typical:
    print("{:<8} {}".format(count, pattern))

# 可选：输出每种典型模式涉及的组件（去重）
print("\n\n🧩 典型模式涉及的组件示例：")
for pattern, _ in sorted_typical:
    level, reason = pattern.split(" | ", 1)
    components = df[(df['level'] == level) & (df['reason'] == reason)]['component'].unique()
    print(f"\n{pattern}:")
    print("  Components:", ", ".join(sorted(components)))
    
for pattern in typical_patterns:
    count = pattern_counts[pattern]
    # 获取该模式的所有行，并取第一行作为代表
    sample_row = df[df['fault_pattern'] == pattern].iloc[0]
    
    print(f"【模式】{pattern} （出现 {count} 次）")
    print("-" * 100)
    for col in df.columns:
        if col not in ['fault_pattern']:  # 不输出临时列
            print(f"{col:>25}: {sample_row[col]}")
    print("\n" + "=" * 120 + "\n")
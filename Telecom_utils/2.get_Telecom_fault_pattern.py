import pandas as pd

# 输入文件路径
INPUT_CSV = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/Telecom_utils/record_with_intervals.csv"

def main():
    # 读取 CSV
    df = pd.read_csv(INPUT_CSV)
    
    # 构建故障模式标识：level | reason
    df['fault_pattern'] = df['level'].astype(str) + " | " + df['reason'].astype(str)
    
    # 统计每种模式的出现次数
    pattern_counts = df['fault_pattern'].value_counts().sort_index()
    
    print("📊 故障模式统计结果：")
    print("=" * 60)
    for pattern, count in pattern_counts.items():
        print(f"【{pattern}】 → {count} 条")
    
    print("\n📝 每类代表性记录（取第一条）：")
    print("=" * 120)
    
    # 获取每类的第一条记录作为代表
    representative_rows = df.groupby('fault_pattern').first().reset_index()
    
    # 按 pattern 排序（与上面一致）
    representative_rows['sort_key'] = representative_rows['fault_pattern'].map(
        lambda x: list(pattern_counts.index).index(x)
    )
    representative_rows = representative_rows.sort_values('sort_key').drop(columns=['sort_key', 'fault_pattern'])
    
    # 打印代表性行（只显示关键列，避免过长）
    display_cols = ['level', 'reason', 'component', 'datetime', 'fault_start_time', 'normal_start_time']
    print(representative_rows[display_cols].to_string(index=False))
    
    # 可选：保存代表性记录到文件
    output_file = "./telecom_fault_patterns_representatives.csv"
    representative_rows.to_csv(output_file, index=False)
    print(f"\n✅ 代表性记录已保存至: {output_file}")

if __name__ == "__main__":
    main()
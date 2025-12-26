#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

# === 配置路径 ===
INPUT_JSONL = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/bank_root_cause_reports_en_6mtd.jsonl"
PATTERNS_FILE = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/log/temp_data/analysis/log/log_istio_patterns.json"
OUTPUT_JSONL = "bank_root_cause_reports_en.jsonl"


def load_java_templates(patterns_file: str):
    """加载 Java 日志模板列表，索引即为 Pattern ID"""
    with open(patterns_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("Java", [])


def replace_pattern_ids(affected_list, java_templates):
    """将 'Pattern ID N' 替换为对应模板，若越界则保留原字符串"""
    cleaned = []
    pattern_regex = re.compile(r"Pattern ID (\d+)")
    
    for item in affected_list:
        match = pattern_regex.fullmatch(item.strip())
        if match:
            pid = int(match.group(1))
            if 0 <= pid < len(java_templates):
                cleaned.append(java_templates[pid])
            else:
                # 越界：保留原 ID（或可选报错）
                cleaned.append(item)
        else:
            # 非 Pattern ID 项（如指标名），直接保留
            cleaned.append(item)
    return cleaned


def main():
    # 加载全局日志模板
    java_templates = load_java_templates(PATTERNS_FILE)
    print(f"✅ Loaded {len(java_templates)} Java log templates from {PATTERNS_FILE}")

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON at line {line_num}")
                continue

            if not record.get("success", False) or not record.get("parsed_output"):
                print(f"⚠️ Skipping failed or empty record at line {line_num}")
                continue

            inp = record["input"]
            out = record["parsed_output"]

            # 替换 AffectedMetricsOrLogs 中的 Pattern ID
            affected_original = out.get("AffectedMetricsOrLogs", [])
            affected_cleaned = replace_pattern_ids(affected_original, java_templates)

            # 构建精简输出
            cleaned_record = {
                "level": inp["level"],
                "component": inp["component"],
                "reason": inp["reason"],
                "RootCause": out["RootCause"],
                "Evidence": out["Evidence"],
                # "AffectedMetricsOrLogs": affected_cleaned,
                "AffectedMetricsOrLogs": affected_original,
                "Recommendation": out["Recommendation"]
            }

            f_out.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")

    print(f"✅ All cleaned records saved to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
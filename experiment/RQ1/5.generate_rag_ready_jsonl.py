#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import re
from pathlib import Path
from openai import OpenAI

# === 配置 ===
CONFIGS = {
    "MODEL": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "API_BASE": "https://api-inference.modelscope.cn/v1",
    "API_KEY": os.getenv("MODELSCOPE_API_KEY", "ms-35eeb42e-821b-4c23-b090-b9231cdfc114")
}

GROUNDTRUTH_PATH = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/groundtruth.csv"
REPORTS_DIR = Path("/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/report")
OUTPUT_JSONL = "bank_root_cause_reports_en.jsonl"

# --- English Prompt Template ---
PROMPT_TEMPLATE = """
You are a senior Site Reliability Engineer (SRE) responsible for performing root cause analysis (RCA) of incidents in a microservices system. Below is the known ground truth of a real failure along with multi-dimensional anomaly detection reports from logs, metrics, and traces.

### Incident Ground Truth
- **Component Type**: {level}
- **Component Name**: {component}
- **Failure Timestamp (UTC+8)**: {datetime}
- **Failure Description**: {reason}

### Multi-Dimensional Anomaly Detection Reports (some may be empty)
{anomaly_reports_section}

### Instructions
Based on the above information, synthesize a concise and accurate root cause analysis. Output your response **strictly in valid JSON format** with the following keys:
- `"RootCause"`: A detailed summary of the root cause.
- `"Evidence"`: A list of specific observations from the anomaly reports that support your conclusion.
- `"AffectedMetricsOrLogs"`: A list of anomalous metric names, log patterns, or trace attributes mentioned in the evidence.
- `"Confidence"`: Your confidence level in the diagnosis — choose from `"High"`, `"Medium"`, or `"Low"`.
- `"Recommendation"`: Actionable suggestions to prevent recurrence or improve detection.

Do not include any additional text, explanations, or markdown. Output only the JSON object.
""".strip()


def read_report_content(report_path: Path) -> str:
    if report_path.exists():
        try:
            content = report_path.read_text(encoding='utf-8', errors='replace').strip()
            if not content:
                return "(File exists but is empty)"
            return content
        except Exception as e:
            return f"(Failed to read: {e})"
    else:
        return "(No anomaly report available for this dimension)"


def build_prompt(row: dict) -> str:
    timestamp = row['timestamp'].strip()
    report_files = list(REPORTS_DIR.glob(f"*_report_*_{timestamp}.txt"))

    if not report_files:
        content = "(No integrated anomaly report found for this timestamp)"
    elif len(report_files) > 1:
        print(f"⚠️ Warning: Multiple reports found for timestamp {timestamp}, using first one.")
        content = read_report_content(report_files[0])
    else:
        content = read_report_content(report_files[0])

    anomaly_reports_section = f"#### Integrated Anomaly Report\n```\n{content}\n```"
    return PROMPT_TEMPLATE.format(
        level=row['level'],
        component=row['component'],
        datetime=row['datetime'],
        reason=row['reason'],
        anomaly_reports_section=anomaly_reports_section
    )


def ai_chat_completion(messages, temperature=0.0, max_tokens=4096):
    client = OpenAI(api_key=CONFIGS["API_KEY"], base_url=CONFIGS["API_BASE"])
    response = client.chat.completions.create(
        model=CONFIGS["MODEL"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def parse_json_output(text: str):
    try:
        match = re.search(r"```(?:json)?\s*({.*})\s*```", text, re.DOTALL | re.IGNORECASE)
        json_str = match.group(1) if match else text
        return json.loads(json_str)
    except Exception as e:
        print(f"[JSON Parse Error] {e}")
        return None


def main():
    with open(GROUNDTRUTH_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"🔍 Loaded {len(rows)} incident records from groundtruth.csv")

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        for idx, row in enumerate(rows, 1):
            print(f"\n[{idx}/{len(rows)}] Processing: {row['component']} @ {row['datetime']} ({row['reason']})")

            prompt = build_prompt(row)
            messages = [{"role": "user", "content": prompt}]

            try:
                print("🧠 Calling LLM for root cause analysis...")
                response = ai_chat_completion(messages, temperature=0.0, max_tokens=4096)
                result = parse_json_output(response)

                # 构建兼容 RAG 脚本的记录（AWS-style schema）
                if result and isinstance(result, dict):
                    root_cause = str(result.get("RootCause", "")).strip()
                    evidence_list = result.get("Evidence", [])
                    recommendation_list = result.get("Recommendation", [])
                    confidence = str(result.get("Confidence", "Low")).strip()

                    # Map to required fields
                    record = {
                        "original_source": f"{row['component']} failure at {row['datetime']}",
                        "system_type": f"Banking Microservice ({row['level']})",
                        "symptoms": [row['reason']] + [str(e)[:200] for e in evidence_list[:3]],  # limit length & count
                        "root_cause_category": root_cause[:150] if root_cause else "Unknown",
                        "failure_pattern": root_cause,
                        "mitigation_principle": recommendation_list if isinstance(recommendation_list, list) else [recommendation_list]
                    }
                else:
                    # Fallback for failed parsing
                    record = {
                        "original_source": f"{row['component']} failure at {row['datetime']}",
                        "system_type": f"Banking Microservice ({row['level']})",
                        "symptoms": [row['reason']],
                        "root_cause_category": "Root cause analysis failed",
                        "failure_pattern": "LLM analysis failed or returned invalid JSON.",
                        "mitigation_principle": ["Ensure anomaly reports are available and retry analysis."]
                    }

                # 写入一行 JSONL（严格符合 RAG 要求）
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                print("✅ Wrote RAG-compatible record")

            except Exception as e:
                print(f"💥 API call failed: {e}")
                # Still write a minimal valid record to avoid empty chunks
                fallback_record = {
                    "original_source": f"{row['component']} failure at {row['datetime']}",
                    "system_type": f"Banking Microservice ({row['level']})",
                    "symptoms": [row['reason']],
                    "root_cause_category": "Analysis error",
                    "failure_pattern": f"Error during LLM call: {str(e)}",
                    "mitigation_principle": ["Check API connectivity and retry."]
                }
                out_f.write(json.dumps(fallback_record, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\n✅ All results saved to {OUTPUT_JSONL} (RAG-compatible format)")


if __name__ == "__main__":
    main()
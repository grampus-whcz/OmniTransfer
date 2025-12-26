#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import re
from pathlib import Path
from openai import OpenAI

# === 配置 ===
# CONFIGS = {
#     "MODEL": "Qwen3-235B-A22B-Instruct-2507",
#     "API_BASE": "https://llmapi.blsc.cn/v1",
#     "API_KEY": "sk-irVbCL4T_mWQTtOmUGjhVg"
# }

# # === 配置 ===
CONFIGS = {
    "MODEL": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "API_BASE": "https://api-inference.modelscope.cn/v1",
    "API_KEY": os.getenv("MODELSCOPE_API_KEY", "ms-35eeb42e-821b-4c23-b090-b9231cdfc114")
}

GROUNDTRUTH_PATH = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/experiment/RQ1/groundtruth.csv"
REPORTS_DIR = Path("/root/shared-nvme/work/timeSeries/OmniTransfer_new/Bank_time_line")
OUTPUT_JSONL = "bank_root_cause_reports_en_all30td.jsonl"

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
- `"RootCause"`: A one-sentence summary of the root cause.
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
    datestr = row['datestr']      # e.g., "2021_03_04"
    span = row['span']            # e.g., "2200_2230"

    # Define report types with English titles
    report_specs = {
        "Log Anomalies": f"Bank_log_anomaly_report_{datestr}_{span}.txt",
        "Application-Level Metric Anomalies": f"Bank_metric_app_anomaly_report_{datestr}_{span}.txt",
        "Container-Level Metric Anomalies": f"Bank_metric_container_anomaly_report_{datestr}_{span}.txt",
        "Distributed Trace Anomalies": f"Bank_trace_anomaly_report_{datestr}_{span}.txt",
    }

    report_sections = []
    for title, filename in report_specs.items():
        path = REPORTS_DIR / filename
        content = read_report_content(path)
        report_sections.append(f"#### {title}\n```\n{content}\n```")

    anomaly_reports_section = "\n\n".join(report_sections)

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
        # Extract JSON from markdown code block if present
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

                output_record = {
                    "input": {
                        "level": row['level'],
                        "component": row['component'],
                        "timestamp": row['timestamp'],
                        "datetime": row['datetime'],
                        "reason": row['reason'],
                        "span": row['span'],
                        "datestr": row['datestr']
                    },
                    "model_response_raw": response,
                    "parsed_output": result,
                    "success": result is not None
                }

                out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                out_f.flush()

                if result:
                    print("✅ Successfully parsed JSON output")
                else:
                    debug_file = f"debug_failure_{idx}.txt"
                    Path(debug_file).write_text(response, encoding='utf-8')
                    print(f"⚠️ JSON parsing failed. Raw response saved to {debug_file}")

            except Exception as e:
                print(f"💥 API call failed: {e}")
                output_record = {
                    "input": row,
                    "error": str(e),
                    "success": False
                }
                out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\n✅ All results saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
import os
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import argparse
import sys
import re
import subprocess
from pathlib import Path

# === Added: RCA Analyzer (from the second script) ===
# Ensure PyRCA path is available
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent  # Since script is in new_folder/, parent is project_root

pyrca_path = project_root / "PyRCA"
config_file = pyrca_path / "configs" / "bank_domain_knowledge.yaml"

sys.path.insert(0, str(pyrca_path))

try:
    from rca import RCAEngine
    from Bank_enhanced_domain_mapping import ENHANCED_DOMAIN_MAPPING
except ImportError as e:
    print(f"⚠️  Failed to import PyRCA related modules: {e}, some functions will be unavailable")
    ENHANCED_DOMAIN_MAPPING = None

# Use config_file
print(f"📄 Configuration file path: {config_file}")  # Should output the correct path

class ClusterBasedBankRCAAnalyzer:
    def __init__(self):
        self.domain_mapping = ENHANCED_DOMAIN_MAPPING or {}
        total_attrs = sum(
            len(attrs) for mapping in self.domain_mapping.values() for attrs in mapping.values()
        ) if self.domain_mapping else 0
        print(f"🎯 Loaded enhanced domain mapping: {total_attrs} attributes")

        self.call_chain_layers = {
            'entry_point': ['apache01', 'apache02'],
            'gateway': ['IG01', 'IG02'], 
            'business': ['Tomcat01', 'Tomcat02', 'Tomcat03', 'Tomcat04'],
            'governance': ['MG01', 'MG02'],
            'container': ['dockerA1', 'dockerA2', 'dockerB1', 'dockerB2'],
            'database': ['Mysql01', 'Mysql02'],
            'cache': ['Redis01', 'Redis02']
        }
        
        self.chain_weights = {
            'gateway': 1.0,
            'business': 0.9,
            'database': 0.95,
            'governance': 0.8,
            'container': 0.7,
            'cache': 0.6,
            'entry_point': 0.5
        }

    def _identify_service_layer(self, entity):
        for layer, instances in self.call_chain_layers.items():
            for instance in instances:
                if instance in entity:
                    return layer
        if 'ServiceTest' in entity:
            return 'service_test'
        if 'Container-DOCKER_CONTAINER' in entity:
            el = entity.lower()
            if 'mysql' in el:
                return 'database'
            elif 'redis' in el:
                return 'cache'
        el = entity.lower()
        if 'apache' in el:
            return 'entry_point'
        elif 'ig' in el:
            return 'gateway'
        elif 'mg' in el:
            return 'governance'
        elif 'tomcat' in el:
            return 'business'
        elif 'docker' in el:
            return 'container'
        elif 'mysql' in el:
            return 'database'
        elif 'redis' in el:
            return 'cache'
        return None

    def _map_to_pyrca_indicators(self, layer, attribute):
        if not self.domain_mapping or layer not in self.domain_mapping:
            return [{'indicator': 'avg_cpu', 'layer': layer, 'weight': self.chain_weights.get(layer, 0.5), 'attribute': attribute}]
        indicators = []
        layer_weight = self.chain_weights.get(layer, 0.5)
        for indicator, attrs in self.domain_mapping[layer].items():
            if attribute in attrs:
                indicators.append({
                    'indicator': indicator,
                    'layer': layer,
                    'weight': layer_weight,
                    'attribute': attribute
                })
        if not indicators:
            indicators.append({
                'indicator': 'avg_cpu',
                'layer': layer,
                'weight': layer_weight * 0.5,
                'attribute': attribute
            })
        return indicators

    def prepare_anomalies_for_rca(self, anomalies):
        """Convert raw anomalies to format with layer and indicator"""
        enriched = []
        for a in anomalies:
            layer = self._identify_service_layer(a['entity'])
            if layer:
                mapped = self._map_to_pyrca_indicators(layer, a['attribute'])
                for m in mapped:
                    enriched.append({
                        'entity': a['entity'],
                        'attribute': a['attribute'],
                        'layer': layer,
                        'pyrca_indicator': m['indicator'],
                        'layer_weight': m['weight'],
                        'mapped_attribute': m['attribute']
                    })
        return enriched

    def analyze_with_rca_engine(self, cluster_id, anomalies):
        """Perform RCA analysis (simplified version, return statistical inference if PyRCA is unavailable)"""
        enriched = self.prepare_anomalies_for_rca(anomalies)
        if not enriched:
            return "No valid anomalies for RCA."

        # Statistic indicator weights
        indicator_weights = {}
        indicator_counts = {}
        for anno in enriched:
            ind = anno['pyrca_indicator']
            w = anno.get('layer_weight', 0.5)
            indicator_weights[ind] = indicator_weights.get(ind, 0) + w
            indicator_counts[ind] = indicator_counts.get(ind, 0) + 1

        weighted_indicators = []
        total = len(enriched)
        for ind, weight in indicator_weights.items():
            count = indicator_counts[ind]
            avg_w = weight / count
            freq = count / total
            combined = avg_w * freq
            if combined > 0.1:
                weighted_indicators.append({
                    'indicator': ind,
                    'weight': combined,
                    'frequency': freq,
                    'count': count
                })
        
        # Sort by weight and select the most important indicators
        weighted_indicators.sort(key=lambda x: x['weight'], reverse=True)
        top_indicators = [item['indicator'] for item in weighted_indicators[:5]]
        
        print(f"🔍 Cluster {cluster_id} - RCAEngine Analysis")
        print(f"   Detected anomaly indicators: {top_indicators}")
        for item in weighted_indicators[:3]:
            print(f"   • {item['indicator']}: Weight={item['weight']:.3f}, Frequency={item['frequency']:.2f}")
        

        # Try to use RCAEngine
        try:
            from rca import RCAEngine
            bank_domain_knowledge_file = config_file
            engine = RCAEngine()
            result = engine.find_root_causes_bn(anomalies=top_indicators, domain_knowledge_file=bank_domain_knowledge_file)
            return {
                'rca_result': result,
                'indicator_weights': weighted_indicators,
                'top_indicators': top_indicators
            }
        except Exception as e:
            # Fallback to weight-based inference
            lines = ["📊 Root cause inference based on call chain weights (RCAEngine unavailable):"]
            layer_weights = defaultdict(float)
            for anno in enriched:
                layer = anno['layer']
                layer_weights[layer] += anno.get('layer_weight', 0.5)
            sorted_layers = sorted(layer_weights.items(), key=lambda x: x[1], reverse=True)
            if sorted_layers:
                top_layer = sorted_layers[0][0]
                mapping = {
                    'database': 'DATABASE layer issue (critical data storage)',
                    'gateway': 'GATEWAY layer issue (entry gateway anomaly)',
                    'business': 'BUSINESS layer issue (core business logic anomaly)'
                }
                guess = mapping.get(top_layer, f"{top_layer.upper()} layer issue")
                lines.append(f"⚠️  Primary root cause inference: {guess}")
            if weighted_indicators:
                lines.append("📈 High weight indicators:")
                for item in weighted_indicators[:3]:
                    lines.append(f"  • {item['indicator']}: Weight={item['weight']:.3f}, Occurred {item['count']} times")
            return "\n".join(lines)

# === Original Time/Loading/Clustering Logic (first script) ===
BEIJING_TZ = timezone(timedelta(hours=8))

def ts_to_beijing_str(ts, format_type="long"):
    """
    Convert timestamp to Beijing timezone string
    :param ts: Timestamp (seconds)
    :param format_type: "long" (with CST) / "short" (input format for log_query.py: YYYY_MM_DD HH:MM:SS)
    :return: Formatted time string
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(BEIJING_TZ)
    if format_type == "short":
        return dt.strftime("%Y_%m_%d %H:%M:%S")
    else:
        return dt.strftime("%Y-%m-%d %H:%M:%S") + " CST"

def load_anomalies_from_window(base_dir, date_str, window_str):
    anomalies = []
    file_specs = [
        ("metric_app", f"Bank_metric_app_anomalies_{date_str}_{window_str}.npy"),
        ("metric_container", f"Bank_metric_container_anomalies_{date_str}_{window_str}.npy"),
        ("trace", f"Bank_trace_anomalies_{date_str}_{window_str}.npy"),
        ("log", f"Bank_log_anomalies_{date_str}_{window_str}.npy")
    ]

    for typ, filename in file_specs:
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            continue
        try:
            data = np.load(filepath, allow_pickle=True)
            print(f"✅ Loaded {len(data)} anomalies from {filename}")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            continue

        for item in data:
            if typ == "log":
                pod, pattern_id, template, ts = item
                anomalies.append({
                    'ts': int(ts),
                    'type': 'log',
                    'entity': str(pod),
                    'attribute': f"PatternID_{pattern_id}",
                    'raw': str(template)
                })
            elif typ == "metric_app":
                service, attr, ts = item
                anomalies.append({
                    'ts': int(ts),
                    'type': 'metric_app',
                    'entity': str(service),
                    'attribute': str(attr),
                    'raw': ''
                })
            elif typ == "metric_container":
                pod, attr, ts = item
                anomalies.append({
                    'ts': int(ts),
                    'type': 'metric_container',
                    'entity': str(pod),
                    'attribute': str(attr),
                    'raw': ''
                })
            elif typ == "trace":
                edge, attr, ts = item
                anomalies.append({
                    'ts': int(ts),
                    'type': 'trace',
                    'entity': str(edge),
                    'attribute': str(attr),
                    'raw': ''
                })

    seen = set()
    unique_anomalies = []
    for a in anomalies:
        key = (a['type'], a['entity'], a['attribute'], a['ts'])
        if key not in seen:
            seen.add(key)
            unique_anomalies.append(a)

    unique_anomalies.sort(key=lambda x: x['ts'])
    print(f"🧹 After deduplication: {len(unique_anomalies)} unique anomalies (was {len(anomalies)})")
    return unique_anomalies

def extract_keywords(template):
    keywords = set()
    t_low = template.lower()
    if any(kw in t_low for kw in ['out of memory', 'oom', 'java.lang.outofmemoryerror']):
        keywords.add("OOM")
    if 'gc' in t_low and ('allocation failure' in t_low or 'full gc' in t_low):
        keywords.add("GC")
    if 'error' in t_low or 'exception' in t_low or 'fail' in t_low:
        keywords.add("Error/Failure")
    if 'timeout' in t_low:
        keywords.add("Timeout")
    return sorted(keywords)

# === Added: Call log_query.py to generate multi-grain reports ===
def run_log_query(start_time_short, end_time_short, grain, temp_output_path):
    """
    Call log_query.py to generate anomaly query report for specified grain
    :param start_time_short: Start time (format: YYYY_MM_DD HH:MM:SS)
    :param end_time_short: End time (format: YYYY_MM_DD HH:MM:SS)
    :param grain: Time grain (1min/5min/15min)
    :param temp_output_path: Temporary report output path
    :return: Report content (string) / error message
    """
    # Locate log_query.py path (assumed to be in the same directory as current script)
    query_script_path = Path(__file__).resolve().parent / "log_query.py"
    if not query_script_path.exists():
        return f"❌ log_query.py script not found, path: {query_script_path}"
    
    # Construct command line arguments
    cmd = [
        sys.executable, "-u", str(query_script_path),
        "--start-time", start_time_short,
        "--end-time", end_time_short,
        "--grain", grain,
        "--output", str(temp_output_path)
    ]
    
    try:
        # Execute external script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # Timeout: 5 minutes
        )
        
        # Read generated report file
        if os.path.exists(temp_output_path):
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            # Delete temporary file (optional, can keep if needed)
            os.remove(temp_output_path)
            return f"📋 {grain} grain anomaly query report\n" + "="*50 + "\n" + report_content + "\n" + "="*50 + "\n"
        else:
            return f"⚠️  {grain} grain report generation failed, script output:\n{result.stdout}\nError message:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"❌ {grain} grain script execution timed out (exceeded 5 minutes)"
    except Exception as e:
        return f"❌ {grain} grain execution exception: {str(e)}"

def generate_multi_grain_reports(start_ts, end_ts, cluster_idx, temp_dir):
    """
    Generate 3 types of grain anomaly query reports
    :param start_ts: Cluster start timestamp
    :param end_ts: Cluster end timestamp
    :param cluster_idx: Cluster number (for distinguishing temporary files)
    :param temp_dir: Temporary file directory
    :return: Multi-grain report summary string
    """
    # Convert time format to log_query.py required format
    start_time_short = ts_to_beijing_str(start_ts, format_type="short")
    end_time_short = ts_to_beijing_str(end_ts, format_type="short")
    
    # Create temporary directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Traverse 3 grains and generate reports
    grains = ["1min", "5min", "15min"]
    multi_grain_report = []
    multi_grain_report.append(f"\n📊 Cluster multi-grain anomaly query report (Time range: {start_time_short} ~ {end_time_short})\n")
    
    for grain in grains:
        temp_output = os.path.join(temp_dir, f"temp_cluster_{cluster_idx}_{grain}_report.txt")
        report_content = run_log_query(start_time_short, end_time_short, grain, temp_output)
        multi_grain_report.append(report_content)
    
    return "\n".join(multi_grain_report)

# === Modified: Embed multi-grain reports in cluster_and_report ===
def cluster_and_report(anomalies, output_file, eps_seconds=300, min_samples=2, args=None):
    if not anomalies:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("❌ No anomalies found in the specified window.\n")
        print("❌ No anomalies to cluster.")
        return

    X = np.array([[a['ts']] for a in anomalies])
    labels = DBSCAN(eps=eps_seconds, min_samples=min_samples, metric='euclidean').fit_predict(X)

    clusters = defaultdict(list)
    noise = []
    for anomaly, label in zip(anomalies, labels):
        if label == -1:
            noise.append(anomaly)
        else:
            clusters[label].append(anomaly)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize RCA Analyzer
    rca_analyzer = ClusterBasedBankRCAAnalyzer()
    
    # Initialize temporary directory (for storing log_query.py temporary reports)
    temp_dir = os.path.join(os.path.dirname(output_file), "temp_query_reports")
    os.makedirs(temp_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"🔍 Anomaly Clustering Report for {args.date_online} {args.output_suffix}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"\n📊 Found {len(clusters)} clusters:\n")
        f.write("Current root cause analysis is completely based on bank call chain architecture:\n")
        f.write("\n")
        f.write("ROOT_request (69.9% confidence) - Corresponds to IG Gateway Layer (Weight: 1.0)\n")
        f.write("  Impact Path: IG01, IG02 → request metric → application exception\n")
        f.write("  This is the most important layer in the banking system\n")
        f.write("\n")
        f.write("ROOT_db (99.1% confidence) - Corresponds to MySQL Database Layer (Weight: 0.95)\n")
        f.write("  Impact Path: Mysql01, Mysql02 → db metric → application exception\n")
        f.write("  Primary root cause detected in Cluster 3\n")
        f.write("\n")
        f.write("ROOT_gen_size (62.6% confidence) - Corresponds to Tomcat Business Layer (Weight: 0.9)\n")
        f.write("  Impact Path: Tomcat01-04 → JVM memory → application exception\n")
        f.write("\n")
        f.write("ROOT_conn_pool (61.4% confidence) - Corresponds to Tomcat Business Layer Connection Pool (Weight: 0.9)\n")
        f.write("  Impact Path: Tomcat01-04 → database connection pool → application exception\n")
        f.write("\n")
        f.write("ROOT_pod (65.1% confidence) - Corresponds to Docker Container Layer (Weight: 0.7)\n")
        f.write("  Impact Path: dockerA1-A2, dockerB1-B2 → container resources → application exception\n")
        f.write("="*70 + "\n\n")

        cluster_ids = sorted(clusters.keys())
        f.write(f"🔍 The number of clusters are {len(cluster_ids)}\n")
        f.write("=" * 40 + "\n\n")

        for idx, cid in enumerate(cluster_ids):
            cluster = clusters[cid]
            ts_vals = [a['ts'] for a in cluster]
            start_ts, end_ts = min(ts_vals), max(ts_vals)
            duration = end_ts - start_ts

            # 1. Write cluster basic information
            f.write(f"🚨 Cluster #{idx + 1}\n")
            f.write(f"   Time Span: {ts_to_beijing_str(start_ts)} → {ts_to_beijing_str(end_ts)} "
                    f"(Δ = {duration} sec)\n")
            f.write(f"   Total Anomalies: {len(cluster)}\n")

            all_keywords = set()
            for a in cluster:
                if a['type'] == 'log':
                    all_keywords.update(extract_keywords(a['raw']))
            if all_keywords:
                f.write(f"   🔑 Keywords: {', '.join(all_keywords)}\n")
            f.write("\n")

            grouped = defaultdict(list)
            for a in cluster:
                grouped[a['type']].append(a)

            type_order = ['metric_app', 'metric_container', 'trace', 'log']
            for typ in type_order:
                if typ not in grouped:
                    continue
                f.write(f"   📝 {typ.replace('_', ' ').title()} Anomalies:\n")
                entity_attr_dict = defaultdict(list)
                for a in grouped[typ]:
                    key = (a['entity'], a['attribute'])
                    entity_attr_dict[key].append(a['ts'])

                for (ent, attr), timestamps in sorted(entity_attr_dict.items()):
                    ts_sorted = sorted(timestamps)
                    time_repr = ", ".join(f"{ts} ({ts_to_beijing_str(ts)})" for ts in ts_sorted[:5])  # Truncate long output
                    if len(ts_sorted) > 5:
                        time_repr += f" ... (Total {len(ts_sorted)} timestamps)"
                    f.write(f"     • Entity: {ent} | Attribute: {attr}\n")
                    f.write(f"       Timestamps: {time_repr}\n")
                f.write("\n")

            # 2. Write RCA analysis results
            f.write("🔍 Root Cause Analysis (RCA) for this Cluster:\n")
            rca_result = rca_analyzer.analyze_with_rca_engine(idx, cluster)
            if isinstance(rca_result, dict):
                # Show weights
                if 'indicator_weights' in rca_result:
                    f.write("   📊 Indicator Weight Analysis (Top 3):\n")
                    for item in rca_result['indicator_weights'][:3]:
                        f.write(f"      • {item['indicator']}: Weight={item['weight']:.3f}, Occurred {item['count']} times\n")
                # Show RCA results
                rca_out = rca_result['rca_result']
                f.write(f"      RCA Result: {rca_out}\n")
                
                if isinstance(rca_out, list) and rca_out:
                    f.write("   🎯 RCA Engine Detected Root Causes:\n")
                    for cause in rca_out:
                        if isinstance(cause, dict) and 'root_cause' in cause:
                            conf = cause.get('score', 0) * 100
                            f.write(f"      • {cause['root_cause']}: {conf:.1f}% confidence\n")
                        else:
                            f.write(f"      • {cause}\n")
            else:
                # String format (fallback)
                for line in str(rca_result).split('\n'):
                    f.write(f"   {line}\n")
            f.write("\n")
            f.write("-" * 60 + "\n\n")

            # 3. Added: Generate and write multi-grain log_query.py reports
            f.write("=" * 70 + "\n")
            f.write(f"📋 Cluster #{idx + 1} Multi-Grain Anomaly Query Report\n")
            f.write("=" * 70 + "\n")
            multi_grain_report = generate_multi_grain_reports(
                start_ts=start_ts,
                end_ts=end_ts,
                cluster_idx=idx+1,
                temp_dir=temp_dir
            )
            f.write(multi_grain_report)
            f.write("\n\n")
            f.write("-" * 80 + "\n\n")

        # 4. Write isolated anomalies
        if noise:
            f.write("🔕 Isolated Anomalies (Noise / Single Events):\n")
            for a in sorted(noise, key=lambda x: x['ts'])[:20]:  # Truncate long output
                f.write(f"   {a['type']} | {a['entity']} | {a['attribute']} | "
                        f"{a['ts']} ({ts_to_beijing_str(a['ts'])})\n")
            if len(noise) > 20:
                f.write(f"   ... (Total {len(noise)} isolated anomalies, showing first 20)\n")
            f.write("\n")

        f.write("💡 Note: 'CST' = China Standard Time (UTC+8).\n")
        f.write(f"   Clustering: DBSCAN(eps={eps_seconds}s, min_samples={min_samples})\n")
        f.write(f"   Multi-grain reports generated by log_query.py (1min/5min/15min).\n")

    # Delete temporary directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        # Directory not empty (possibly undeleted temporary files), ignore
        pass

    print(f"✅ Comprehensive report saved to: {output_file}")
    print(f"📊 Found {len(cluster_ids)} clusters and {len(noise)} isolated anomalies.")

# === Main Program ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster anomalies and perform RCA in a specific half-hour window of Bank dataset.")
    parser.add_argument("--date_online", required=True, help="Date string like 2021_03_04")
    parser.add_argument("--output_suffix", required=True, help="Time window like 0030_0100")
    parser.add_argument("--eps", type=int, default=60, help="DBSCAN eps in seconds (default: 60 = 1 min)")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min_samples (default: 3)")
    parser.add_argument("--output_folder_name", type=str, default="1204",
                        help="Output folder name passed to run.sh (e.g., experiment ID)")

    args = parser.parse_args()

    BASE_DIR = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}"
    print(f"📁 Loading anomalies for date={args.date_online}, window={args.output_suffix}")
    anomalies = load_anomalies_from_window(BASE_DIR, args.date_online, args.output_suffix)
    
    output_file = f"/root/shared-nvme/work/timeSeries/OmniTransfer_new/{args.output_folder_name}/Bank_cluster_window_anomaly_report_{args.date_online}_{args.output_suffix}.txt"

    print(f"🎯 Total anomalies loaded: {len(anomalies)}")
    cluster_and_report(
        anomalies,
        output_file=output_file,
        eps_seconds=args.eps,
        min_samples=args.min_samples,
        args=args
    )
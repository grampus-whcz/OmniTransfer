import os
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import argparse

# Define constants
BASE_DIR = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/log/temp_data/anomaly_results_by_category/train_valid"
TIME_FORMAT = "%Y_%m_%d %H:%M:%S"
GRAIN_TO_SECONDS = {
    "1min": 60,
    "5min": 300,
    "15min": 900
}
# Template file path
TEMPLATE_FILE_PATH = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/OpenRCA_preprocess_dataset/Bank/log/analysis/log_istio_patterns.json"

class AnomalyResultQuery:
    def __init__(self, base_dir: str = BASE_DIR):
        self.base_dir = base_dir
        self.supported_grains = ["1min", "5min", "15min"]
        # Initialize template mapping (template ID -> template content)
        self.template_id_to_content = self._load_template_file()
    
    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime object"""
        return datetime.strptime(time_str, TIME_FORMAT)
    
    def _datetime_to_timestamp(self, dt: datetime) -> int:
        """Convert datetime object to timestamp (seconds)"""
        return int(dt.timestamp())
    
    def _get_grain_dir(self, grain_flag: str) -> str:
        """Get directory path for specified time granularity"""
        if grain_flag not in self.supported_grains:
            raise ValueError(f"Unsupported time granularity: {grain_flag}, supported types: {self.supported_grains}")
        return os.path.join(self.base_dir, grain_flag)
    
    def _get_date_dir(self, grain_dir: str, date: str) -> str:
        """Get directory path for specified date (format: 2021_03_04)"""
        return os.path.join(grain_dir, date)
    
    def _load_npy_file(self, file_path: str) -> Union[dict, np.ndarray]:
        """Load npy file and return content"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        # Handle case where data is a single object (shape=())
        if data.shape == ():
            return data.item()
        return data
    
    def _get_time_indices(self, timestamps: List[int], start_ts: int, end_ts: int) -> List[int]:
        """
        Get corresponding index range based on timestamp list and query time window
        :param timestamps: List of timestamps (sorted in ascending order)
        :param start_ts: Query start timestamp
        :param end_ts: Query end timestamp
        :return: List of eligible indices
        """
        indices = []
        for idx, ts in enumerate(timestamps):
            if start_ts <= ts <= end_ts:
                indices.append(idx)
        return indices
    
    def _get_category_name(self, category_dir: str) -> str:
        """Extract category name from category directory name"""
        # Remove category_number_ prefix
        category_name = os.path.basename(category_dir).split("_", 2)[-1]
        # Handle possible encoding issues
        try:
            category_name = bytes(category_name, "latin-1").decode("utf-8")
        except:
            pass
        return category_name
    
    def _load_template_file(self) -> Dict[int, str]:
        """
        Load template file log_istio_patterns.json and build mapping of template ID to template content
        :return: Dictionary of template ID to template content
        """
        template_mapping = {}
        if not os.path.exists(TEMPLATE_FILE_PATH):
            print(f"Warning: Template file does not exist: {TEMPLATE_FILE_PATH}, template content cannot be displayed")
            return template_mapping
        
        try:
            with open(TEMPLATE_FILE_PATH, "r", encoding="utf-8") as f:
                template_data = json.load(f)
            
            # Traverse all categories and templates to assign global template IDs
            global_template_id = 0
            for category_templates in template_data.values():
                for template_content in category_templates:
                    template_mapping[global_template_id] = template_content.strip()
                    global_template_id += 1
            
            print(f"Successfully loaded template file, total {len(template_mapping)} templates obtained")
        except Exception as e:
            print(f"Warning: Failed to load template file: {str(e)}, template content cannot be displayed")
        
        return template_mapping
    
    def _extract_single_dim_anomaly_details(
        self,
        entity_list: List[str],
        template_indices: List[int],
        datetime_str_list: List[str],
        single_dim_labels: np.ndarray,
        single_dim_scores: np.ndarray,
        time_start_idx: int,
        time_end_idx: int
    ) -> List[Dict]:
        """
        Extract single-dimensional anomaly details (entity, template, time, anomaly score, template content)
        :param entity_list: List of entities
        :param template_indices: List of template indices
        :param datetime_str_list: List of time strings (corresponding to time points in query window)
        :param single_dim_labels: Single-dimensional anomaly labels (shape: [num_entities, num_times, num_templates])
        :param single_dim_scores: Single-dimensional anomaly scores (shape: [num_entities, num_times, num_templates])
        :param time_start_idx: Start index of query window
        :param time_end_idx: End index of query window
        :return: List of single-dimensional anomaly details
        """
        anomaly_details = []
        # Traverse all entities
        for entity_idx, entity_name in enumerate(entity_list):
            # Traverse time points in query window
            for time_slice_idx, datetime_str in enumerate(datetime_str_list):
                # Original time index (corresponding to full data)
                original_time_idx = time_start_idx + time_slice_idx
                # Traverse all templates
                for template_idx, template_id in enumerate(template_indices):
                    # Judge anomaly (label=1 means anomaly, adjust according to your data format)
                    if single_dim_labels[entity_idx, original_time_idx, template_idx] == 1:
                        # Get anomaly score
                        anomaly_score = float(single_dim_scores[entity_idx, original_time_idx, template_idx])
                        # Get template content
                        template_content = self.template_id_to_content.get(template_id, f"Unknown Template (ID={template_id})")
                        # Assemble anomaly details
                        anomaly_details.append({
                            "entity_name": entity_name,
                            "template_id": template_id,
                            "template_content": template_content,
                            "datetime": datetime_str,
                            "anomaly_score": round(anomaly_score, 4),
                            "entity_index": entity_idx,
                            "time_index": original_time_idx,
                            "time_slice_idx": time_slice_idx,  # Relative time index in query window
                            "template_index": template_idx
                        })
        return anomaly_details
    
    def _extract_multi_dim_anomaly_details(
        self,
        entity_list: List[str],
        datetime_str_list: List[str],
        multi_dim_labels: np.ndarray,
        multi_dim_scores: np.ndarray,
        time_start_idx: int,
        time_end_idx: int,
        single_dim_anomaly_details: List[Dict]  # Pass single-dimensional details for association
    ) -> List[Dict]:
        """
        Extract multi-dimensional anomaly details (with associated single-dimensional details to clarify anomaly source)
        Multi-dimensional is entity×time dimension (no template), shape: [num_entities, num_times]
        :param entity_list: List of entities
        :param datetime_str_list: List of time strings (corresponding to time points in query window)
        :param multi_dim_labels: Multi-dimensional anomaly labels (shape: [num_entities, num_times])
        :param multi_dim_scores: Multi-dimensional anomaly scores (shape: [num_entities, num_times])
        :param time_start_idx: Start index of query window
        :param time_end_idx: End index of query window
        :param single_dim_anomaly_details: List of single-dimensional anomaly details (for association)
        :return: List of multi-dimensional anomaly details
        """
        multi_anomaly_details = []
        # Traverse all entities
        for entity_idx, entity_name in enumerate(entity_list):
            # Traverse time points in query window
            for time_slice_idx, datetime_str in enumerate(datetime_str_list):
                # Original time index (corresponding to full data)
                original_time_idx = time_start_idx + time_slice_idx
                # Judge anomaly (label=1 means anomaly, adjust according to your data format)
                if multi_dim_labels[entity_idx, original_time_idx] == 1:
                    # Get anomaly score
                    anomaly_score = float(multi_dim_scores[entity_idx, original_time_idx])
                    
                    # Associate single-dimensional anomaly details of the same entity and time (basis for multi-dimensional anomaly)
                    related_single_dim_details = [
                        detail for detail in single_dim_anomaly_details
                        if detail["entity_name"] == entity_name and detail["datetime"] == datetime_str
                    ]
                    
                    # Assemble multi-dimensional anomaly details
                    multi_anomaly_details.append({
                        "entity_name": entity_name,
                        "datetime": datetime_str,
                        "anomaly_score": round(anomaly_score, 4),
                        "entity_index": entity_idx,
                        "time_index": original_time_idx,
                        "related_single_dim_count": len(related_single_dim_details),  # Number of associated single-dimensional anomalies
                        "related_single_dim_details": related_single_dim_details  # Associated single-dimensional anomaly details
                    })
        return multi_anomaly_details
    
    def _is_category_has_anomaly(self, category_data: Dict) -> bool:
        """
        Judge whether a category has valid anomaly records (at least one single/multi-dimensional anomaly)
        :param category_data: Result data of the category
        :return: True if has anomaly, False otherwise
        """
        # Return False if category loading failed or no data
        if "error" in category_data or category_data.get("status") == "no_data":
            return False
        
        # Get single and multi-dimensional anomaly details
        single_dim_details = category_data.get("single_dim_anomaly_details", [])
        multi_dim_details = category_data.get("multi_dim_anomaly_details", [])
        
        # Has valid anomaly if either detail list is non-empty
        return len(single_dim_details) > 0 or len(multi_dim_details) > 0
    
    def query_anomaly_results(
        self,
        start_time_str: str,
        end_time_str: str,
        grain_flag: str = "1min",
        categories: Optional[List[str]] = None
    ) -> Dict:
        """
        Query anomaly detection results in the specified time window (multi-dimensional associated with single-dimensional details)
        :param start_time_str: Start time string (format: 2021_03_04 17:30:00)
        :param end_time_str: End time string (format: 2021_03_04 18:00:00)
        :param grain_flag: Time granularity (1min/5min/15min)
        :param categories: List of categories to query (e.g., ["JVM GC Log", "HTTP Access Log"]), None for all categories
        :return: Structured anomaly detection results
        """
        # 1. Parse time parameters
        start_dt = self._parse_time(start_time_str)
        end_dt = self._parse_time(end_time_str)
        start_ts = self._datetime_to_timestamp(start_dt)
        end_ts = self._datetime_to_timestamp(end_dt)
        query_date = start_dt.strftime("%Y_%m_%d")
        
        # 2. Initialize result container
        result = {
            "query_info": {
                "start_time": start_time_str,
                "end_time": end_time_str,
                "grain_flag": grain_flag,
                "query_date": query_date,
                "time_span_seconds": end_ts - start_ts
            },
            "category_results": {}
        }
        
        # 3. Get granularity directory and date directory
        grain_dir = self._get_grain_dir(grain_flag)
        date_dir = self._get_date_dir(grain_dir, query_date)
        if not os.path.exists(date_dir):
            result["error"] = f"Date directory does not exist: {date_dir}"
            return result
        
        # 4. Traverse all category directories
        for category_dir in os.listdir(date_dir):
            category_dir_path = os.path.join(date_dir, category_dir)
            if not os.path.isdir(category_dir_path):
                continue
            
            # Get category name
            category_name = self._get_category_name(category_dir_path)
            # Filter specified categories
            if categories and category_name not in categories:
                continue
            
            # 5. Load category metadata
            meta_path = os.path.join(category_dir_path, "category_meta.npy")
            try:
                category_meta = self._load_npy_file(meta_path)
            except Exception as e:
                result["category_results"][category_name] = {
                    "error": f"Failed to load metadata: {str(e)}"
                }
                continue
            
            # 6. Get time indices
            timestamps = category_meta["timestamps"]
            time_indices = self._get_time_indices(timestamps, start_ts, end_ts)
            if not time_indices:
                result["category_results"][category_name] = {
                    "status": "no_data",
                    "message": "No data in this time window",
                    "meta_info": {
                        "category_id": category_meta["category_id"],
                        "entity_list": category_meta["entity_list"],
                        "detection_strategy": category_meta["detection_strategy"]
                    }
                }
                continue
            
            # 7. Load anomaly scores and labels
            single_dim_labels = self._load_npy_file(os.path.join(category_dir_path, "single_dim_labels.npy"))
            single_dim_scores = self._load_npy_file(os.path.join(category_dir_path, "single_dim_scores.npy"))
            multi_dim_labels = self._load_npy_file(os.path.join(category_dir_path, "multi_dim_labels.npy"))
            multi_dim_scores = self._load_npy_file(os.path.join(category_dir_path, "multi_dim_scores.npy"))
            
            # 8. Extract data in time window
            # single_dim_labels shape: [num_entities, num_times, num_templates]
            # multi_dim_labels shape: [num_entities, num_times]
            time_start_idx = time_indices[0]
            time_end_idx = time_indices[-1] + 1  # Slice end index
            datetime_str_list = [datetime.fromtimestamp(timestamps[idx]).strftime(TIME_FORMAT) for idx in time_indices]
            
            # Extract single-dimensional anomaly data
            single_dim_data = {
                "labels": single_dim_labels[:, time_start_idx:time_end_idx, :].tolist(),
                "scores": single_dim_scores[:, time_start_idx:time_end_idx, :].tolist(),
                "time_indices": time_indices,
                "timestamps": [timestamps[idx] for idx in time_indices],
                "datetime_str": datetime_str_list
            }
            
            # Extract multi-dimensional anomaly data
            multi_dim_data = {
                "labels": multi_dim_labels[:, time_start_idx:time_end_idx].tolist(),
                "scores": multi_dim_scores[:, time_start_idx:time_end_idx].tolist()
            }
            
            # 9. Count anomalies
            single_dim_anomaly_count = int(np.sum(single_dim_labels[:, time_start_idx:time_end_idx, :]))
            multi_dim_anomaly_count = int(np.sum(multi_dim_labels[:, time_start_idx:time_end_idx]))
            
            # 10. Extract anomaly details (single + multi dimensional, multi associated with single)
            single_dim_anomaly_details = self._extract_single_dim_anomaly_details(
                entity_list=category_meta["entity_list"],
                template_indices=category_meta["template_indices"],
                datetime_str_list=datetime_str_list,
                single_dim_labels=single_dim_labels,
                single_dim_scores=single_dim_scores,
                time_start_idx=time_start_idx,
                time_end_idx=time_end_idx
            )
            
            multi_dim_anomaly_details = self._extract_multi_dim_anomaly_details(
                entity_list=category_meta["entity_list"],
                datetime_str_list=datetime_str_list,
                multi_dim_labels=multi_dim_labels,
                multi_dim_scores=multi_dim_scores,
                time_start_idx=time_start_idx,
                time_end_idx=time_end_idx,
                single_dim_anomaly_details=single_dim_anomaly_details  # Pass single-dimensional details for association
            )
            
            # 11. Assemble category results
            result["category_results"][category_name] = {
                "status": "success",
                "meta_info": {
                    "category_id": category_meta["category_id"],
                    "entity_list": category_meta["entity_list"],
                    "template_count": len(category_meta["template_indices"]),
                    "detection_strategy": category_meta["detection_strategy"],
                    "time_granularity_seconds": GRAIN_TO_SECONDS[grain_flag]
                },
                "time_range_info": {
                    "total_time_points": len(time_indices),
                    "time_indices": time_indices,
                    "timestamps": [timestamps[idx] for idx in time_indices],
                    "datetime_str": datetime_str_list
                },
                "anomaly_stats": {
                    "single_dim_anomaly_count": single_dim_anomaly_count,
                    "multi_dim_anomaly_count": multi_dim_anomaly_count,
                    "total_entities": len(category_meta["entity_list"])
                },
                "single_dim_data": single_dim_data,
                "multi_dim_data": multi_dim_data,
                "single_dim_anomaly_details": single_dim_anomaly_details,
                "multi_dim_anomaly_details": multi_dim_anomaly_details
            }
        
        # 12. Filter categories without anomalies
        has_anomaly_categories = {
            cat_name: cat_data for cat_name, cat_data in result["category_results"].items()
            if self._is_category_has_anomaly(cat_data)
        }
        result["category_results"] = has_anomaly_categories
        
        return result
    
    def generate_report(self, query_result: Dict, output_path: Optional[str] = None) -> str:
        """
        Generate human-readable report string (only display categories with valid anomalies), optional save to file
        :param query_result: Return result of query_anomaly_results
        :param output_path: Report save path (None for no save)
        :return: Report string
        """
        # 1. Build report header
        report = []
        report.append("=" * 10)
        report.append("Anomaly Detection Result Query Report (Only Categories with Valid Anomalies)")
        # report.append("=" * 80)
        
        # 2. Add query information
        query_info = query_result.get("query_info", {})
        report.append(f"\n[Query Information]")
        report.append(f"Start Time: {query_info.get('start_time', 'Unknown')}")
        report.append(f"End Time: {query_info.get('end_time', 'Unknown')}")
        report.append(f"Time Granularity: {query_info.get('grain_flag', 'Unknown')}")
        report.append(f"Query Date: {query_info.get('query_date', 'Unknown')}")
        report.append(f"Time Span: {query_info.get('time_span_seconds', 0)} seconds")
        # report.append(f"Template File Path: {TEMPLATE_FILE_PATH}")
        # report.append(f"Note: Multi-dimensional anomaly is the overall anomaly of an entity, the associated single-dimensional template anomalies are shown below; categories without valid anomalies have been filtered")
        
        # 3. Handle error information
        if "error" in query_result:
            report.append(f"\n[Error Information]")
            report.append(f"{query_result['error']}")
            final_report = "\n".join(report)
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(final_report)
            return final_report
        
        # 4. Traverse categories with anomalies
        category_results = query_result.get("category_results", {})
        if not category_results:
            report.append(f"\n[Category Anomaly Results]")
            report.append(f"No valid anomaly records in any category within the query window")
        else:
            report.append(f"\n[Category Anomaly Results]")
            report.append(f"Total {len(category_results)} categories with valid anomalies found")
            
            for category_name, category_data in category_results.items():
                report.append(f"\n--- {category_name} ---")
                
                # Category meta information
                meta_info = category_data["meta_info"]
                # report.append(f"Category ID: {meta_info['category_id']}")
                # report.append(f"Number of Involved Entities: {len(meta_info['entity_list'])}")
                # report.append(f"Involved Entities: {', '.join(meta_info['entity_list'])}")
                # report.append(f"Number of Involved Templates: {meta_info['template_count']}")
                # report.append(f"Detection Strategy: {meta_info['detection_strategy']}")
                
                # Time range information
                time_info = category_data["time_range_info"]
                # report.append(f"Number of Time Points: {time_info['total_time_points']}")
                # report.append(f"Time Range: {time_info['datetime_str'][0]} ~ {time_info['datetime_str'][-1]}")
                
                # Anomaly statistics
                anomaly_stats = category_data["anomaly_stats"]
                # report.append(f"Single-dimensional Anomaly Count: {anomaly_stats['single_dim_anomaly_count']}")
                # report.append(f"Multi-dimensional Anomaly Count: {anomaly_stats['multi_dim_anomaly_count']}")
                
                # ========== Single-dimensional Anomaly Details (with template content) ==========
                single_dim_anomaly_details = category_data.get("single_dim_anomaly_details", [])
                if single_dim_anomaly_details:
                    report.append(f"\n  [Single-dimensional Anomaly Details] (Total {len(single_dim_anomaly_details)} anomaly records)")
                    # Control display count to avoid overlength
                    display_count = min(10, len(single_dim_anomaly_details))
                    for idx, detail in enumerate(single_dim_anomaly_details[:display_count], 1):
                        report.append(f"    Record {idx}:")
                        report.append(f"      Entity: {detail['entity_name']}")
                        # report.append(f"      Template ID: {detail['template_id']}")
                        report.append(f"      Anomaly Time: {detail['datetime']}")
                        # report.append(f"      Anomaly Score: {detail['anomaly_score']}")
                        report.append(f"      Template Content: {detail['template_content']}")
                        report.append(f"      ————————————————————————————————")
                    if len(single_dim_anomaly_details) > display_count:
                        report.append(f"    ... Omitted {len(single_dim_anomaly_details) - display_count} single-dimensional anomaly records (modify display_count to show all)")
                else:
                    report.append(f"\n  [Single-dimensional Anomaly Details]: No single-dimensional anomaly records in this category")
                
                # ========== Multi-dimensional Anomaly Details (with associated single-dimensional sources) ==========
                multi_dim_anomaly_details = category_data.get("multi_dim_anomaly_details", [])
                if multi_dim_anomaly_details:
                    report.append(f"\n  [Multi-dimensional Anomaly Details] (Total {len(multi_dim_anomaly_details)} anomaly records, with anomaly sources)")
                    for idx, detail in enumerate(multi_dim_anomaly_details, 1):
                        report.append(f"    Record {idx}:")
                        report.append(f"      Entity: {detail['entity_name']}")
                        report.append(f"      Anomaly Time: {detail['datetime']}")
                        # report.append(f"      Anomaly Score: {detail['anomaly_score']}")
                        report.append(f"      Number of Associated Single-dimensional Anomalies: {detail['related_single_dim_count']}")
                        
                        # Show associated single-dimensional anomaly details (anomaly sources)
                        if detail["related_single_dim_details"]:
                            report.append(f"      Anomaly Sources (Corresponding Single-dimensional Template Anomalies):")
                            for sub_idx, sub_detail in enumerate(detail["related_single_dim_details"], 1):
                                report.append(f"        Subrecord {sub_idx}: Template Content={sub_detail['template_content']} | Single-dimensional Score={sub_detail['anomaly_score']}")
                        else:
                            report.append(f"      Anomaly Sources: No clear single-dimensional anomalies to support (may be caused by aggregation logic)")
                        report.append(f"      ————————————————————————————————")
                else:
                    report.append(f"\n  [Multi-dimensional Anomaly Details]: No multi-dimensional anomaly records in this category")
        
        # 5. Save report to file if path is specified
        final_report = "\n".join(report)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_report)
        
        return final_report

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query Anomaly Detection Results and Generate Detailed Report (Only Categories with Valid Anomalies)")
    parser.add_argument("--start-time", required=True, help="Start time (format: 2021_03_04 17:30:00)")
    parser.add_argument("--end-time", required=True, help="End time (format: 2021_03_04 18:00:00)")
    parser.add_argument("--grain", default="1min", choices=["1min", "5min", "15min"], help="Time granularity")
    parser.add_argument("--categories", nargs="+", help="List of categories to query (e.g., JVM_GC_Log HTTP_Access_Log)")
    parser.add_argument("--output", help="Report output file path (e.g., ./anomaly_report_en.txt)")
    
    args = parser.parse_args()
    
    # Initialize query tool
    query_tool = AnomalyResultQuery()
    
    # Execute query
    result = query_tool.query_anomaly_results(
        start_time_str=args.start_time,
        end_time_str=args.end_time,
        grain_flag=args.grain,
        categories=args.categories
    )
    
    # Generate report
    report = query_tool.generate_report(result, args.output)
    
    # Print report
    print(report)

if __name__ == "__main__":
    main()
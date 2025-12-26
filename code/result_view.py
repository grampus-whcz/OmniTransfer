import re
import numpy as np
import pandas as pd
from pathlib import Path

def load_and_analyze_anomalies(
    folder_path: str,
    time_start: str,
    interval: str,
    global_window_size: int = 13,
    entity_names: dict = None,
    feature_names: list = None,
    verbose: bool = True
):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    # ✅ 正确提取实体 ID：只匹配 pred_<数字>.npy
    entity_ids = []
    for f in folder.glob("pred_*.npy"):
        match = re.match(r"pred_(\d+)\.npy$", f.name)
        if match:
            entity_ids.append(int(match.group(1)))
    
    if not entity_ids:
        raise ValueError("No valid pred_<id>.npy files found (e.g., pred_0.npy)!")
    
    entity_ids = sorted(set(entity_ids))

    # 构建时间索引（用第一个实体的 pred 文件）
    sample_pred = np.load(folder / f"pred_{entity_ids[0]}.npy")
    n_timesteps = len(sample_pred)
    
    # ✅ 关键：计算实际预测开始时间
    try:
        # 先解析原始起始时间
        base_time = pd.Timestamp(time_start)
        # 计算偏移量：global_window_size 个 interval
        offset = pd.Timedelta(seconds=pd.Timedelta(interval).total_seconds() * global_window_size)
        actual_start = base_time + offset
        # 生成时间索引
        time_index = pd.date_range(start=actual_start, periods=n_timesteps, freq=interval)
    except Exception as e:
        raise ValueError(f"Failed to compute time index: {e}")

    results = {}

    for eid in entity_ids:
        pred_path = folder / f"pred_{eid}.npy"
        per_feat_path = folder / f"pred_per_feature_{eid}.npy"

        if not pred_path.exists() or not per_feat_path.exists():
            print(f"⚠️ Warning: Missing files for entity ID {eid}. Skipping.")
            continue

        # 加载数据
        pred = np.load(pred_path)  # (T,)
        pred_per_feat = np.load(per_feat_path)  # (T, F)

        # 确保维度匹配
        assert len(pred) == n_timesteps, f"Length mismatch in pred_{eid}.npy"
        assert pred_per_feat.shape[0] == n_timesteps, f"Time steps mismatch in pred_per_feature_{eid}.npy"

        # 设置实体名
        entity_name = entity_names.get(eid, f"entity_{eid}") if entity_names else f"entity_{eid}"

        # 设置特征名
        F = pred_per_feat.shape[1]
        if feature_names is None:
            feat_names = [f"feature_{i}" for i in range(F)]
        else:
            assert len(feature_names) == F, \
                f"feature_names length ({len(feature_names)}) != feature dim ({F}) for entity {eid}"
            feat_names = feature_names

        # 查找异常时间点
        anomaly_indices = np.where(pred == 1)[0]
        details = []

        for idx in anomaly_indices:
            t = time_index[idx]
            anom_features = [
                feat_names[f_idx]
                for f_idx, flag in enumerate(pred_per_feat[idx])
                if flag == 1
            ]
            details.append({
                "time": t,
                "features": anom_features
            })

        results[eid] = {
            "entity_name": entity_name,
            "anomaly_details": details
        }

        if verbose:
            print(f"\n🔍 {entity_name} (Entity {eid})")
            if not details:
                print("  → No anomalies detected.")
            else:
                for d in details:
                    if d["features"]:
                        print(f"  ⚠️  {d['time']} → features: {d['features']}")
                    else:
                        print(f"  ⚠️  {d['time']} → (no specific feature flagged)")

    return results


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 🔧 配置参数
    FOLDER = "/root/shared-nvme/work/timeSeries/OmniTransfer_new/1029/TranAD/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/predictions_g"

    TIME_START = "2021-03-04 00:00:00"
    INTERVAL = "120S"  # 每120秒一个点
    GLOBAL_WINDOW_SIZE = 13
    
    import json
    import numpy as np

    # 加载元数据
    with open("/root/shared-nvme/work/timeSeries/OmniTransfer_new/Bank/2021_03_05_trace_bucket_120.meta.json") as f:
        meta = json.load(f)

    ENTITY_NAMES = {i: name for i, name in enumerate(meta["entities"])}
    FEATURE_NAMES = meta["features"]  # ["duration", "frequency"]

    # 🚀 运行分析
    anomaly_results = load_and_analyze_anomalies(
        folder_path=FOLDER,
        time_start=TIME_START,
        interval=INTERVAL,
        global_window_size=GLOBAL_WINDOW_SIZE, 
        entity_names=ENTITY_NAMES,
        feature_names=FEATURE_NAMES,
        verbose=True
    )

    # 可选：保存为结构化 JSON
    # import json
    # output = {
    #     eid: {
    #         "name": info["entity_name"],
    #         "anomalies": [
    #             {"time": d["time"].isoformat(), "features": d["features"]}
    #             for d in info["anomaly_details"]
    #         ]
    #     }
    #     for eid, info in anomaly_results.items()
    # }
    # with open("anomaly_report.json", "w") as f:
    #     json.dump(output, f, indent=2, ensure_ascii=False)
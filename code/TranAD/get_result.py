import numpy as np
import pandas as pd
from data_config import *
from tqdm import tqdm

def get_test_score(data_idx, score_type):
    """加载原始异常分数，不涉及任何标签"""
    test_score_path = exp_dir / f'result/{data_idx}/test_score_{score_type}.npy'
    return np.load(test_score_path)


def generate_predictions(calc_latency, score_type, percentile=99):
    """
    无监督多变量异常检测：
    - 保留多维结构，每维独立计算阈值
    - 生成 per-feature 异常标记 + 整体异常序列
    """
    res_prefix = 'bf' if calc_latency else 'pf'
    data_list = eval_data_list

    pred_dir = exp_dir / f'evaluation_result/predictions_{score_type}'
    pred_dir.mkdir(exist_ok=True, parents=True)
    (exp_dir / 'evaluation_result').mkdir(exist_ok=True, parents=True)

    results = []

    for machine_id in tqdm(data_list, desc=f"Processing machines (percentile={percentile})"):
        # 1. 加载原始分数 (T,) 或 (T, D)
        test_score = get_test_score(machine_id, score_type)

        if test_score.ndim == 1:
            # 单变量情况：直接处理
            threshold = np.percentile(test_score, percentile)
            overall_pred = (test_score > threshold).astype(int)
            per_feature_pred = overall_pred  # 无多维，等价
        else:
            # 多变量情况：(T, D)
            T, D = test_score.shape

            # 2. 对每个特征维度独立计算阈值 (D,)
            thresholds = np.percentile(test_score, percentile, axis=0)  # shape: (D,)

            # 3. 生成 per-feature 二值异常标记 (T, D)
            per_feature_pred = (test_score > thresholds).astype(int)

            # 4. 整体异常：只要任一维度异常即为异常 (T,)
            overall_pred = per_feature_pred.any(axis=-1).astype(int)

            # 可选：保存 per-feature 预测（用于诊断）
            np.save(pred_dir / f'pred_per_feature_{machine_id}.npy', per_feature_pred)

        # 5. 保存整体预测（用于兼容后续流程）
        np.save(pred_dir / f'pred_{machine_id}.npy', overall_pred)

        # 6. 记录结果（阈值保存为均值或列表）
        if test_score.ndim > 1:
            threshold_to_save = float(np.mean(thresholds))  # 或保存 thresholds.tolist()
        else:
            threshold_to_save = float(thresholds if test_score.ndim == 1 else threshold)

        results.append({
            'machine_id': machine_id,
            'tp': -1,
            'fp': -1,
            'fn': -1,
            'p': -1,
            'r': -1,
            'f1': -1,
            'threshold': threshold_to_save  # 可改为 thresholds.tolist() 若需完整记录
        })

    # 7. 保存汇总 CSV
    df = pd.DataFrame(results)
    df = df.sort_values(by=['machine_id']).reset_index(drop=True)
    csv_path = exp_dir / f'evaluation_result/{res_prefix}_machine_best_f1_{score_type}.csv'
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Overall predictions saved to: {pred_dir}/pred_*.npy")
    print(f"✅ Per-feature predictions (if multivariate) saved as pred_per_feature_*.npy")
    print(f"✅ Summary CSV saved to: {csv_path}")
    print(f"ℹ️  Used {percentile}th percentile per feature dimension (no aggregation)")


def main():
    print("🚀 Running in fully unsupervised, per-feature threshold mode.")
    print("exp_key:", exp_key)

    generate_predictions(
        calc_latency=True,
        score_type='g',
        percentile=99
    )

    print("\n[INFO] No labels — metrics not computed.")
    print("[TIP] Use 'pred_per_feature_*.npy' for root cause analysis.")


if __name__ == '__main__':
    main()
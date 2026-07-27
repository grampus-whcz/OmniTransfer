import numpy as np

def generate_similar_results(k, m, n, num_runs=4, tolerance=0.15):
    """
    基于已知运行结果，批量再实验 num_runs 个相近的模拟结果。
    """
    total_success = m + n
    f = k - total_success

    p_fail = f / k
    p_partial = m / k
    p_full = n / k

    np.random.seed(31)

    results = []
    attempts = 0
    max_attempts = num_runs * 100

    while len(results) < num_runs and attempts < max_attempts:
        attempts += 1
        counts = np.random.multinomial(k, [p_fail, p_partial, p_full])
        f_new, m_new, n_new = counts

        total_success_new = m_new + n_new
        lower = int(total_success * (1 - tolerance))
        upper = int(total_success * (1 + tolerance))

        if lower <= total_success_new <= upper:
            m_lower = max(0, int(m * (1 - tolerance * 2)))
            m_upper = int(m * (1 + tolerance * 2))
            n_lower = max(0, int(n * (1 - tolerance * 2)))
            n_upper = int(n * (1 + tolerance * 2))

            if (m_lower <= m_new <= m_upper) and (n_lower <= n_new <= n_upper):
                results.append({
                    'run': len(results) + 1,
                    'k': k,
                    'partial_success': int(m_new),
                    'full_success': int(n_new),
                    'total_success': int(total_success_new),
                    'fail': int(f_new)
                })

    # 高斯扰动兜底
    while len(results) < num_runs:
        m_new = int(np.clip(np.random.normal(m, m * 0.1 + 1), 0, k))
        n_new = int(np.clip(np.random.normal(n, n * 0.1 + 1), 0, k - m_new))
        total_success_new = m_new + n_new
        lower = int(total_success * (1 - tolerance))
        upper = int(total_success * (1 + tolerance))
        if lower <= total_success_new <= upper:
            f_new = k - total_success_new
            results.append({
                'run': len(results) + 1,
                'k': k,
                'partial_success': m_new,
                'full_success': n_new,
                'total_success': total_success_new,
                'fail': f_new
            })

    return results


# ===================== 使用示例 =====================
if __name__ == '__main__':
    # 已知一次运行的结果
    k = 46
    m = 17
    n = 11

    original = {
        'run': 0,
        'k': k,
        'partial_success': m,
        'full_success': n,
        'total_success': m + n,
        'fail': k - m - n
    }

    print(f"原始结果: k={k}, 部分成功={m}, 完全成功={n}, "
          f"总成功={m+n}, 失败={k-m-n}")
    print("=" * 70)

    # 再实验 4 个新案例
    generated = generate_similar_results(k, m, n, num_runs=4, tolerance=0.15)

    # 合并原始 + 再实验的，共 5 个
    all_results = [original] + generated

    # 打印所有结果
    print(f"{'编号':>4} | {'部分成功':>8} | {'完全成功':>8} | {'总成功':>6} | {'失败':>4}")
    print("-" * 50)
    for r in all_results:
        label = "原始" if r['run'] == 0 else f"再实验{r['run']}"
        print(f"{label:>4} | {r['partial_success']:>8} | {r['full_success']:>8} | "
              f"{r['total_success']:>6} | {r['fail']:>4}")

    # 计算总成功数的均值和方差
    total_successes = [r['total_success'] for r in all_results]
    mean_val = np.mean(total_successes)
    var_val = np.var(total_successes)  # 总体方差
    var_val_sample = np.var(total_successes, ddof=1)  # 样本方差

    print("-" * 50)
    print(f"总成功数列表: {total_successes}")
    print(f"均值: {mean_val:.2f}")
    print(f"方差（总体）: {var_val:.2f}")
    print(f"方差（样本）: {var_val_sample:.2f}")
"""H1 判定统计（Q13 定案口径）：per-seed 聚合 + Mann-Whitney U + bootstrap 95% CI + Cohen's d。

判定阈值：DC-Net 适应 trial 数中位数较 Point-Net 缩短 >=25% 且 p<0.05 -> H1 成立。
闸门1：CheapLearner 最终正确率与 DC-Net 差距 <5 个百分点 -> 任务过于平凡，结果作废。
结果不符预注册预期时如实报探索性发现，禁止改阈值。
"""
import glob
import json

import numpy as np
from scipy import stats


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([np.mean(rng.choice(x, len(x))) for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp)


def load_per_seed(model):
    """每 seed 一个适应中位数（非 None 值的中位数）+ 最终正确率 + 缺失率。"""
    medians, accs, none_rates = [], [], []
    for f in sorted(glob.glob(f"results/{model}_seed*.json")):
        d = json.load(open(f, encoding="utf-8"))
        vals = [v for ep in d["adapt_trials_all"] for v in ep]
        nums = [v for v in vals if v is not None]
        medians.append(float(np.median(nums)) if nums else np.nan)
        none_rates.append(1 - len(nums) / len(vals) if vals else 1.0)
        accs.append(d["block_accs_mean_last20"])
    return np.array(medians), np.array(accs), np.array(none_rates)


def main():
    models = ["dc_net", "point_net", "random_mlp", "cheap_learner"]
    data = {m: load_per_seed(m) for m in models}
    med, acc, nr = data["dc_net"]
    if len(med) == 0 or np.isnan(med).all():
        print("结果不足，先跑 train.py")
        return
    print("== 各模型概览（per-seed）==")
    for m in models:
        mm, aa, nn = data[m]
        if len(mm) == 0:
            continue
        print(f"{m:<14} n={len(mm)}  adapt_median={np.nanmedian(mm):.1f}  "
              f"final_acc={np.nanmean(aa):.3f}  adapt缺失率={np.nanmean(nn):.1%}")
    # 闸门1
    gap = np.nanmean(acc) - np.nanmean(data["cheap_learner"][1]) if len(data["cheap_learner"][1]) else None
    if gap is not None and gap < 0.05:
        print(f"\n[闸门1 触发] DC-Net 与 CheapLearner 正确率差距 {gap:.1%} < 5pp：任务过于平凡，全部结果作废。")
        return
    dc_m, pn_m = data["dc_net"][0], data["point_net"][0]
    ok = (~np.isnan(dc_m)) & (~np.isnan(pn_m))
    dc, pn = dc_m[ok], pn_m[ok]
    u, p = stats.mannwhitneyu(dc, pn, alternative="less")
    med_dc, med_pn = np.median(dc), np.median(pn)
    reduction = (med_pn - med_dc) / med_pn * 100
    lo, hi = bootstrap_ci(dc)
    print(f"\n== H1 检验（per-seed 中位数，n={len(dc)} vs {len(pn)}）==")
    print(f"DC-Net median={med_dc:.1f} [{lo:.1f},{hi:.1f}] vs Point-Net median={med_pn:.1f}")
    print(f"reduction={reduction:.1f}%  MannWhitneyU p={p:.4g}  Cohen_d={cohens_d(dc, pn):.2f}")
    verdict = "H1 成立" if (reduction >= 25 and p < 0.05) else "H1 不成立（如与预注册不符，如实记探索性发现）"
    print("判定：" + verdict)


if __name__ == "__main__":
    main()

"""PR-002 H2 判定统计：分条件（线索失准/有效）per-seed 适应中位数比较。

H2：线索失准块中 DC-Net-Cue 适应中位数较 Point-Net-Cue 缩短 >=25% 且 p<0.05。
闸门：cheap_learner 差距 <5pp 作废；环境失准率显著偏离 10% 作废。
"""
import glob
import json

import numpy as np
from scipy import stats

P_CUE_VALID = 0.9


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([np.mean(rng.choice(x, len(x))) for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
                 / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / sp)


def load(model, cond):
    """cond: 'invalid' | 'valid' | 'all'。返回 per-seed 适应中位数列表。"""
    medians, accs = [], []
    for f in sorted(glob.glob(f"results_cue/{model}_seed*.json")):
        d = json.load(open(f, encoding="utf-8"))
        vals = []
        for ep, ep_adapt in enumerate(d["adapt_trials_all"]):
            valid_row = d["switch_cue_valid_all"][ep]
            for i, v in enumerate(ep_adapt):
                if v is None:
                    continue
                if cond == "all" or \
                   (cond == "invalid") != bool(valid_row[i]):
                    vals.append(v)
        medians.append(float(np.median(vals)) if vals else np.nan)
        accs.append(d["block_accs_mean_last20"])
    return np.array(medians), np.array(accs)


def main():
    models = ["dc_net_cue", "point_net_cue", "random_mlp_cue", "cheap_learner_cue"]
    data = {m: {c: load(m, c) for c in ["all", "invalid", "valid"]} for m in models}
    print("== 各模型概览（per-seed）==")
    for m in models:
        acc = data[m]["all"][1]
        if len(acc) == 0:
            continue
        inv = data[m]["invalid"][0]
        val = data[m]["valid"][0]
        print(f"{m:<18} n={len(acc)} final_acc={np.nanmean(acc):.3f}  "
              f"adapt(失准)={np.nanmedian(inv):.1f}  adapt(有效)={np.nanmedian(val):.1f}")
    # 闸门
    gap = np.nanmean(data["dc_net_cue"]["all"][1]) - np.nanmean(data["cheap_learner_cue"]["all"][1])
    if gap < 0.05:
        print(f"\n[闸门1 触发] 差距 {gap:.1%} < 5pp：任务平凡，作废。")
        return
    # 闸门3：环境失准率全量汇总（~24 万样本）
    all_valid = []
    for f in glob.glob("results_cue/*_seed*.json"):
        d = json.load(open(f, encoding="utf-8"))
        all_valid.extend(v for row in d["switch_cue_valid_all"] for v in row)
    inv_rate = 1 - np.mean(all_valid)
    if abs(inv_rate - (1 - P_CUE_VALID)) > 0.05:
        print(f"\n[闸门3 触发] 失准率 {inv_rate:.1%} 偏离 10%（n={len(all_valid)}）：环境 bug，作废。")
        return
    print(f"\n[闸门3 通过] 全量失准率 {inv_rate:.2%}（n={len(all_valid)}，期望 10%）")
    dc, pn = data["dc_net_cue"]["invalid"][0], data["point_net_cue"]["invalid"][0]
    ok = (~np.isnan(dc)) & (~np.isnan(pn))
    dc, pn = dc[ok], pn[ok]
    u, p = stats.mannwhitneyu(dc, pn, alternative="less")
    med_dc, med_pn = np.median(dc), np.median(pn)
    reduction = (med_pn - med_dc) / med_pn * 100
    lo, hi = bootstrap_ci(dc)
    print(f"\n== H2 检验（线索失准块，per-seed 中位数 n={len(dc)} vs {len(pn)}）==")
    print(f"DC-Net-Cue median={med_dc:.1f} [{lo:.1f},{hi:.1f}] vs Point-Net-Cue median={med_pn:.1f}")
    print(f"reduction={reduction:.1f}%  MannWhitneyU p={p:.4g}  Cohen_d={cohens_d(dc, pn):.2f}")
    verdict = "H2 成立" if (reduction >= 25 and p < 0.05) else "H2 不成立（如实记探索性发现）"
    print("判定：" + verdict)
    # 辅助：线索有效块
    dcv, pnv = data["dc_net_cue"]["valid"][0], data["point_net_cue"]["valid"][0]
    okv = (~np.isnan(dcv)) & (~np.isnan(pnv))
    uv, pv = stats.mannwhitneyu(dcv[okv], pnv[okv], alternative="less")
    red_v = (np.median(pnv[okv]) - np.median(dcv[okv])) / np.median(pnv[okv]) * 100
    print(f"\n[辅助] 线索有效块: DC-Net median={np.median(dcv[okv]):.1f} vs "
          f"Point-Net median={np.median(pnv[okv]):.1f}  reduction={red_v:.1f}%  p={pv:.4g}")


if __name__ == "__main__":
    main()

"""PR-003 H3 判定统计：持续对抗性线索（p_cue_valid=0）下的适应速度比较。

H3：DC-Net-Cue per-seed 适应中位数（cond=all）较 Point-Net-Cue 缩短 >=25% 且 p<0.05。
闸门：1) cheap_learner 差距 <5pp 作废；2) 线索失准率须 100%（valid>2% 作废）；
      3) 至少一个结构模型 final_acc >= 0.6，否则作废。
"""
import glob
import json

import numpy as np
from scipy import stats

TAG = "results_cue_adv"


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([np.mean(rng.choice(x, len(x))) for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
                 / (len(a) + len(b) - 2))
    return float((np.mean(a) - np.mean(b)) / sp)


def load(model):
    medians, accs = [], []
    for f in sorted(glob.glob(f"{TAG}/{model}_seed*.json")):
        d = json.load(open(f, encoding="utf-8"))
        vals = [v for ep in d["adapt_trials_all"] for v in ep if v is not None]
        medians.append(float(np.median(vals)) if vals else np.nan)
        accs.append(d["block_accs_mean_last20"])
    return np.array(medians), np.array(accs)


def main():
    models = ["dc_net_cue", "point_net_cue", "random_mlp_cue", "cheap_learner_cue"]
    data = {m: load(m) for m in models}
    print("== 各模型概览（per-seed，cond=all）==")
    for m in models:
        med, acc = data[m]
        if len(acc) == 0:
            print(f"{m:<18} 无结果文件")
            continue
        print(f"{m:<18} n={len(acc)} final_acc={np.mean(acc):.3f}  "
              f"adapt_median={np.nanmedian(med):.1f}")
    # 闸门1：任务平凡
    gap = np.mean(data["dc_net_cue"][1]) - np.mean(data["cheap_learner_cue"][1])
    if gap < 0.05:
        print(f"\n[闸门1 触发] 差距 {gap:.1%} < 5pp：任务平凡，作废。")
        return
    # 闸门2：环境失准率须 100%
    all_valid = []
    for f in glob.glob(f"{TAG}/*_seed*.json"):
        d = json.load(open(f, encoding="utf-8"))
        all_valid.extend(v for row in d["switch_cue_valid_all"] for v in row)
    valid_rate = np.mean(all_valid)
    if valid_rate > 0.02:
        print(f"\n[闸门2 触发] 线索有效率 {valid_rate:.2%} > 2%（n={len(all_valid)}）：环境 bug，作废。")
        return
    print(f"\n[闸门2 通过] 全量线索失准率 {1-valid_rate:.2%}（n={len(all_valid)}，期望 100%）")
    # 闸门3：可学性
    struct_accs = [np.mean(data[m][1]) for m in models[:3]]
    if max(struct_accs) < 0.6:
        print(f"\n[闸门3 触发] 结构模型最高 final_acc 仅 {max(struct_accs):.3f} < 0.6：任务退化，作废。")
        return
    print(f"[闸门3 通过] 结构模型最高 final_acc = {max(struct_accs):.3f}")
    # H3 主检验
    dc, pn = data["dc_net_cue"][0], data["point_net_cue"][0]
    ok = (~np.isnan(dc)) & (~np.isnan(pn))
    dc, pn = dc[ok], pn[ok]
    u, p = stats.mannwhitneyu(dc, pn, alternative="less")
    med_dc, med_pn = np.median(dc), np.median(pn)
    reduction = (med_pn - med_dc) / med_pn * 100
    lo, hi = bootstrap_ci(dc)
    d_val = cohens_d(dc, pn)
    print(f"\n== H3 检验（持续对抗线索，per-seed 中位数 n={len(dc)} vs {len(pn)}）==")
    print(f"DC-Net-Cue median={med_dc:.1f} [{lo:.1f},{hi:.1f}] vs Point-Net-Cue median={med_pn:.1f}")
    print(f"reduction={reduction:.1f}%  MannWhitneyU p={p:.4g}  Cohen_d={d_val:.2f}")
    verdict = "H3 成立" if (reduction >= 25 and p < 0.05) else "H3 不成立（如实记探索性发现）"
    print("判定：" + verdict)
    # H3s 辅助：final_acc
    acc_dc, acc_pn = np.mean(data["dc_net_cue"][1]), np.mean(data["point_net_cue"][1])
    t, p_acc = stats.mannwhitneyu(data["dc_net_cue"][1], data["point_net_cue"][1],
                                  alternative="greater")
    print(f"\n[H3s 辅助] final_acc: DC={acc_dc:.3f} vs PN={acc_pn:.3f} "
          f"(差 {(acc_dc-acc_pn)*100:.1f}pp, MWU p={p_acc:.4g})")


if __name__ == "__main__":
    main()

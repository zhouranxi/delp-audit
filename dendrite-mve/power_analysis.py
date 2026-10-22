"""PR-001/PR-002 功效与最小可检出效应（MDE）分析。

问题：n=8 per-seed 中位数、单侧 Mann-Whitney U（alpha=0.05）下，
(a) 预注册阈值（中位数缩短 25%）的检验功效是多少？
(b) 80% 功效下最小可检出效应（MDE）是多少？
(c) 观测效应（~7.1%）的功效是多少？

方法：以 Point-Net per-seed 适应中位数的经验分布为参照，模拟 DC 样本为
f x PN 自助抽样（f=DC/PN 中位数比），每档 20000 次模拟估计功效。
"""
import glob
import json

import numpy as np
from scipy import stats

N_SIM = 20000
RNG = np.random.default_rng(20261022)


def per_seed_medians(pattern, cond="all"):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f, encoding="utf-8"))
        vals = []
        for ep, ep_adapt in enumerate(d["adapt_trials_all"]):
            rows = d.get("switch_cue_valid_all")
            for i, v in enumerate(ep_adapt):
                if v is None:
                    continue
                if cond == "all" or (cond == "invalid") != bool(rows[ep][i]):
                    vals.append(v)
        if vals:
            out.append(float(np.median(vals)))
    return np.array(out)


def power_at(pn, f, n_sim=N_SIM):
    """DC 样本 = f x PN 自助抽样时的单侧检验功效。"""
    hits = 0
    for _ in range(n_sim):
        b = RNG.choice(pn, size=len(pn), replace=True)
        u, p = stats.mannwhitneyu(f * b, b, alternative="less")
        hits += p < 0.05
    return hits / n_sim


def analyse(name, dc, pn):
    print(f"\n===== {name} =====")
    print(f"per-seed: DC n={len(dc)} med={np.median(dc):.1f} | "
          f"PN n={len(pn)} med={np.median(pn):.1f}")
    obs_f = np.median(dc) / np.median(pn)
    u, p = stats.mannwhitneyu(dc, pn, alternative="less")
    print(f"复现观测：reduction={(1-obs_f)*100:.1f}%  p={p:.4g}")
    # 预注册阈值 25% 的功效（f=0.75）
    pw_thresh = power_at(pn, 0.75, 20000)
    print(f"(a) 预注册阈值 25%（f=0.75）功效 = {pw_thresh:.1%}")
    # MDE：80% 功效对应的 f（二分搜索 0.40–1.00）
    lo, hi = 0.40, 1.00
    for _ in range(18):
        mid = (lo + hi) / 2
        if power_at(pn, mid, 8000) >= 0.80:
            lo = mid
        else:
            hi = mid
    mde_f = (lo + hi) / 2
    mde_pct = (1 - mde_f) * 100
    pw_mde = power_at(pn, mde_f, 20000)
    print(f"(b) MDE（80% 功效）= 缩短 {mde_pct:.1f}%（f={mde_f:.3f}，校验功效 {pw_mde:.1%}）")
    # 观测效应的功效
    pw_obs = power_at(pn, obs_f, 20000)
    print(f"(c) 观测效应 {((1-obs_f)*100):.1f}%（f={obs_f:.3f}）功效 = {pw_obs:.1%}")
    return {"name": name, "dc_med": float(np.median(dc)), "pn_med": float(np.median(pn)),
            "obs_reduction": float((1 - obs_f) * 100), "obs_p": float(p),
            "power_at_25": float(pw_thresh), "mde_pct": float(mde_pct),
            "power_at_obs": float(pw_obs)}


def main():
    res = []
    # PR-001：无线索，全条件
    dc1 = per_seed_medians("results_full/dc_net_seed*.json")
    pn1 = per_seed_medians("results_full/point_net_seed*.json")
    res.append(analyse("PR-001（无线索，cond=all）", dc1, pn1))
    # PR-002：线索，失准块条件
    dc2 = per_seed_medians("results_cue_rerun/dc_net_cue_seed*.json", cond="invalid")
    pn2 = per_seed_medians("results_cue_rerun/point_net_cue_seed*.json", cond="invalid")
    res.append(analyse("PR-002（10% 失准线索，cond=invalid）", dc2, pn2))
    json.dump(res, open("power_analysis.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nsaved power_analysis.json")


if __name__ == "__main__":
    main()

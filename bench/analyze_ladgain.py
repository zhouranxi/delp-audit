"""analyze_ladgain.py -- the ladder, with every rung tuned on gain AND lambda.

The question this answers
-------------------------
The first pass of the ladder tuned lambda only, holding each rung's own free
scale at 1.0.  On that protocol the only surviving mechanism was the usage
weighting (+1.79 at T=20, p=0.005) and the activation change dominated
everything (+5.37 at T=5).  But the peer-gain probe showed that the scale knob is
worth +3 to +10 points and is worth MORE to the dendritic arm than to any
baseline, so a ladder measured without touching it compares rungs that are not
equally tuned.  This script redoes the comparison on a joint (gain x lambda)
grid, selected on held-out accuracy, and reports what each step is worth under
equal tuning effort.

Two selections are reported side by side, because the honest output is the
DIFFERENCE between them:
  gain=1   the old protocol (lambda only)
  joint    best (gain, lambda) per rung per T

Statistics
----------
Paired t-test on seeds (n=5, df=4), two-sided, exact t via the incomplete beta
function so no scipy is needed on the cluster.  Selection is on the held-out
validation split; the test split is only ever read after the cell is chosen.
Selection over 36 cells per rung inflates the maximum, so the top-3 mean is
reported next to the top-1 for every rung; a claim that only survives as a
single lucky cell will show up as a large gap between the two columns.

Usage
    python analyze_ladgain.py [results_dir] > ladgain.md
"""
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

ORDER = ["mlp_ewc", "mlp_nosig", "ad_nocx", "ad_ctx", "ad_prot", "ad_usage",
         "ad_usage_ce"]
SHORT = {"mlp_ewc": "R0  mlp+EWC", "mlp_nosig": "R0b kwta(0.5t)",
         "ad_nocx": "R1  +dendrites(ctx=0)", "ad_ctx": "R2  +context",
         "ad_prot": "R3  +protection", "ad_usage": "R4  +usage (=ours)",
         "ad_usage_ce": "R5  R4 with CE Fisher"}
STEPS = [("mlp_ewc", "mlp_nosig", "activation: kwta(sigmoid(t)) -> kwta(0.5t)"),
         ("mlp_nosig", "ad_nocx", "dendrite capacity alone (null control)"),
         ("ad_nocx", "ad_ctx", "context modulation"),
         ("ad_ctx", "ad_prot", "branch protection"),
         ("ad_prot", "ad_usage", "usage-weighted Fisher"),
         ("ad_usage", "ad_usage_ce", "proxy Fisher -> true-label CE Fisher")]
TS = (5, 10, 20)


# --------------------------------------------------------------- exact t test
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb + a * math.log(x) + b * math.log1p(-x)) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_paired(a, b):
    """Two-sided p for a paired t-test. Returns (mean_diff, t, p)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return (st.mean(d) if d else float("nan"), float("nan"), float("nan"))
    md = st.mean(d)
    sd = st.stdev(d)
    if sd == 0:
        return md, float("inf") if md else 0.0, 0.0 if md else 1.0
    t = md / (sd / math.sqrt(n))
    return md, t, _betai(0.5 * (n - 1), 0.5, (n - 1) / ((n - 1) + t * t))


def sem(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


# ------------------------------------------------------------------- loading
def load(folder):
    """cells[(rung, T)][(gain, lam)][seed] = rec"""
    cells = defaultdict(lambda: defaultdict(dict))
    bad = []
    nfile = 0
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        nfile += 1
        try:
            d = json.load(open(f))
        except Exception:
            bad.append(os.path.basename(f) + " (unreadable)")
            continue
        if "error" in d:
            bad.append(os.path.basename(f) + " (" + d["error"][:40] + ")")
            continue
        if "final_avg" not in d or "val_final_avg" not in d:
            bad.append(os.path.basename(f) + " (no val)")
            continue
        # the two families carry their own knob; a cell is defined by the pair
        gain = d.get("hgain", 1.0) if d["arm"].startswith("ad_") else d.get("hsig", 1.0)
        cells[(d["arm"], d["ntask"])][(float(gain), float(d["lam"]))][int(d["seed"])] = d
    return cells, nfile, bad


def mean_at(cell, nseed=5):
    """(acc, bwt, val) means over seeds, or None if incomplete."""
    if len(cell) < nseed:
        return None
    return (st.mean(r["final_avg"] for r in cell.values()),
            st.mean(r["bwt"] for r in cell.values()),
            st.mean(r["val_final_avg"] for r in cell.values()))


def select(cells, rung, T, only_gain1=False, nseed=5):
    """Pick (gain, lambda) by held-out accuracy. Returns (gain, lam, m, cell) or None."""
    grid = cells.get((rung, T))
    if not grid:
        return None
    best = None
    for (g, lam), cell in grid.items():
        if only_gain1 and g != 1.0:
            continue
        m = mean_at(cell, nseed)
        if m is None:
            continue
        if best is None or m[2] > best[2][2]:
            best = (g, lam, m, cell)
    return best


def topk(cells, rung, T, k=3, nseed=5):
    grid = cells.get((rung, T)) or {}
    rows = []
    for (g, lam), cell in grid.items():
        m = mean_at(cell, nseed)
        if m is not None:
            rows.append((m[2], g, lam, m))
    rows.sort(reverse=True)
    if not rows:
        return None
    k = min(k, len(rows))
    return (st.mean(r[3][0] for r in rows[:k]), st.mean(r[3][1] for r in rows[:k]),
            [(r[1], r[2], r[3][0]) for r in rows[:k]])


# ------------------------------------------------------------------ plumbing
def plumbing_check(cells, verbose=True):
    """mlp_nosig(HSIG=g) and ad_nocx(HGAIN=g) compute the same function.

    Both reduce to kwta(0.5*g*t) (the dendrite layer has no bias, so sigmoid(0)
    is exactly 0.5), and every gain in the grid is a power of two, so both forms
    are exact in float32 and must agree BIT FOR BIT -- the full accuracy matrix,
    not just the average.  Any mismatch is a bug in the gain plumbing, not a
    scientific result, so it is checked before anything else is reported.
    """
    tot = same = 0
    diffs = []
    for T in TS:
        a = cells.get(("mlp_nosig", T)) or {}
        b = cells.get(("ad_nocx", T)) or {}
        for key in set(a) & set(b):
            for s in set(a[key]) & set(b[key]):
                tot += 1
                ra, rb = a[key][s], b[key][s]
                if ra.get("acc_matrix") == rb.get("acc_matrix"):
                    same += 1
                else:
                    diffs.append((T, key, s))
    if verbose:
        print(f"## 0  Data integrity\n")
        print(f"* 完整性检查（mlp_nosig 与 ad_nocx 在同一 gain 下必须逐位相同）："
              f"**{same}/{tot}** 对一致" + (" ✅" if same == tot and tot else ""))
        if diffs:
            print(f"* ⚠️ 不一致示例（说明增益通路有 bug，先修再谈结论）：{diffs[:5]}")
    return tot, same, diffs


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results_ladgain"
    cells, nfile, bad = load(folder)
    print(f"# 阶梯消融（联合 gain × λ 网格）：每一级都站在自己的最优尺度上\n")
    plumbing_check(cells)
    ncell = sum(len(v) for v in cells.values())
    print(f"* 网格规模：{nfile} 个文件，{len(cells)} 个 (rung,T) 组合，"
          f"{ncell} 个 (gain,λ) 格子 × 5 seeds")
    if bad:
        print(f"* ⚠️ 跳过 {len(bad)} 个文件：{bad[:4]}")

    # ------------------------------------------------ 1 joint vs lambda-only
    print("\n## 1  联合调优 vs 只调 λ：协议本身值多少\n")
    print("| T | rung | λ*(gain=1) | acc | gain* | λ* | acc | Δacc (联合−旧协议) |")
    print("|---|---|---|---|---|---|---|---|")
    joint = {}
    old = {}
    for T in TS:
        for r in ORDER:
            o = select(cells, r, T, only_gain1=True)
            j = select(cells, r, T)
            if not o or not j:
                continue
            old[(r, T)] = o
            joint[(r, T)] = j
            print(f"| {T} | {SHORT[r]} | {o[1]:g} | {o[2][0]:.2f} | {j[0]:g} | "
                  f"{j[1]:g} | {j[2][0]:.2f} | {j[2][0]-o[2][0]:+.2f} |")

    # ------------------------------------------------- 1b edge audit
    # A winner sitting on the last row or column of the grid is a truncation
    # artefact, not a measurement -- the same rule that invalidated the old
    # lambda grids.  Both knobs have to be checked, not just lambda.
    print("\n## 1b 边界审查：最优解是不是被网格截断了\n")
    allg = sorted({d["hgain"] for (r, T), g in cells.items()
                   for (gg, _) in g for d in [{"hgain": gg}]})
    alll = sorted({lam for (r, T), g in cells.items() for (_, lam) in g})
    gmin, gmax = min(allg), max(allg)
    lmin, lmax = min(l for l in alll if l > 0), max(alll)
    print(f"增益网格 [{gmin:g}, {gmax:g}]，λ 网格 [{lmin:g}, {lmax:g}]")
    print()
    print("| T | rung | gain* | 增益贴边 | λ* | λ 贴边 |")
    print("|---|---|---|---|---|---|")
    n_edge_g = n_edge_l = 0
    for T in TS:
        for r in ORDER:
            j = joint.get((r, T))
            if not j:
                continue
            eg = "**顶边**" if j[0] >= gmax else ("底边" if j[0] <= gmin else "内部")
            el = "**顶边**" if j[1] >= lmax else ("底边" if j[1] <= lmin else "内部")
            n_edge_g += eg != "内部"
            n_edge_l += el != "内部"
            print(f"| {T} | {SHORT[r]} | {j[0]:g} | {eg} | {j[1]:g} | {el} |")
    n = len(joint)
    print(f"\n* 增益贴边 **{n_edge_g}/{n}**，λ 贴边 **{n_edge_l}/{n}**")
    if n_edge_g:
        print(f"* ⚠️ 增益顶边 {n_edge_g} 格 ⇒ 第 1/2/3 节的\"联合最优\"是**下界**，"
              f"必须扩增益重测才能用。")

    # -------------------------------------------- 2 what each step is worth
    for tag, sel in (("联合网格（每级各自最优 gain×λ）", joint),
                     ("仅调 λ（旧协议，gain 固定 1）", old)):
        print(f"\n## 2  每一步机制值多少 —— {tag}\n")
        for T in TS:
            have = [r for r in ORDER if (r, T) in sel]
            if len(have) < 2:
                continue
            print(f"\n### T = {T}\n")
            print("| step | dAcc | p | dBWT | p | 判读 |")
            print("|---|---|---|---|---|---|")
            for a, b, label in STEPS:
                if (a, T) not in sel or (b, T) not in sel:
                    continue
                ca, cb = sel[(a, T)][3], sel[(b, T)][3]
                seeds = sorted(set(ca) & set(cb))
                if len(seeds) < 3:
                    continue
                da = t_paired([cb[s]["final_avg"] for s in seeds],
                              [ca[s]["final_avg"] for s in seeds])
                ba = t_paired([cb[s]["bwt"] for s in seeds],
                              [ca[s]["bwt"] for s in seeds])
                if abs(da[0]) < 0.05:
                    read = "零（结构性，见 §0）"
                elif da[2] < 0.05:
                    read = "**显著**" if da[0] > 0 else "**显著为负**"
                else:
                    read = "不显著"
                print(f"| {label} | {da[0]:+.2f} | {da[2]:.3f} | "
                      f"{ba[0]:+.2f} | {ba[2]:.3f} | {read} |")

    # ------------------------------------------------- 3 gain sensitivity
    print("\n## 3  增益这条旋钮对各 rung 值多少\n")
    print("| T | rung | gain=1 最优 acc | 联合最优 acc | 增益带来的增益 |")
    print("|---|---|---|---|---|")
    for T in TS:
        for r in ORDER:
            if (r, T) not in old or (r, T) not in joint:
                continue
            o, j = old[(r, T)], joint[(r, T)]
            print(f"| {T} | {SHORT[r]} | {o[2][0]:.2f} | {j[2][0]:.2f} | "
                  f"{j[2][0]-o[2][0]:+.2f} |")

    # ------------------------------------------------- 4 winner's curse
    print("\n## 4  稳健性：top-1 是不是一个幸运格子\n")
    print("| T | rung | top-1 acc | top-3 均值 | 落差 |")
    print("|---|---|---|---|---|")
    for T in TS:
        for r in ORDER:
            k = topk(cells, r, T, 3)
            j = joint.get((r, T))
            if not k or not j:
                continue
            print(f"| {T} | {SHORT[r]} | {j[2][0]:.2f} | {k[0]:.2f} | "
                  f"{j[2][0]-k[0]:+.2f} |")

    # ------------------------------------------------- 5 heat tables
    for T, rungs in ((20, ["mlp_nosig", "ad_usage"]),):
        for r in rungs:
            grid = cells.get((r, T)) or {}
            if not grid:
                continue
            gains = sorted({g for g, _ in grid})
            lams = sorted({l for _, l in grid})
            print(f"\n## 5  T={T} {SHORT[r]} 的 acc 热表（行=gain，列=λ）\n")
            print("| gain \\ λ | " + " | ".join(f"{l:g}" for l in lams) + " |")
            print("|---|" + "---|" * len(lams))
            for g in gains:
                row = []
                for l in lams:
                    m = mean_at(grid.get((g, l), {}))
                    row.append(f"{m[0]:.1f}" if m else "·")
                print(f"| {g:g} | " + " | ".join(row) + " |")

    # ------------------------------------------------- 6 bottom line
    print("\n## 6  结论\n")
    for T in TS:
        if ("ad_usage", T) not in joint or ("mlp_nosig", T) not in joint:
            continue
        cu, cn = joint[("ad_usage", T)][3], joint[("mlp_nosig", T)][3]
        seeds = sorted(set(cu) & set(cn))
        d = t_paired([cu[s]["final_avg"] for s in seeds],
                     [cn[s]["final_avg"] for s in seeds])
        print(f"* T={T}: ours(联合最优) {joint[('ad_usage', T)][2][0]:.2f} vs "
              f"无树突对照 {joint[('mlp_nosig', T)][2][0]:.2f} = {d[0]:+.2f} "
              f"(p={d[2]:.3f})")


if __name__ == "__main__":
    main()

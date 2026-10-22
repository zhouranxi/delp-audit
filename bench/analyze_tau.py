"""analyze_tau.py -- is the context horizon set by the leak, and with what exponent?

The pre-registration (plan section 2.4f) said: if the horizon is set by the
leak, then h(tau) = c / (1 - tau) -- a straight line with slope 1 on a
log-log plot of h against 1/(1-tau), and the SAME c for every architecture.
The falsification condition was "c is not constant across tau".

d2wm.md reports c = h(1-tau) falling from 2.60 to 0.15 as tau goes 0.9 -> 0.999,
which would mean the linear law is wrong. But those h values come from a curve
FIT, and a fit can hide the trend. So this script measures the horizon the
crude, assumption-free way instead:

    h = the gap at which mean accuracy crosses HORIZON_ACC (45, i.e. halfway
        between the no-memory floor 33.3 and 100), by log-linear
        interpolation between the two adjacent tested gaps.

then fits log h = log c + b log(1/(1-tau)) and reports b with its R^2.

b = 1    -> the pre-registered linear leak law holds.
b < 1    -> the horizon grows sublinearly in the leak timescale, and the
            amount of "extra hold" bought by a longer tau is much less than
            the leak would suggest. That is a falsification of the law as
            stated, and it is reported as such.

Point_slow at tau=0.97 is duplicated in the results (an earlier 5-seed run and
the sweeps' 5 seeds); both are kept, since they are independent seeds.

usage
    python analyze_tau.py results_d2_wm
"""
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

GAPS = [2, 10, 40, 80, 160, 320]
CROSS = 45.0          # halfway between the 33.3 floor and 100
CHANCE = 100.0 / 3.0


def mean(v):
    return st.mean(v) if v else float("nan")


def crossing(curve):
    """Gap where accuracy crosses CROSS, log-linear between adjacent gaps.

    Returns (h, kind) where kind is 'measured' for an interpolated crossing,
    'below' if the model is already under CROSS at the shortest gap (h is then
    an upper bound of the first gap) and 'above' if it is still over CROSS at
    the longest gap (h is a lower bound, i.e. "beyond the tested range").
    """
    gs = [g for g in GAPS if g in curve]
    if not gs:
        return float("nan"), "none"
    a = [mean(curve[g]) for g in gs]
    if a[0] <= CROSS:
        return float(gs[0]), "below"
    for i in range(1, len(gs)):
        if a[i] <= CROSS < a[i - 1]:
            # linear in gap (not in log gap): the tested grid is nearly log
            # spaced, and log-linear extrapolation over one octave is the
            # gentler assumption of the two.
            f = (a[i - 1] - CROSS) / (a[i - 1] - a[i])
            return math.exp(math.log(gs[i - 1]) +
                            f * (math.log(gs[i]) - math.log(gs[i - 1]))), "measured"
    return float(gs[-1]), "above"


def linfit(x, y):
    n = len(x)
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sxx
    a0 = my - b * mx
    ss_res = sum((yi - (a0 + b * xi)) ** 2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return a0, b, r2


def main(folder):
    by = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        d = json.load(open(f))
        if "error" in d or d["arm"] not in ("point_slow", "dual"):
            continue
        by[(d["arm"], d.get("couple", "-"), float(d["tau_a"]))][d["gap"]].append(d["acc"])

    print("# 方向二：视界是不是由泄漏决定的（τ_a 定量律，可复现版）\n")
    print(f"视界定义：acc 穿过 {CROSS:g}% 的 gap（对数线性插值）。"
          f"无记忆地板 33.3%。\n")
    print("## 1  实测视界（不用任何曲线拟合假设）\n")
    print("| arm | τ_a | 1/(1−τ) | 视界 h | 类型 | h·(1−τ) |")
    print("|---|---|---|---|---|---|")
    table = {}
    for key in sorted(by, key=lambda k: (k[0], k[2])):
        arm, couple, tau = key
        h, kind = crossing(by[key])
        table[key] = (h, kind)
        inv = 1.0 / (1.0 - tau) if tau < 1 else float("inf")
        tag = {"measured": "插值", "above": ">320（超范围）",
               "below": "<2（已归零）"}.get(kind, kind)
        print(f"| {arm}{'' if couple == '-' else '/' + couple} | {tau:g} | {inv:6.1f} | "
              f"{h:6.0f} | {tag} | {h * (1 - tau):.2f} |")

    print("\n## 2  拟合 log h = log c + b·log(1/(1−τ))\n")
    print("| arm | n 点 | b（指数） | R² | 预注册 b=1 | 判读 |")
    print("|---|---|---|---|---|---|")
    for arm in sorted({k[0] for k in table}):
        pts = [(math.log(1.0 / (1.0 - tau)), math.log(h))
               for (a, c, tau), (h, kind) in table.items()
               if a == arm and kind == "measured" and h > 0]
        if len(pts) < 3:
            print(f"| {arm} | {len(pts)} | — | — | — | 点太少，无法拟合 |")
            continue
        a0, b, r2 = linfit([p[0] for p in pts], [p[1] for p in pts])
        if b < 0.85:
            verdict = f"**推翻 b=1**，视界按 1/(1−τ) 的 **{b:.2f} 次幂**增长"
        elif b > 1.15:
            verdict = f"**超线性**（b={b:.2f}），比泄漏预测保持得更远"
        else:
            verdict = "与 b=1 相容，预注册律成立"
        print(f"| {arm} | {len(pts)} | **{b:.2f}** | {r2:.3f} | b=1 | {verdict} |")

    print("\n## 3  这条律的实际含义：把 τ 换成 h，需要多少代价\n")
    print("| arm | τ_a | 1/(1−τ) 相对 0.97 | 视界相对 0.97 | 弹性 dh/dτ |")
    print("|---|---|---|---|---|")
    for arm in sorted({k[0] for k in table}):
        base = table.get((arm, "-", 0.97)) or table.get((arm, "add", 0.97))
        bh = base[0] if base else float("nan")
        for (a, c, tau), (h, kind) in sorted(table.items(), key=lambda kv: kv[0][2]):
            if a != arm or kind != "measured" or bh != bh or bh <= 0:
                continue
            inv = 1.0 / (1.0 - tau)
            print(f"| {arm} | {tau:g} | {inv/33.3:5.1f}× | {h/bh:5.2f}× | "
                  f"{(h/bh - 1) / (inv/33.3 - 1) if inv > 33.3 else float('nan'):.2f} |")

    print("\n**读法**：弹性列 <1 的意思是「把泄漏时间常数拉长 N 倍，只能买到不到 N 倍的视界」。")
    print("如果弹性随 τ 上升而下降，说明存在一个与泄漏无关的天花板，")
    print("那么「继续加大 τ」不是一条可用的路线——必须换机制（见 §2.4g）。\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_d2_wm")

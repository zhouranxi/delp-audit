"""analyze_wm.py -- did we move the context horizon, and did we move it for the right reason?

Three questions, three tables.

1. tau_a law.  If the ~40-step horizon is the leak, then the hazard is linear in
   the decay, so the crossing point should scale as c/(1-tau_a) with c the same
   constant for every architecture. With c = 1.2 (40 steps at tau=0.97) the
   prediction is: tau 0.90 -> 12, 0.97 -> 40, 0.99 -> 120, 0.995 -> 240,
   0.999 -> 1200 (i.e. past the end of the ladder). If the measured crossings
   follow that ordering and the fitted c is flat across architectures, the
   horizon is a leak property and "40 steps" becomes a prediction, not a
   limitation. If c drifts, something else is setting the ceiling.

2. Write gate.  A gated state is compared against a capacity-matched leaky arm
   (matched by parameter count, not by width). The mechanism claim needs the
   ablation columns, not the accuracy column: `acc_gatefreeze` freezes the gate
   at its own mean (keeps every parameter, removes the input-dependence) and
   `acc_gateone` holds it fully open (keeps the extra input->state pathway,
   removes the memory). If the trained accuracy is much higher than both, the
   learned gate is the mechanism. Any other pattern means the gain came from
   capacity or from the extra pathway, and must be reported as such.

3. The horizon is fitted, not thresholded. The crossing point comes from a
   logistic fit with the floor pinned at chance (33.3), so a single noisy cell
   at gap=80 cannot define it. The >=45 rule is reported alongside for continuity
   with the earlier tables.

usage
    python analyze_wm.py [results_dir]
"""
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

# The no-memory floor is 1/C, not 1/(C*M): the sensory template is directly
# visible in the readout window, so a model with no memory of the cue can
# always classify the template and then has to guess the context -> 1/3.
# Measured: 33.0 both with the gate frozen and with it held open.
CHANCE = 100.0 / 3.0
C_LEAK = 1.2                  # calibrated so tau=0.97 gives the observed ~40 steps
TAUS = [0.90, 0.97, 0.99, 0.995, 0.999]
GAPS = [2, 10, 40, 80, 160, 320]


def load(folder):
    rows = defaultdict(list)      # (arm, tau, nh, gap) -> [rec]
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "acc" not in d:
            continue
        rows[(d["arm"], float(d.get("tau_a", 0.97)), int(d.get("nparam", 0)),
              int(d["gap"]))].append(d)
    return rows


def fit_mid(gaps, accs, nparam=None):
    """Logistic crossing: acc = lo + (hi-lo)/(1+exp((gap-mid)/s)), lo=chance.

    Fitted by grid search (no scipy). Returns (mid, rms, hi). mid is the gap at
    which accuracy is halfway between chance and its own peak, which is the
    honest definition of a horizon here: it does not depend on which gap points
    happened to be sampled.
    """
    if len(gaps) < 3:
        return float("nan"), float("nan"), float("nan")
    lo = CHANCE
    hi0 = max(accs)
    if hi0 - lo < 2:
        return float("nan"), float("nan"), hi0
    best = (float("nan"), 1e9, hi0)
    for hi in [hi0, min(100.0, hi0 + 3), min(100.0, hi0 + 8)]:
        if hi <= lo:
            continue
        for mid in [g for g in range(2, 700, 2)]:
            for s in [3, 5, 8, 12, 18, 26, 36, 50, 70]:
                err = 0.0
                for g, a in zip(gaps, accs):
                    pred = lo + (hi - lo) / (1 + math.exp((g - mid) / s))
                    err += (pred - a) ** 2
                rms = math.sqrt(err / len(gaps))
                if rms < best[1]:
                    best = (mid, rms, hi)
    return best


def mean(v):
    return st.mean(v) if v else float("nan")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results_d2_wm"
    rows = load(folder)
    if not rows:
        print("no results yet")
        return
    print("# 方向二：上下文的保持视界是谁定的（τ_a 律 + 写入门控）\n")
    print(f"读出固定为最后 15 步（诚实仪器），机会水平 {CHANCE:.1f}%。\n")

    by = defaultdict(lambda: defaultdict(list))
    for (arm, tau, npar, gap), recs in rows.items():
        by[(arm, tau, npar)][gap].extend(r["acc"] for r in recs)

    # ------------------------------------------------ 1 tau_a law
    print("## 1  τ_a 剂量–响应：视界是不是泄漏决定的\n")
    print("| arm | τ_a | 1/(1−τ) | 观测 acc @g=40 | @g=160 | 拟合视界 | 95% 判据视界 | 拟合 c=h(1−τ) |")
    print("|---|---|---|---|---|---|---|---|")
    law = {}
    for (arm, tau, npar), gs in sorted(by.items()):
        if not arm.startswith(("point", "dual")) or arm.endswith(("wm", "wmh")):
            continue
        if npar > 20000:
            continue                       # capacity control, handled in §3
        gaps = [g for g in GAPS if g in gs and len(gs[g]) >= 3]
        if len(gaps) < 3:
            continue
        accs = [mean(gs[g]) for g in gaps]
        mid, rms, hi = fit_mid(gaps, accs)
        thr = max([g for g in gaps if mean(gs[g]) >= 45] or [0])
        law[(arm, tau)] = (mid, thr, hi)
        c = mid * (1 - tau) if mid == mid else float("nan")
        print(f"| {arm} | {tau:g} | {1/(1-tau):6.1f} | "
              f"{mean(gs.get(40, [float('nan')])):6.1f} | "
              f"{mean(gs.get(160, [float('nan')])):6.1f} | "
              f"{mid:6.0f} | {thr:4d}{'+' if thr == max(GAPS) else ' '} | "
              f"{c:5.2f} |")
    cs = []
    for (arm, tau), (mid, thr, hi) in law.items():
        if mid == mid:
            cs.append(mid * (1 - tau))
    if cs:
        print(f"\n* 拟合常数 c = h·(1−τ_a)：中位 {st.median(cs):.2f}，"
              f"范围 {min(cs):.2f}–{max(cs):.2f}")
        print(f"* 若 c 在各架构间基本不变 ⇒ 视界由泄漏决定，"
              f"可写成 h ≈ {st.median(cs):.1f}/(1−τ_a) 的定量律。")

    # ------------------------------------------------ 2 write gate
    print("\n## 2  写入门控：机制来自门，还是来自多出来的参数\n")
    print("| arm | n_param | acc@g=40 | acc@g=160 | 冻结门(均值) | 门全开 | gate 线索/间隙 |")
    print("|---|---|---|---|---|---|---|")
    gate_rows = []
    for (arm, tau, npar), gs in sorted(by.items()):
        if not (arm.endswith("wm") or arm.endswith("wmh")):
            continue
        recs = [r for gap in gs for r in rows[(arm, tau, npar, gap)]]
        fr = mean([r["acc_gatefreeze"] for r in recs if "acc_gatefreeze" in r])
        on = mean([r["acc_gateone"] for r in recs if "acc_gateone" in r])
        gc = mean([r["gate_cue"] for r in recs if "gate_cue" in r])
        gg = mean([r["gate_gap"] for r in recs if "gate_gap" in r])
        gate_rows.append((arm, tau, npar, gs, fr, on))
        print(f"| {arm} | {npar} | "
              f"{mean(gs.get(40, [float('nan')])):6.1f} | "
              f"{mean(gs.get(160, [float('nan')])):6.1f} | "
              f"{fr:6.1f} | {on:6.1f} | {gc:.2f}/{gg:.2f} |")

    # ------------------------------------------------ 3 capacity control
    print("\n## 3  容量对照：同样参数量的泄漏臂能不能做到\n")
    print("| arm | n_param | @g=40 | @g=80 | @g=160 | 拟合视界 |")
    print("|---|---|---|---|---|---|")
    for (arm, tau, npar), gs in sorted(by.items()):
        if tau != 0.97:
            continue
        gaps = [g for g in GAPS if g in gs and len(gs[g]) >= 3]
        if len(gaps) < 3:
            continue
        accs = [mean(gs[g]) for g in gaps]
        mid, rms, hi = fit_mid(gaps, accs)
        print(f"| {arm} | {npar} | "
              f"{mean(gs.get(40, [float('nan')])):6.1f} | "
              f"{mean(gs.get(80, [float('nan')])):6.1f} | "
              f"{mean(gs.get(160, [float('nan')])):6.1f} | {mid:6.0f} |")

    # ------------------------------------------------ 4 verdicts
    print("\n## 4  判读\n")
    for (arm, tau, npar), gs in sorted(by.items()):
        if not (arm.endswith("wm") or arm.endswith("wmh")):
            continue
        recs = [r for gap in gs for r in rows[(arm, tau, npar, gap)]]
        a40 = mean(gs.get(40, [float("nan")]))
        fr = mean([r["acc_gatefreeze"] for r in recs if "acc_gatefreeze" in r])
        on = mean([r["acc_gateone"] for r in recs if "acc_gateone" in r])
        verdict = ("门是机制（训练后远高于两个消融）" if a40 - max(fr, on) > 10
                   else "**增益不是门带来的**（消融没掉下来，疑为容量或通路）")
        print(f"* {arm}: @g=40 训练 {a40:.1f} vs 冻结门 {fr:.1f} vs 门全开 {on:.1f} "
              f"→ {verdict}")


if __name__ == "__main__":
    main()

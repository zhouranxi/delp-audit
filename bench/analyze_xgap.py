"""analyze_xgap.py -- cross-gap generalisation table for D2.

Reads results_d2_xgap/*.json (written by d2_xgap.py: one training run per
(arm, seed), evaluated at every gap in --test-gaps) and answers one question:

    does the arm hold the cue for a RANGE of delays, or does it only survive
    the delay it was trained on?

The same-gap grid in results_d2_wm cannot tell those apart, because every cell
trains and tests at one gap. Here the training gap is fixed at 40 and the
weights are reused, so a curve that is flat near 40 and falls off on both
sides is a tuned-to-the-delay fit, while a curve that decays monotonically
away from 40 (and is high at short gaps) is a real leak.

usage
    python analyze_xgap.py results_d2_xgap
"""
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

GAPS = [2, 10, 40, 80, 160, 320]
CHANCE = 100.0 / 3.0
TRAIN_GAP = 40


def mean(v):
    return st.mean(v) if v else float("nan")


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def main(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        d = json.load(open(f))
        if "error" in d:
            print(f"* ⚠️ {os.path.basename(f)}: {d['error']}")
            continue
        rows.append(d)
    if not rows:
        print("no rows")
        return

    print("# 方向二：跨延迟泛化（训练只用一个 gap，权重复用到所有 gap）\n")
    print(f"训练 gap 固定为 **{TRAIN_GAP}**，读出为最后 15 步，机会水平 33.3%。\n")

    # arm -> gap -> [acc]
    by = defaultdict(lambda: defaultdict(list))
    for d in rows:
        key = (d["arm"], d.get("couple", "-"), float(d.get("tau_a", 0.0)))
        for g, v in d["curve"].items():
            by[key][int(g)].append(v["acc"])

    print("## 1  逐格结果（均值 ± sd，n=种子数）\n")
    hdr = " | ".join(f"{g}" for g in GAPS)
    print(f"| arm | n | {hdr} | 训练点排名 | 离训点落差 |")
    print("|---|---|" + "---|" * len(GAPS) + "---|---|")
    verdict = {}
    for key in sorted(by, key=lambda k: (k[0], k[2])):
        arm, couple, tau = key
        gs = by[key]
        n = max(len(gs[g]) for g in gs)
        cells, means = [], []
        for g in GAPS:
            v = gs.get(g, [])
            m = mean(v)
            means.append(m)
            cells.append(f"{m:.1f}±{sd(v):.1f}" if v else "·")
        ranked = sorted(range(len(GAPS)), key=lambda i: -means[i])
        at_train = means[GAPS.index(TRAIN_GAP)]
        # how much is lost by moving one octave either side of the training gap
        near = [means[GAPS.index(g)] for g in (10, 80)]
        drop = mean(near) - at_train if all(m == m for m in near) else float("nan")
        verdict[key] = dict(means=means, train_rank=ranked.index(GAPS.index(TRAIN_GAP)) + 1,
                            drop=drop, n=n)
        print(f"| {arm} | {n} | " + " | ".join(cells) +
              f" | {ranked.index(GAPS.index(TRAIN_GAP)) + 1} | {drop:+.1f} |")

    print("\n## 2  判别：这是记忆，还是把训练延迟背下来了\n")
    print("判据（先验设定，不是事后拟合）：\n")
    print("* **真记忆**：训练点不是孤峰。短延迟一侧应接近或高于训练点，")
    print("  长延迟一侧单调衰减，且相邻倍的落差与泄漏臂同量级（≤ 一个数量级的 acc）。")
    print("* **背延迟**：训练点是**孤峰**——短延迟一侧也掉到机会水平。")
    print("  一个真正的积分器没有理由记住 40 步却记不住 10 步。\n")
    print("| arm | 训练点 acc | 短延迟(10) | 长延迟(80) | 短侧落差 | 长短比 | 判读 |")
    print("|---|---|---|---|---|---|---|")
    for key in sorted(verdict):
        v = verdict[key]
        arm, couple, tau = key
        a10 = v["means"][GAPS.index(10)]
        a40 = v["means"][GAPS.index(TRAIN_GAP)]
        a80 = v["means"][GAPS.index(80)]
        short_drop = a40 - a10
        ratio = a80 / a10 if a10 > 0 else float("nan")
        if a10 < CHANCE + 6 and a40 > 60:
            tag = "**背延迟**（短延迟也归零）"
        elif a40 < CHANCE + 6:
            tag = "无保持"
        elif a80 > CHANCE + 10 and a10 > a40 - 15:
            tag = "**真记忆**（两侧都保持）"
        else:
            tag = "弱保持"
        print(f"| {arm} | {a40:.1f} | {a10:.1f} | {a80:.1f} | {short_drop:+.1f} | "
              f"{ratio:.2f} | {tag} |")

    print("\n## 3  与同 gap 训练协议的对照\n")
    print("同 gap 训练（results_d2_wm）把每个延迟都单独优化过一次，")
    print("所以它的\"视界\"是**能力上界**；这里是**一次训练、复用到所有延迟**，")
    print("两者之差就是训练协议本身值多少。\n")
    same = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob("results_d2_wm/*.json")):
        d = json.load(open(f))
        if "error" in d:
            continue
        same[(d["arm"], float(d.get("tau_a", 0.0)))][d["gap"]].append(d["acc"])
    print("| arm | τ | gap | 同 gap 训练 | 跨 gap（训 40） | 差 |")
    print("|---|---|---|---|---|---|")
    for key in sorted(verdict):
        arm, couple, tau = key
        for g in GAPS:
            s = same.get((arm, tau), {}).get(g)
            c = verdict[key]["means"][GAPS.index(g)]
            if not s or c != c:
                continue
            sm = mean(s)
            print(f"| {arm} | {tau:g} | {g} | {sm:.1f} | {c:.1f} | {c-sm:+.1f} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_d2_xgap")

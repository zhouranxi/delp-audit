"""d2_long.py -- does the dual compartment ever pay off at long cue-to-input gaps?

The narrowed claim (decision 2) is that the slow input-driven state is what
carries the context, and the compartment is merely one way to build it. That
claim is only safe if the compartment does not win in the regime the original
hypothesis named: gap >> tau_s. The first sweep stopped at gap=80, which is
inside that regime's doorstep, so 160 and 320 were run as the evidence gap.

Reads the original sweep (d2_summary.csv, gaps 2..80) and the extension
(results_d2_long/*.json, gaps 160/320). Behaves sensibly when the extension is
still partially written.
"""
import csv
import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

ARMS = [
    ("point", "-", "point neuron"),
    ("point_gate", "-", "point + gated ctx"),
    ("point_slow", "thr", "point + slow state"),
    ("alif", "thr", "ALIF (slow threshold)"),
    ("dual", "add", "dual compartment (add)"),
    ("dual", "mul", "dual compartment (mul)"),
    ("dual", "thr", "dual compartment (thr)"),
]


def welch(a, b):
    """Two-sided Welch t-test; avoids a scipy dependency on the cluster."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 1.0
    t = (ma - mb) / se
    # normal approximation is adequate here (n>=3, df small but we only need a
    # coarse read; the paper will quote exact t-tests from scipy).
    return math.erfc(abs(t) / math.sqrt(2))


def load():
    acc = defaultdict(list)
    probe = defaultdict(list)
    age = defaultdict(list)

    if os.path.exists("d2_summary.csv"):
        for r in csv.DictReader(open("d2_summary.csv")):
            try:
                arm, cp = r["arm"], r["couple"]
                gap = int(float(r["gap"]))
                acc[(arm, cp, gap)].append(float(r["acc"]))
                probe[(arm, cp, gap)].append(float(r["probe"]))
            except (KeyError, ValueError):
                continue

    for f in glob.glob("results_d2_long/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "acc" not in d:
            continue
        arm, cp = d["arm"], d.get("cfg", {}).get("couple", "-")
        gap = int(d["gap"])
        acc[(arm, cp, gap)].append(float(d["acc"]))
        probe[(arm, cp, gap)].append(float(d.get("ctx_probe", float("nan"))))
        age[(arm, cp, gap)].append(d["seed"])
    return acc, probe, age


def main():
    acc, probe, age = load()
    gaps = sorted({k[2] for k in acc})
    L = []
    L.append("# D2: does the dual compartment win when gap >> tau_s?\n")
    L.append("Accuracy / context probe. Probe = same input, different context. "
             "`n` is seeds.\n")

    hdr = "| gap | " + " | ".join(name for _, _, name in ARMS) + " |"
    L.append(hdr)
    L.append("|---" * (len(ARMS) + 1) + "|")
    for gap in gaps:
        cells = []
        for arm, cp, _ in ARMS:
            v = acc.get((arm, cp, gap))
            p = probe.get((arm, cp, gap))
            if not v:
                cells.append("-")
                continue
            pv = st.mean([x for x in p if x == x]) if p and any(x == x for x in p) else float("nan")
            cells.append(f"{st.mean(v):.1f} / {pv:.0f} (n={len(v)})")
        L.append(f"| {gap} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## The decisive comparison: dual(add) vs point_slow\n")
    L.append("| gap | point_slow | dual(add) | dual - point_slow | p (Welch) |")
    L.append("|---|---|---|---|---|")
    verdict_rows = []
    for gap in gaps:
        ps = acc.get(("point_slow", "thr", gap))
        du = acc.get(("dual", "add", gap))
        if not ps or not du:
            continue
        d = st.mean(du) - st.mean(ps)
        p = welch(du, ps)
        L.append(f"| {gap} | {st.mean(ps):.2f} | {st.mean(du):.2f} | {d:+.2f} | {p:.3f} |")
        verdict_rows.append((gap, d, p))
    L.append("")

    L.append("## Does the compartment ever beat ALIF?\n")
    L.append("ALIF also carries a slow state, so a compartment claim needs to beat "
             "it too, not just the plain point neuron.\n")
    L.append("| gap | dual(add) | alif | dual - alif | p (Welch) |")
    L.append("|---|---|---|---|---|")
    for gap in gaps:
        du = acc.get(("dual", "add", gap))
        al = acc.get(("alif", "thr", gap))
        if not du or not al:
            continue
        L.append(f"| {gap} | {st.mean(du):.2f} | {st.mean(al):.2f} | "
                 f"{st.mean(du) - st.mean(al):+.2f} | {welch(du, al):.3f} |")
    L.append("")

    # The falsification condition written into the plan: dual must gain >= 5
    # points on point_slow at gap >= 160 AND simultaneously beat alif.
    L.append("## Verdict against the pre-registered falsification condition\n")
    cond = [r for r in verdict_rows if r[0] >= 160]
    if not cond:
        L.append("Extension data not present yet; nothing to judge.\n")
    else:
        fires = []
        for gap, d, p in cond:
            du = acc.get(("dual", "add", gap))
            al = acc.get(("alif", "thr", gap))
            beats_alif = bool(du and al and st.mean(du) > st.mean(al))
            if d >= 5.0 and beats_alif:
                fires.append(gap)
            L.append(f"- gap={gap}: dual - point_slow = {d:+.2f} (p={p:.3f}); "
                     f"beats alif: {beats_alif}")
        L.append("")
        if fires:
            L.append(f"**Condition fires at gap {fires}** -- the compartment does "
                     "earn its keep at long delay, so decision 2 must be reverted "
                     "to the original compartment claim (with the delay range "
                     "stated explicitly).\n")
        else:
            L.append("**Condition does not fire.** The compartment never recovers "
                     "a >=5 point advantage over a single membrane carrying the "
                     "same slow state, at any gap tested up to the longest. The "
                     "narrowed claim in decision 2 stands, and the delay range "
                     "tested must be stated in the paper.\n")

    # Decay shape: a compartment that only works at long delay would show a
    # shallower decay, so report the loss from the shortest to the longest gap.
    L.append("## Decay from gap 2 to the longest gap\n")
    longest = max(gaps)
    L.append("| arm | acc@gap2 | acc@%d | loss |" % longest)
    L.append("|---|---|---|---|")
    for arm, cp, name in ARMS:
        a0 = acc.get((arm, cp, 2))
        a1 = acc.get((arm, cp, longest))
        if a0 and a1:
            L.append(f"| {name} | {st.mean(a0):.1f} | {st.mean(a1):.1f} | "
                     f"{st.mean(a1) - st.mean(a0):+.1f} |")
    L.append("")

    txt = "\n".join(L)
    open("d2_long.md", "w").write(txt)
    print(txt)
    print("wrote d2_long.md")


if __name__ == "__main__":
    main()

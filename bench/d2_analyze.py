"""d2_analyze.py -- turn results_d2/*.json into the paper's main table and figure data.

Outputs
  gap_curve.md   accuracy / context-probe as a function of gap, per arm
  ktau.md        kappa x tau_a surface
  taulaw.md      Delta_1/2 (the gap at which accuracy falls to half its short-gap
                 value) vs the apical half-life t_half = ln2 / (-ln tau_a)
  d2_summary.csv one row per run
"""
import csv, glob, json, math, statistics as st, sys
from collections import defaultdict


def load(pattern="results_d2/*.json"):
    rows = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "acc" not in d:
            continue
        c = d.get("cfg", {})
        rows.append(dict(arm=d["arm"], couple=c.get("couple", "thr"), gap=d["gap"],
                         kappa=c.get("kappa"), tau_a=c.get("tau_a"),
                         tau_s=c.get("tau_s"), seed=d["seed"], acc=d["acc"],
                         probe=d["ctx_probe"], spikes=d["spikes"],
                         nparam=d["nparam"], T=d["T"],
                         file=f.split("/")[-1]))
    return rows


def cell(rows, keys=("arm", "couple", "gap", "kappa", "tau_a")):
    d = defaultdict(list)
    for r in rows:
        d[tuple(r[k] for k in keys)].append(r)
    out = []
    for k, v in d.items():
        out.append(dict(zip(keys, k), n=len(v),
                        acc=st.mean([x["acc"] for x in v]),
                        acc_sd=st.pstdev([x["acc"] for x in v]) if len(v) > 1 else 0.0,
                        probe=st.mean([x["probe"] for x in v]),
                        spikes=st.mean([x["spikes"] for x in v]),
                        T=v[0]["T"]))
    return sorted(out, key=lambda r: (r["arm"], r["couple"], r["tau_a"] or 0, r["gap"]))


def half_gap(curve, chance):
    """Linear interpolation of the gap at which accuracy falls halfway between the
    short-gap plateau and chance. Returns None if the curve never gets there."""
    pts = sorted(curve.items())
    if len(pts) < 3:
        return None
    plateau = st.mean([v for g, v in pts[:2]])          # two shortest gaps
    if plateau - chance < 5:
        return None
    target = chance + 0.5 * (plateau - chance)
    for (g0, a0), (g1, a1) in zip(pts, pts[1:]):
        if a0 >= target >= a1:
            if abs(a0 - a1) < 1e-9:
                return g1
            return g0 + (a0 - target) / (a0 - a1) * (g1 - g0)
    return None


def main(pattern="results_d2/*.json", out="."):
    rows = load(pattern)
    if not rows:
        print("no d2 results yet"); return
    cells = cell(rows)
    with open(f"{out}/d2_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- gap curves (main figure)
    L = ["# accuracy vs silent gap\n",
         "Chance = 1/9 = 11.1%. `ctx@gap-end` = linear decodability of the context",
         "from the slow state at the last silent step (chance 33.3%).\n"]
    byarm = defaultdict(dict)
    for c in cells:
        if c["kappa"] == 0.8 and c["tau_a"] == 0.97:
            byarm[(c["arm"], c["couple"])][c["gap"]] = c["acc"]
    L.append("| arm | couple | " + " | ".join(f"g{g}" for g in sorted(
        {g for v in byarm.values() for g in v})) + " |")
    L.append("|---" * (2 + len({g for v in byarm.values() for g in v})) + "|")
    gaps = sorted({g for v in byarm.values() for g in v})
    for (arm, cp), v in sorted(byarm.items()):
        L.append(f"| {arm} | {cp} | " + " | ".join(
            f"{v[g]:.1f}" if g in v else "-" for g in gaps) + " |")
    open(f"{out}/gap_curve.md", "w").write("\n".join(L) + "\n")

    # ---- probe table (mechanism)
    P = ["# context decodability from the slow state at the end of the silent gap\n",
         "| arm | couple | " + " | ".join(f"g{g}" for g in gaps) + " |",
         "|---" * (2 + len(gaps)) + "|"]
    byprobe = defaultdict(dict)
    for c in cells:
        if c["kappa"] == 0.8 and c["tau_a"] == 0.97:
            byprobe[(c["arm"], c["couple"])][c["gap"]] = c["probe"]
    for (arm, cp), v in sorted(byprobe.items()):
        P.append(f"| {arm} | {cp} | " + " | ".join(
            f"{v[g]:.1f}" if g in v else "-" for g in gaps) + " |")
    open(f"{out}/probe_table.md", "w").write("\n".join(P) + "\n")

    # ---- kappa x tau_a surface
    K = ["# kappa x tau_a (dual, thr coupling)\n"]
    for gap in sorted({c["gap"] for c in cells if c["arm"] == "dual"}):
        sub = [c for c in cells if c["arm"] == "dual" and c["gap"] == gap]
        taus = sorted({c["tau_a"] for c in sub})
        kaps = sorted({c["kappa"] for c in sub})
        if len(taus) < 2 or len(kaps) < 2:
            continue
        K.append(f"\n## gap = {gap}\n")
        K.append("| kappa \\ tau_a | " + " | ".join(f"{t:g}" for t in taus) + " |")
        K.append("|---" * (1 + len(taus)) + "|")
        for k in kaps:
            row = []
            for t in taus:
                m = [c for c in sub if c["kappa"] == k and c["tau_a"] == t]
                row.append(f"{m[0]['acc']:.1f}" if m else "-")
            K.append(f"| {k:g} | " + " | ".join(row) + " |")
    open(f"{out}/ktau.md", "w").write("\n".join(K) + "\n")

    # ---- the law: half-decay gap vs apical half-life
    T = ["# Delta_1/2 vs apical half-life\n",
         "Hypothesis: the gap at which accuracy halves scales with",
         "t_half = ln2 / (-ln tau_a). A linear relation in log-log or in t_half",
         "supports 'the apical membrane is what carries the context'.\n",
         "| tau_a | t_half (steps) | Delta_1/2 | plateau acc | chance-adj |",
         "|---|---|---|---|---|"]
    for tau_a in sorted({c["tau_a"] for c in cells if c["arm"] == "dual"}):
        sub = {c["gap"]: c["acc"] for c in cells
               if c["arm"] == "dual" and c["couple"] == "thr" and c["tau_a"] == tau_a
               and c["kappa"] == 0.8}
        if len(sub) < 3:
            continue
        hg = half_gap(sub, 11.1)
        th = math.log(2) / (-math.log(tau_a))
        plat = st.mean([v for g, v in sorted(sub.items())[:2]])
        T.append(f"| {tau_a:g} | {th:.1f} | "
                 f"{('%.1f' % hg) if hg else '> max gap'} | {plat:.1f} | "
                 f"{('%.1f' % (hg / th)) if hg else '-'} |")
    open(f"{out}/taulaw.md", "w").write("\n".join(T) + "\n")
    print(f"wrote gap_curve.md probe_table.md ktau.md taulaw.md d2_summary.csv "
          f"({len(rows)} runs, {len(cells)} cells)")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_d2/*.json",
         sys.argv[2] if len(sys.argv) > 2 else ".")

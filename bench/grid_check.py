"""grid_check.py -- does each family's lambda grid actually bracket its optimum?

A lambda sweep is only informative if the argmax sits strictly inside the grid.
If it lands on an edge, the "best" number reported for that family is just the
best of what happened to be tried, and any family comparison built on it is
comparing two differently-truncated searches. This is the exact defect found in
exp6 (lambda=5000 was the top of its grid) and it must be cleared before D1's
exchange-rate table can be called a mechanism finding.

Reads the D1 result json files. For every (arm, ntask) it reports the argmax on
the grid, whether that argmax is an edge point, and how much accuracy lies
between the argmax and the nearest interior neighbour -- i.e. how much is
plausibly being left on the table outside the grid.
"""
import glob
import json
import statistics as st
import sys
from collections import defaultdict


def load(patterns):
    # hgain is part of the cell key: an alpha-calibrated DeLP run is a different
    # method configuration, not another lambda sample of the same curve. Averaging
    # them together would produce a curve that belongs to no model at all.
    cells = defaultdict(list)
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if "error" in d or "lam" not in d or "final_avg" not in d:
                continue
            g = float(d.get("hgain") or 1.0)
            cells[(d["arm"], int(d["ntask"]), float(d["lam"]), g)].append(d)
    return cells


def main():
    pats = sys.argv[1:] or ["results/*.json"]
    cells = load(pats)
    keys = sorted({(k[0], k[3]) for k in cells})
    nts = sorted({k[1] for k in cells})

    def name(arm, g):
        return arm if g == 1.0 else f"{arm}@alpha={g:g}"

    L = []
    L.append("# lambda-grid adequacy for the D1 families\n")
    L.append("For each (family, T): the argmax of held-out accuracy, whether that "
             "argmax sits on a grid edge, and the accuracy gap to the adjacent "
             "interior point. `edge` means the reported optimum is a truncation "
             "artefact, not a measurement.\n")

    bad = []
    for nt in nts:
        L.append(f"## T = {nt}\n")
        L.append("| family | grid | argmax | edge? | acc@argmax | BWT@argmax | "
                 "gap to neighbour |")
        L.append("|---|---|---|---|---|---|---|")
        for arm, gg in keys:
            pts = sorted((l, st.mean([r["final_avg"] for r in v]),
                          st.mean([r["bwt"] for r in v]),
                          len(v), st.stdev([r["final_avg"] for r in v]) if len(v) > 1 else 0.0)
                         for (a, n, l, g), v in cells.items()
                         if a == arm and n == nt and g == gg)
            pts = [p for p in pts if p[3] >= 2]
            if not pts:
                continue
            nonzero = [p for p in pts if p[0] > 0]
            if not nonzero:
                L.append(f"| {name(arm, gg)} | {pts[0][0]:g} (single) | - | - | "
                         f"{pts[0][1]:.2f} | {pts[0][2]:+.2f} | - |")
                continue
            grid = [p[0] for p in nonzero]
            best = max(nonzero, key=lambda p: p[1])
            lo, hi = min(grid), max(grid)
            at_lo, at_hi = best[0] == lo, best[0] == hi
            edge = "**floor**" if at_lo else ("**ceiling**" if at_hi else "no")
            if at_lo or at_hi:
                i = nonzero.index(best)
                nb = nonzero[i + 1] if at_lo else nonzero[i - 1]
                nbTxt = f"{best[1] - nb[1]:+.2f} vs lam={nb[0]:g}"
                bad.append((name(arm, gg), nt, "floor" if at_lo else "ceiling",
                            best[0], best[1], nbTxt))
            else:
                nbTxt = "-"
            L.append(f"| {name(arm, gg)} | {lo:g}..{hi:g} | lam={best[0]:g} | {edge} | "
                     f"{best[1]:.2f} | {best[2]:+.2f} | {nbTxt} |")
        L.append("")

    L.append("## Verdict\n")
    if not bad:
        L.append("Every family's argmax is interior at every T. The grids bracket "
                 "their optima and the reported numbers are measurements, not "
                 "truncations.\n")
    else:
        L.append(f"**{len(bad)} of the family x T cells are edge-pinned**, so the "
                 "published numbers for those cells are truncation artefacts:\n")
        for arm, nt, side, lam, acc, nb in bad:
            L.append(f"- `{arm}` at T={nt}: argmax at the {side} (lambda={lam:g}, "
                     f"acc={acc:.2f}); {nb}")
        L.append("")
        L.append("These grids must be extended in the pinned direction before the "
                 "family comparison is reported. Note that the direction is the "
                 "same one the ladder ablation found for DeLP, so the D1 table "
                 "would otherwise inherit the same bias.\n")

    # Cross-family comparability at a common lambda is a separate question from
    # each family's own optimum, and reviewers ask it, so report it too.
    L.append("## Common-lambda comparability\n")
    L.append("The lambdas that all non-trivial families share, and how many "
             "families each covers:\n")
    shared = defaultdict(list)
    for (a, n, l, g), v in cells.items():
        if l > 0 and len(v) >= 2:
            shared[l].append(name(a, g))
    for l in sorted(shared):
        fams = sorted(set(shared[l]))
        if len(fams) >= 3:
            L.append(f"- lambda={l:g}: {len(fams)} families ({', '.join(fams)})")
    L.append("")

    txt = "\n".join(L)
    open("grid_check.md", "w").write(txt)
    print(txt)
    print("wrote grid_check.md")


if __name__ == "__main__":
    main()

"""analyze_alpha.py -- is DeLP's position on the exchange-rate axis a mechanism
property, or an artefact of an un-tuned hidden-layer scale?

DeLP's gate is h = kwta(t * sigmoid(d)). sigmoid(d) sits near 0.58 and never
opens (measured), so the dendritic arm runs with a hidden vector at ~0.58x the
scale of the same network without the gate. kwta is scale-equivariant, so this
one scalar is a free re-parameterisation of the same function class -- but EWC
penalises (theta - theta_ref)^2, which is scale-sensitive, so the same lambda
buys a different amount of stability depending on that scale.

Therefore two questions, and they must be answered separately:

  1. At a *matched* lambda, does raising the gain change accuracy? (paired by
     seed, same lambda -> isolates the direct effect)
  2. After each gain is allowed its own best lambda, where does that
     configuration sit on the exchange-rate axis? (this is the number C4 needs)

Exchange rate R = |d(acc)/d log10 A| / (d(BWT)/d log10 A), with A = lambda *
protected mass, i.e. the penalised distance actually applied. R answers "how
many points of accuracy does one point of forgetting buy", which is comparable
across families whose lambdas live on different scales.
"""
import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict


def load(pats):
    cells = defaultdict(list)
    for pat in pats:
        for f in glob.glob(pat):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if "error" in d or "final_avg" not in d:
                continue
            g = float(d.get("hgain") or 1.0)
            cells[(d["arm"], g, int(d["ntask"]), float(d["lam"]))].append(d)
    return cells


def usable(recs, min_n=3):
    return [r for r in recs if r.get("log", {}).get("A")
            and r["log"]["A"][-1] and r["log"]["A"][-1] > 0]


def slope(xs, ys):
    """least-squares slope + R^2; returns (nan, nan) if degenerate."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    syy = sum((y - my) ** 2 for y in ys)
    r2 = (sxy ** 2) / (sxx * syy) if syy > 0 else float("nan")
    return b, r2


def main():
    pats = sys.argv[1:] or ["results/*.json", "results_alpha/*.json"]
    cells = load(pats)
    gains = sorted({k[1] for k in cells})
    nts = sorted({k[2] for k in cells})

    L = []
    L.append("# alpha-calibrated DeLP: is the family position a scale artefact?\n")
    L.append("`alpha` is the hidden-layer gain (`--hgain`); alpha=1 is the published "
             "configuration. Each cell is 5 seeds unless stated.\n")

    # ---------------------------------------------------------------- per alpha
    L.append("## 1. Best lambda per gain, and where each gain lands on the axis\n")
    L.append("| T | alpha | best lambda | acc | BWT | log10 A | R | R^2 of slopes |")
    L.append("|---|---|---|---|---|---|---|---|")
    best_by = {}
    for nt in nts:
        for g in gains:
            pts = []
            for (arm, gg, n, lam), recs in cells.items():
                if arm != "delp" or gg != g or n != nt or lam <= 0:
                    continue
                recs = usable(recs)
                if len(recs) < 3:
                    continue
                A = st.mean([r["log"]["A"][-1] for r in recs])
                pts.append((lam, st.mean([r["final_avg"] for r in recs]),
                            st.mean([r["bwt"] for r in recs]), math.log10(A), len(recs)))
            if len(pts) < 3:
                continue
            pts.sort()
            best = max(pts, key=lambda p: p[1])
            best_by[(nt, g)] = best
            la = [p[3] for p in pts]
            sa, _ = slope(la, [p[1] for p in pts])
            sb, r2 = slope(la, [p[2] for p in pts])
            R = abs(sa / sb) if sb and sb == sb and sb != 0 else float("nan")
            L.append(f"| {nt} | {g:g} | {best[0]:g} | {best[1]:.2f} | "
                     f"{best[2]:+.2f} | {best[3]:.3f} | {R:.2f} | {r2:.3f} |")
    L.append("")

    # ------------------------------------------------- matched-lambda effect
    L.append("## 2. Direct effect of the gain, at matched lambda (paired by seed)\n")
    L.append("Same lambda, same seeds, only alpha differs. This isolates the "
             "re-parameterisation effect from any lambda re-placement.\n")
    L.append("| T | lambda | alpha | d(acc) vs alpha=1 | p | d(BWT) | d(log10 A) |")
    L.append("|---|---|---|---|---|---|---|")
    for nt in nts:
        for g in gains:
            if g == 1.0:
                continue
            for lam in sorted({k[3] for k in cells if k[2] == nt and k[1] == g}):
                base = usable(cells.get(("delp", 1.0, nt, lam), []))
                cur = usable(cells.get(("delp", g, nt, lam), []))
                if len(base) < 3 or len(cur) < 3:
                    continue
                bmap = {r["seed"]: r for r in base}
                pairs = [(bmap[r["seed"]], r) for r in cur if r["seed"] in bmap]
                if len(pairs) < 3:
                    continue
                da = [b["final_avg"] - a["final_avg"] for a, b in pairs]
                db = [b["bwt"] - a["bwt"] for a, b in pairs]
                dA = [math.log10(b["log"]["A"][-1]) - math.log10(a["log"]["A"][-1])
                      for a, b in pairs]
                s = st.stdev(da) if len(da) > 1 else 0.0
                t = st.mean(da) / (s / math.sqrt(len(da))) if s else 0.0
                p = math.erfc(abs(t) / math.sqrt(2))
                L.append(f"| {nt} | {lam:g} | {g:g} | {st.mean(da):+.2f} | {p:.4f} | "
                         f"{st.mean(db):+.2f} | {st.mean(dA):+.3f} |")
    L.append("")

    # ------------------------------------------------------------- verdict
    L.append("## 3. Verdict for contribution C4\n")
    if not any(g != 1.0 for g in gains):
        L.append("No alpha arm present yet.\n")
    else:
        L.append("The reading that decides C4: if a gain-calibrated DeLP still sits "
                 "among the expensive families, the family grouping is a mechanism "
                 "result and can be published. If it moves to the cheap cluster, "
                 "the grouping was tracking the hidden-layer scale and C4 must be "
                 "withdrawn.\n")
        for nt in nts:
            b1 = best_by.get((nt, 1.0))
            if not b1:
                continue
            moved = []
            for g in gains:
                if g == 1.0:
                    continue
                bg = best_by.get((nt, g))
                if bg:
                    moved.append(f"alpha={g:g}: acc {bg[1]:.2f} ({bg[1] - b1[1]:+.2f}), "
                                 f"BWT {bg[2]:+.2f} ({bg[2] - b1[2]:+.2f})")
            if moved:
                L.append(f"- T={nt}: alpha=1 best acc {b1[1]:.2f}. " + "; ".join(moved))
        L.append("")

    txt = "\n".join(L)
    open("alpha_probe.md", "w").write(txt)
    print(txt)
    print("wrote alpha_probe.md")


if __name__ == "__main__":
    main()

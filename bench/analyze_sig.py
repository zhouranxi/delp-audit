"""analyze_sig.py -- is the ladder's R0 -> R0b gap about saturation, or about
the hidden activation in general?

Two sweeps of the SAME instrument (identical data, permutations, seeds,
optimiser, epochs, batch, kWTA k and multi-anchor EWC penalty), differing only
in the hidden nonlinearity:

  gain   h = kwta(sigmoid(g*t)),  g = 0 meaning h = kwta(0.5*t)
         The ladder's hypothesis was that saturation is what hurts: saturating
         sigmoid piles units near 1.0 so the top-k becomes arbitrary. If so,
         accuracy must improve monotonically as g -> 0 and must move opposite
         to the measured saturation fraction.
  scale  h = kwta(alpha*t),  alpha = 0.5 reproducing R0b bit for bit.
         The rival hypothesis: any magnitude-destroying squash hurts, and what
         actually matters is the scale/shape of the hidden vector -- i.e. a
         hyperparameter, not a mechanism.

Both are reported against the ladder's own R0 / R0b / R4 numbers read from
ladder.csv, so the comparison never drifts from the main table.
"""
import csv, glob, json, os, statistics as st, sys
from collections import defaultdict

REFS = {"R0": "mlp_ewc", "R0b": "mlp_nosig", "R4": "ad_usage"}


def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = sum((v - mx) ** 2 for v in x) ** 0.5
    sy = sum((v - my) ** 2 for v in y) ** 0.5
    return sum((u - mx) * (v - my) for u, v in zip(x, y)) / (sx * sy + 1e-12)


def ladder_refs(path="ladder.csv"):
    """val-selected accuracy per (arm, T), same rule as analyze_ladder.py."""
    if not os.path.exists(path):
        return {}
    cells = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        try:
            cells[(r["arm"], int(r["ntask"]))][float(r["lam"])].append(r)
        except (KeyError, ValueError):
            continue
    out = {}
    for (arm, nt), by_lam in cells.items():
        va = {l: st.mean([float(x["val_final_avg"]) for x in rs
                          if x.get("val_final_avg")])
              for l, rs in by_lam.items() if any(x.get("val_final_avg") for x in rs)}
        va = {l: v for l, v in va.items() if v == v}
        if not va:
            continue
        best = max(va, key=va.get)
        rs = by_lam[best]
        out[(arm, nt)] = (best,
                          st.mean([float(x["final_avg"]) for x in rs]),
                          st.mean([float(x["bwt"]) for x in rs]))
    return out


def main():
    pats = sys.argv[1:] or ["results_sig/*.json", "results_sc/*.json"]
    gains = defaultdict(list)      # (T, g, lam) -> recs
    scales = defaultdict(list)     # (T, alpha, lam) -> recs
    for pt in pats:
        for f in sorted(glob.glob(pt)):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            for _, recs in d.get("runs", {}).items():
                for r in recs:
                    if "bwt" not in r or r.get("error"):
                        continue
                    if r.get("alpha") is not None:
                        scales[(r["ntask"], float(r["alpha"]),
                                float(r["lam"]))].append(r)
                    elif r.get("gain") is not None:
                        gains[(r["ntask"], float(r["gain"]),
                               float(r["lam"]))].append(r)
    refs = ladder_refs()

    L = ["# Hidden activation: what actually moves the number?\n",
         "One instrument, one change: the hidden nonlinearity. Every cell uses "
         "the same MNIST permutations, seeds, optimiser, epochs, batch size, "
         "kWTA k = 64 and multi-anchor EWC penalty as the ladder.\n"]

    tasks = sorted({k[0] for k in gains} | {k[0] for k in scales})
    for nt in tasks:
        L.append(f"## T = {nt}\n")
        if refs:
            L.append("Ladder reference at each rung's validation-selected lambda:\n")
            L.append("| rung | lambda* | final acc | BWT |")
            L.append("|---|---|---|---|")
            for tag, arm in REFS.items():
                if (arm, nt) in refs:
                    lam, a, b = refs[(arm, nt)]
                    L.append(f"| {tag} ({arm}) | {lam:g} | {a:.2f} | {b:+.2f} |")
            L.append("")

        # ---- gain sweep ----
        rows = sorted(k for k in gains if k[0] == nt)
        if rows:
            L.append("### a. sigmoid gain  h = kwta(sigmoid(g*t))\n")
            L.append("| gain g | lambda | final acc | BWT | sat (>0.99) | n |")
            L.append("|---|---|---|---|---|---|")
            pts = []
            for (_, g, lam) in rows:
                rs = gains[(nt, g, lam)]
                a = st.mean([r["final_avg"] for r in rs])
                b = st.mean([r["bwt"] for r in rs])
                s = st.mean([r.get("sat_mean", 0.0) for r in rs])
                L.append(f"| {g:g} | {lam:g} | {a:.2f} | {b:+.2f} | "
                         + ("n/a (no squash)" if g == 0 else f"{s:.3f}")
                         + f" | {len(rs)} |")
                if g > 0:
                    pts.append((g, s, a))
            L.append("")
            if len(pts) > 2:
                gs = [p[0] for p in pts]
                ss = [p[1] for p in pts]
                aa = [p[2] for p in pts]
                L.append(f"- corr(gain, saturation) = {corr(gs, ss):+.3f}")
                L.append(f"- corr(saturation, accuracy) = {corr(ss, aa):+.3f}")
                L.append(f"- accuracy spans {min(aa):.2f} … {max(aa):.2f} "
                         f"over the g > 0 grid\n")

        # ---- scale sweep ----
        rows = sorted(k for k in scales if k[0] == nt)
        if rows:
            L.append("### b. hidden scale  h = kwta(alpha*t)\n")
            L.append("| alpha | lambda | final acc | BWT | n |")
            L.append("|---|---|---|---|---|")
            best = {}
            for (_, al, lam) in rows:
                rs = scales[(nt, al, lam)]
                a = st.mean([r["final_avg"] for r in rs])
                b = st.mean([r["bwt"] for r in rs])
                L.append(f"| {al:g} | {lam:g} | {a:.2f} | {b:+.2f} | {len(rs)} |")
                if al not in best or a > best[al][0]:
                    best[al] = (a, lam, b)
            L.append("")
            if best:
                alat = max(best, key=lambda k: best[k][0])
                L.append(f"- best over alpha and lambda: alpha = {alat:g} at "
                         f"lambda = {best[alat][1]:g} -> acc {best[alat][0]:.2f}, "
                         f"BWT {best[alat][2]:+.2f}")
                if ("ad_usage", nt) in refs:
                    r4 = refs[("ad_usage", nt)][1]
                    L.append(f"- that is {best[alat][0] - r4:+.2f} against R4 "
                             f"(ours) at its own lambda* = {r4:.2f}")
                L.append("")

    open("sig_probe.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

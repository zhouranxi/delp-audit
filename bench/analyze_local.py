"""analyze_local.py -- exchange rate measured as a chord, not a derivative.

Two earlier definitions both failed, and the second failure is instructive:

  * whole-grid line: mixes the ascending limb with the descending one;
  * derivative at lambda*: degenerate by construction. lambda* IS the accuracy
    maximum, so d(acc)/dlog(lambda) == 0 there and the ratio is 0 divided by a
    non-zero number. That is why the values came out tiny and the fits had
    R^2 around 0.4 -- not noise, but a singularity we chose to sit on.

The well-posed quantity is a chord over a fixed multiplicative window:

    R = |acc(c*lambda*) - acc(lambda*)| / |BWT(lambda*) - BWT(c*lambda*)|

with c = 4, i.e. "if you push the stability penalty four times harder than the
setting that maximises accuracy, what do you pay and what do you buy?" Both
endpoints are measured, non-degenerate, and the window is fixed in log units so
it is comparable across families whose lambdas differ by orders of magnitude.

The frontier is still the primary object; this chord is a single number extracted
from it under a stated convention, and the convention is printed with the table.
"""
import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict

C = 4.0  # multiplicative window

STAR = {
    "delp":       {5: 1000.0, 10: 100.0, 20: 30.0},
    "ewc_online": {5: 10.0,   10: 100.0, 20: 100.0},
    "ewc_multi":  {5: 1000.0, 10: 10.0,  20: 10.0},
    "si":         {5: 10.0,   10: 10.0,  20: 1.0},
    "mas":        {5: 1.0,    10: 0.3,   20: 0.1},
}


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
            A = d.get("log", {}).get("A") or []
            if not A or not A[-1] or A[-1] <= 0:
                continue
            cells[(d["arm"], int(d["ntask"]), float(d["lam"]))].append(
                dict(seed=d["seed"], acc=d["final_avg"], bwt=d["bwt"],
                     logA=math.log10(A[-1])))
    return cells


def pick(cells, arm, nt, lam, tol=1e-6):
    """Seeds at the lambda closest to `lam` (relative tolerance)."""
    best, bestd = None, None
    for (a, n, l), recs in cells.items():
        if a != arm or n != nt or len(recs) < 3:
            continue
        d = abs(math.log(l / lam))
        if bestd is None or d < bestd:
            best, bestd = (l, recs), d
    if best is None or bestd > tol + 0.35:
        return None
    return best


def chord(cells, arm, nt, lam0, factor=C):
    lo = pick(cells, arm, nt, lam0)
    hi = pick(cells, arm, nt, lam0 * factor)
    if not lo or not hi or lo[0] == hi[0]:
        return None
    a0 = st.mean([r["acc"] for r in lo[1]])
    b0 = st.mean([r["bwt"] for r in lo[1]])
    a1 = st.mean([r["acc"] for r in hi[1]])
    b1 = st.mean([r["bwt"] for r in hi[1]])
    dA = st.mean([r["logA"] for r in hi[1]]) - st.mean([r["logA"] for r in lo[1]])
    dacc, dbwt = a1 - a0, b1 - b0
    if dbwt == 0:
        return None
    # BWT rises toward 0 (less forgetting) as lambda grows; the price paid is
    # the accuracy drop. Both are signed so the ratio is positive when the trade
    # behaves normally.
    R = abs(dacc / dbwt)
    return dict(lam0=lo[0], lam1=hi[0], acc0=a0, acc1=a1, bwt0=b0, bwt1=b1,
                dacc=dacc, dbwt=dbwt, dlogA=dA, R=R)


def detect_star(cells, min_n=3):
    """Re-detect the accuracy-optimal lambda from every lambda available now.

    The STAR table below was hardcoded from an earlier, coarser grid. Once extra
    lambdas arrive (results_fix, results_fine) the argmax can move, and using a
    stale lambda* would measure the chord from a point that is not the optimum --
    exactly the truncation error this whole analysis is about.
    """
    star = defaultdict(dict)
    for (a, n, lam), recs in cells.items():
        if lam <= 0 or len(recs) < min_n:
            continue
        star[(a, n)][lam] = st.mean([r["acc"] for r in recs])
    out = defaultdict(dict)
    for (a, n), d in star.items():
        if len(d) < 3:
            continue
        out[a][n] = max(d.items(), key=lambda kv: kv[1])[0]
    return out


def main():
    pats = sys.argv[1:] or ["results/*.json", "results_fix/*.json",
                            "results_local/*.json", "results_fine/*.json"]
    cells = load(pats)
    detected = detect_star(cells)

    L = []
    L.append("# Exchange rate as a chord over lambda* -> 4 x lambda*\n")
    L.append("Both endpoints are measured. The derivative at lambda* is NOT used: "
             "lambda* is the accuracy maximum, so that derivative is zero by "
             "construction and the ratio is degenerate (the earlier attempt "
             "produced R^2 ~ 0.4 and sign flips, which is the signature of "
             "sitting on the singularity, not of noise).\n")

    L.append("| family | T | lambda* | lambda* x4 | acc: lo -> hi | "
             "BWT: lo -> hi | d(acc) | d(BWT) | R |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    summary = defaultdict(dict)
    for arm in sorted(detected):
        for nt in sorted(detected[arm]):
            lam0 = detected[arm][nt]
            r = chord(cells, arm, nt, lam0)
            if not r:
                continue
            summary[arm][nt] = r
            L.append(f"| {arm} | {nt} | {r['lam0']:g} | {r['lam1']:g} | "
                     f"{r['acc0']:.2f} -> {r['acc1']:.2f} | "
                     f"{r['bwt0']:+.2f} -> {r['bwt1']:+.2f} | "
                     f"{r['dacc']:+.2f} | {r['dbwt']:+.2f} | {r['R']:.2f} |")
    L.append("")

    L.append("## Is R stable across T?\n")
    L.append("| family | R (T=5 / 10 / 20) | spread | verdict |")
    L.append("|---|---|---|---|")
    for arm in sorted(summary):
        row = {n: v["R"] for n, v in summary[arm].items()}
        vals = [row[n] for n in sorted(row)]
        if not vals or min(vals) <= 0:
            continue
        spread = max(vals) / min(vals)
        L.append(f"| {arm} | " + " / ".join(
            f"{row[n]:.2f}" if n in row else "-" for n in (5, 10, 20)) +
            f" | {spread:.1f}x | {'stable' if spread < 1.5 else 'unstable'} |")
    L.append("")

    L.append("## Ordering at each T\n")
    for nt in (5, 10, 20):
        order = sorted(((v[nt]["R"], k) for k, v in summary.items() if nt in v))
        if not order:
            continue
        L.append(f"**T = {nt}** (cheapest first): " +
                 ", ".join(f"{k} {r:.2f}" for r, k in order) + "\n")

    txt = "\n".join(L)
    open("d1_local.md", "w").write(txt)
    print(txt)
    print("wrote d1_local.md")


if __name__ == "__main__":
    main()

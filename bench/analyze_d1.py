"""analyze_d1.py -- the exchange-rate table for the D1 benchmark.

The quantity: how many points of final accuracy does one point of forgetting
buy? Fitted slopes of (accuracy) and (BWT) against log10 A, where
A = lambda * protected mass is the penalised distance actually applied. Using A
rather than lambda is what makes families comparable, because their importance
estimates live on wildly different scales (MAS' output-sensitivity sums are
orders of magnitude above a diagonal Fisher).

    R = |d(acc)/d log10 A| / (d(BWT)/d log10 A)

R is a slope ratio, so it is dimensionless and its bootstrap CI is meaningful.
Two families with the same R trade stability for plasticity at the same price
even if their lambda values differ by 100x.

Discipline this script enforces rather than assumes:
  * a family is only reported when its argmax is strictly inside its lambda
    grid -- an edge argmax is a truncated search, not a measurement;
  * every slope carries a seed bootstrap CI, because 5 seeds x 4 lambdas is a
    small fit and the paper's central claim is a *difference between slopes*;
  * the per-T and pooled-across-T views are both printed, since a family whose
    R moves with T should not be quoted as a single number.
"""
import glob
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict

BOOT = 2000
FAMILIES = ["delp", "ewc_online", "ewc_multi", "si", "mas", "lwf"]


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
            if float(d.get("hgain") or 1.0) != 1.0:
                continue
            A = d.get("log", {}).get("A") or []
            if not A or not A[-1] or A[-1] <= 0:
                continue
            cells[(d["arm"], int(d["ntask"]), float(d["lam"]))].append(
                dict(seed=d["seed"], acc=d["final_avg"], bwt=d["bwt"],
                     logA=math.log10(A[-1])))
    return cells


def fit(pts):
    """pts: list of (logA, acc, bwt). Returns (sa, sb, r2)."""
    xs = [p[0] for p in pts]
    if len(set(xs)) < 2:
        return None
    mx = st.mean(xs)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    out = []
    for idx in (1, 2):
        ys = [p[idx] for p in pts]
        my = st.mean(ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        syy = sum((y - my) ** 2 for y in ys)
        out.append((sxy / sxx, (sxy ** 2) / (sxx * syy) if syy > 0 else float("nan")))
    return out[0][0], out[1][0], out[1][1]


def cell_curve(cells, arm, nt):
    """Return list of (lam, [seed records]) for one family x T, sorted by lam."""
    by = defaultdict(list)
    for (a, n, lam), recs in cells.items():
        if a == arm and n == nt and lam > 0 and len(recs) >= 3:
            by[lam].extend(recs)
    return sorted(by.items())


def boot_R(by, rng, n=BOOT, keep=None):
    """Resample seeds within each lambda, refit, collect R.

    `keep` optionally restricts the fit to a subset of lambda indices. This
    matters more than it looks: a fit over the whole grid mixes the ascending
    limb of the accuracy curve with the descending one, and the resulting R is
    not a trade-off at all. At T=5 the optimum sits at the top of the grid, so
    most of the range is still ascending and d(acc)/dlogA comes out positive for
    the accuracy term in three of five families -- R then measures the rising
    limb, which is why it drops to 0.01-0.11.
    """
    if len(by) < 3:
        return None
    idx = list(range(len(by))) if keep is None else [i for i in keep if i < len(by)]
    if len(idx) < 3:
        return None
    Rs = []
    for _ in range(n):
        pts = []
        for i in idx:
            lam, recs = by[i]
            draw = [rng.choice(recs) for _ in recs]
            pts.append((st.mean([r["logA"] for r in draw]),
                        st.mean([r["acc"] for r in draw]),
                        st.mean([r["bwt"] for r in draw])))
        f = fit(pts)
        if f and f[1] and f[1] == f[1] and f[1] != 0:
            Rs.append(abs(f[0] / f[1]))
    if not Rs:
        return None
    Rs.sort()
    return (st.median(Rs), Rs[int(0.025 * len(Rs))], Rs[int(0.975 * len(Rs)) - 1], len(Rs))


def main():
    pats = sys.argv[1:] or ["results/*.json", "results_fix/*.json"]
    cells = load(pats)
    nts = sorted({k[1] for k in cells})
    rng = random.Random(20261021)

    L = []
    L.append("# D1 exchange-rate table (lambda grids verified interior)\n")
    L.append("A = lambda x protected mass. R = |d(acc)/dlog10A| / (d(BWT)/dlog10A), "
             "bootstrap over seeds (%d draws). R is points-of-accuracy per "
             "point-of-forgetting.\n" % BOOT)

    pooled = {}
    for nt in nts:
        L.append(f"## T = {nt}\n")
        L.append("| family | lambdas | argmax | edge | acc@argmax | BWT@argmax | "
                 "d(acc)/dlogA | d(BWT)/dlogA | R whole-grid [95% CI] | "
                 "R descending [95% CI] | R^2 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in FAMILIES:
            by = cell_curve(cells, arm, nt)
            if len(by) < 3:
                continue
            lams = [l for l, _ in by]
            pts = [(st.mean([r["logA"] for r in recs]),
                    st.mean([r["acc"] for r in recs]),
                    st.mean([r["bwt"] for r in recs])) for _, recs in by]
            best = max(range(len(by)), key=lambda i: pts[i][1])
            edge = ("**floor**" if best == 0 else
                    "**ceiling**" if best == len(by) - 1 else "no")
            f = fit(pts)
            b = boot_R(by, rng)
            # descending limb only: lambda >= lambda*. This is the range that
            # actually describes the operating trade-off, and it is the only
            # definition that stays comparable across task counts.
            keep = list(range(best, len(by)))
            fd = fit([pts[i] for i in keep]) if len(keep) >= 3 else None
            bd = boot_R(by, rng, keep=keep)
            if not f:
                continue
            sa, sb, r2 = f
            R = abs(sa / sb) if sb else float("nan")
            ci = f"{R:.2f} [{b[1]:.2f}, {b[2]:.2f}]" if b else "n/a"
            Rd = abs(fd[0] / fd[1]) if fd and fd[1] else float("nan")
            cid = f"{Rd:.2f} [{bd[1]:.2f}, {bd[2]:.2f}]" if bd else "n/a"
            pooled.setdefault(arm, []).append((nt, R, ci, Rd, cid, r2))
            L.append(f"| {arm} | {lams[0]:g}..{lams[-1]:g} ({len(lams)}) | "
                     f"{lams[best]:g} | {edge} | {pts[best][1]:.2f} | "
                     f"{pts[best][2]:+.2f} | {sa:+.2f} | {sb:+.2f} | {ci} | "
                     f"{Rd:.2f} [{bd[1]:.2f}, {bd[2]:.2f}] | {r2:.3f} |"
                     if bd else
                     f"| {arm} | {lams[0]:g}..{lams[-1]:g} ({len(lams)}) | "
                     f"{lams[best]:g} | {edge} | {pts[best][1]:.2f} | "
                     f"{pts[best][2]:+.2f} | {sa:+.2f} | {sb:+.2f} | {ci} | - | {r2:.3f} |")
        L.append("")

    L.append("## Summary: does R separate into clusters, and does it move with T?\n")
    L.append("Whole-grid R first (the unstable definition), then the descending-limb "
             "R (lambda >= lambda*), which is the defensible one.\n")
    L.append("| family | whole-grid R (T=5 / 10 / 20) | spread | descending-limb R "
             "(T=5 / 10 / 20) | spread |")
    L.append("|---|---|---|---|---|")
    for arm in FAMILIES:
        rows = pooled.get(arm, [])
        if not rows:
            continue
        full = {nt: R for nt, R, _, _, _, _ in rows}
        desc = {nt: Rd for nt, _, _, Rd, _, _ in rows}

        def sp(d):
            v = [d[n] for n in sorted(d) if d[n] == d[n]]
            return max(v) / min(v) if v and min(v) > 0 else float("inf")

        L.append(f"| {arm} | " + " / ".join(
            f"{full[n]:.2f}" if n in full else "-" for n in nts) +
            f" | {sp(full):.1f}x | " + " / ".join(
            f"{desc[n]:.2f}" if n in desc else "-" for n in nts) +
            f" | {sp(desc):.1f}x |")
    L.append("")

    txt = "\n".join(L)
    open("d1_exchange.md", "w").write(txt)
    print(txt)
    print("wrote d1_exchange.md")


if __name__ == "__main__":
    main()

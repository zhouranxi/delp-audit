"""bench_analyze.py -- turn results/*.json into the tables and figures' data.

Outputs
-------
  summary.csv        one row per run
  by_cell.md         mean+-sd per (arm, ntask, lam)
  frontier.md        the (BWT, acc) frontier per arm x ntask
  collapse.md        does one protection-mass coordinate explain the frontier?
  power.md           how many seeds / tasks before rankings stabilise?
"""
import glob, json, math, statistics as st, sys, os
from collections import defaultdict

import csv


def load(pattern="results/*.json"):
    rows = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "bwt" not in d:
            continue
        lg = d.get("log", {})
        def avg(xs):
            xs = [x for x in xs if x is not None]
            return st.mean(xs) if xs else float("nan")
        rows.append(dict(arm=d["arm"], ntask=d["ntask"], lam=d["lam"], seed=d["seed"],
                         prune=d.get("prune", 0), bwt=d["bwt"], acc=d["final_avg"],
                         ell=d["ell"], resid=d.get("identity_resid", 0),
                         A=avg(lg.get("A", [])), O=avg(lg.get("O", [])),
                         P=avg(lg.get("P", [])),
                         rho=avg(lg.get("rho", [])),
                         mass=avg(lg.get("mass", [])),
                         freed=(min([f for f in lg.get("freed", []) if f is not None])
                                if any(f is not None for f in lg.get("freed", [])) else None),
                         elapsed=d.get("elapsed", 0), nparam=d.get("nparam", 0),
                         file=os.path.basename(f)))
    return rows


def cell(rows):
    d = defaultdict(list)
    for r in rows:
        d[(r["arm"], r["ntask"], r["lam"], r["prune"])].append(r)
    out = []
    for k, v in sorted(d.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        arm, nt, lam, pr = k
        out.append(dict(arm=arm, ntask=nt, lam=lam, prune=pr, n=len(v),
                        bwt=st.mean([x["bwt"] for x in v]),
                        bwt_sd=st.pstdev([x["bwt"] for x in v]) if len(v) > 1 else 0,
                        acc=st.mean([x["acc"] for x in v]),
                        acc_sd=st.pstdev([x["acc"] for x in v]) if len(v) > 1 else 0,
                        ell=st.mean([x["ell"] for x in v]),
                        A=st.mean([x["A"] for x in v]),
                        P=st.mean([x["P"] for x in v]),
                        O=st.mean([x["O"] for x in v]),
                        rho=st.mean([x["rho"] for x in v]),
                        freed=v[0]["freed"], secs=st.mean([x["elapsed"] for x in v])))
    return out


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy + 1e-12)


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def spearman(a, b):
    return corr(rank(a), rank(b))


def main(pattern="results/*.json"):
    rows = load(pattern)
    if not rows:
        print("no results"); return
    cells = cell(rows)
    with open("summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    L = []
    L.append("# by-cell summary (mean over seeds)\n")
    L.append(f"runs={len(rows)}  cells={len(cells)}\n")
    L.append("| arm | T | lam | n | BWT | acc | ell | P | A | O | rho | freed | s/run |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        L.append(f"| {c['arm']} | {c['ntask']} | {c['lam']:.4g} | {c['n']} | "
                 f"{c['bwt']:.2f}±{c['bwt_sd']:.2f} | {c['acc']:.2f}±{c['acc_sd']:.2f} | "
                 f"{c['ell']:.1f} | {c['P']:.3g} | {c['A']:.3g} | "
                 f"{c['O'] if c['O']==c['O'] else 0:.3f} | "
                 f"{c['rho']:.3f} | {c['freed']} | {c['secs']:.0f} |")
    open("by_cell.md", "w").write("\n".join(L) + "\n")

    # ---- frontier: per (arm, T) the lambda curve
    F = ["# stability-plasticity frontier (each row = one lambda)\n"]
    for nt in sorted(set(r["ntask"] for r in cells)):
        F.append(f"\n## T = {nt}\n")
        F.append("| arm | lam | BWT | acc | ell | P | d.ell/d.BWT |")
        F.append("|---|---|---|---|---|---|---|")
        for arm in sorted(set(c["arm"] for c in cells if c["ntask"] == nt)):
            cs = sorted([c for c in cells if c["ntask"] == nt and c["arm"] == arm],
                        key=lambda c: c["lam"])
            prev = None
            for c in cs:
                slope = ""
                if prev is not None and abs(c["bwt"] - prev["bwt"]) > 1e-6:
                    slope = f"{(c['ell'] - prev['ell']) / (c['bwt'] - prev['bwt']):+.2f}"
                F.append(f"| {arm} | {c['lam']:.4g} | {c['bwt']:.2f} | {c['acc']:.2f} | "
                         f"{c['ell']:.1f} | {c['P']:.3g} | {slope} |")
                prev = c
    open("frontier.md", "w").write("\n".join(F) + "\n")

    # ---- C1': universal shape + family offset
    # Fit  BWT = a_f + b_f * log10(A)  and  ell = c_f + d_f * log10(A)  per family.
    # "Universal shape" <=> the slopes b_f agree across families; the family only
    # shifts the curve horizontally, i.e. differs in intercept.
    def lin(xs, ys):
        n = len(xs)
        if n < 3:
            return None
        mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx < 1e-12:
            return None
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        a = my - b * mx
        ss = sum((y - my) ** 2 for y in ys)
        res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        return dict(a=a, b=b, r2=(1 - res / ss) if ss > 1e-12 else float("nan"), n=n)

    C = ["# C1\': universal shape + family offset\n",
         "For each family f and task count T we fit  BWT = a_f + b_f*log10(A)  and",
         "ell = c_f + d_f*log10(A).  If the slopes agree across families, the frontier",
         "has a family-independent SHAPE and families differ only by an offset",
         "(a horizontal shift in log A = the unit mismatch of their Omega).\n"]
    fits = {}
    for nt in sorted(set(c["ntask"] for c in cells)):
        sub = [c for c in cells if c["ntask"] == nt and c["A"] > 0]
        fams = sorted({c["arm"] for c in sub})
        if len(sub) < 4 or len(fams) < 2:
            continue
        C.append(f"\n## T = {nt}\n")
        C.append("| family | n | BWT slope | BWT R^2 | BWT a | ell slope | ell R^2 | ell a |")
        C.append("|---|---|---|---|---|---|---|---|")
        for f in fams:
            cc = [c for c in sub if c["arm"] == f]
            x = [math.log10(max(c["A"], 1e-12)) for c in cc]
            r1 = lin(x, [c["bwt"] for c in cc])
            r2 = lin(x, [c["ell"] for c in cc])
            if not r1 or not r2:
                continue
            fits[(nt, f)] = (r1, r2)
            C.append(f"| {f} | {r1['n']} | {r1['b']:+.2f} | {r1['r2']:.3f} | {r1['a']:+.1f} "
                     f"| {r2['b']:+.2f} | {r2['r2']:.3f} | {r2['a']:+.1f} |")
        slopes = [v[0]["b"] for k, v in fits.items() if k[0] == nt]
        if len(slopes) > 1:
            C.append(f"\n  BWT slope spread across families: "
                     f"{min(slopes):+.2f} .. {max(slopes):+.2f} "
                     f"(sd {st.pstdev(slopes):.2f}) -> "
                     f"{'shape looks family-independent' if st.pstdev(slopes) < 0.5 else 'shape differs, C1-prime fails'}")
    open("collapse.md", "w").write("\n".join(C) + "\n")

    # ---- rank stability across T and across seeds
    P = ["# power / rank stability\n"]
    Tset = sorted(set(r["ntask"] for r in cells))
    for a, b in zip(Tset, Tset[1:]):
        for lam in (1000.0,):
            ka = {c["arm"]: c["bwt"] for c in cells if c["ntask"] == a and c["lam"] == lam}
            kb = {c["arm"]: c["bwt"] for c in cells if c["ntask"] == b and c["lam"] == lam}
            common = [k for k in ka if k in kb]
            if len(common) >= 3:
                P.append(f"- Spearman(BWT ranking T={a}, T={b}) at lam={lam:.0f} over "
                         f"{len(common)} arms: "
                         f"{spearman([ka[k] for k in common], [kb[k] for k in common]):+.2f}")
    # seed subsampling: how stable is a cell mean at n seeds?
    P.append("\n## mean |error| of a cell BWT estimate vs number of seeds\n")
    P.append("| arm | T | lam | n seeds | sd of cell mean |")
    P.append("|---|---|---|---|---|")
    d = defaultdict(list)
    for r in rows:
        d[(r["arm"], r["ntask"], r["lam"])].append(r["bwt"])
    for k, v in sorted(d.items()):
        if len(v) >= 3:
            P.append(f"| {k[0]} | {k[1]} | {k[2]:.4g} | {len(v)} | "
                     f"{st.pstdev(v)/math.sqrt(len(v)):.2f} |")
    open("power.md", "w").write("\n".join(P) + "\n")
    print("wrote by_cell.md frontier.md collapse.md power.md summary.csv")
    print(f"runs={len(rows)} cells={len(cells)}")
    bad = [r for r in rows if r["resid"] and r["resid"] > 1e-6]
    print(f"identity check: {len(bad)}/{len(rows)} runs violate a_bar = ell + (T-1)/T*BWT")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/*.json")

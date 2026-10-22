"""analyze_ei.py -- does fixing the audited defects change the published ranking?

Reads the sharded sweep/main outputs of exp6_fixed.py and prints:

  S1  lambda sensitivity of each variant. The diagnostic of interest: a variant
      whose penalty is effectively disabled shows ~zero response to lambda.
  S2  same-lambda head-to-head (old vs fixed). This isolates the implementation
      defect from the lambda-selection rule.
  S3  main table at each variant's own best lambda, with the significance tests
      exp12 reported.
"""
import glob, json, math, statistics as st, sys
from collections import defaultdict


def load(pattern):
    sweep, main = defaultdict(list), defaultdict(list)
    for f in glob.glob(pattern):
        d = json.load(open(f))
        for k, v in d.get("sweep", {}).items():
            sweep[k].extend(v)
        for k, v in d.get("main", {}).items():
            main[k].extend(v)
    return sweep, main


def agg(rows):
    ok = [r for r in rows if "bwt" in r]
    if not ok:
        return None
    b = [r["bwt"] for r in ok]
    a = [r["final_avg"] for r in ok]
    return dict(n=len(ok), bwt=st.mean(b), bwt_sd=st.pstdev(b),
                acc=st.mean(a), acc_sd=st.pstdev(a))


def slope(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-12:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def main():
    pats = sys.argv[1:] or ["results_ei/*.json"]
    sweep, mainres = defaultdict(list), defaultdict(list)
    for pt in pats:
        sw, mr = load(pt)
        for k, v in sw.items():
            sweep[k].extend(v)
        for k, v in mr.items():
            mainres[k].extend(v)
    L = []

    # ---------------- S1: lambda sensitivity ----------------
    L.append("# S1  lambda sensitivity  (T=5, seeds 0-1)\n")
    L.append("d(BWT)/d(log10 lambda) is the elasticity. A variant whose penalty is")
    L.append("effectively off has an elasticity near zero no matter how wide the grid.\n")
    L.append("| arm | lambda grid | BWT at min lam | BWT at max lam | range | elasticity |")
    L.append("|---|---|---|---|---|---|")
    best = {}
    by_arm = defaultdict(dict)
    for key, rows in sweep.items():
        arm, lam = key.split("@")
        a = agg(rows)
        if a:
            by_arm[arm][float(lam)] = a
    for arm in sorted(by_arm):
        cells = by_arm[arm]
        lams = sorted(cells)
        if len(lams) < 2:
            continue
        xs = [math.log10(l) for l in lams]
        ys = [cells[l]["bwt"] for l in lams]
        el = slope(xs, ys)
        rng = max(ys) - min(ys)
        L.append(f"| {arm} | {len(lams)} pts | {ys[0]:+.2f} | {ys[-1]:+.2f} | "
                 f"{rng:.2f} | {el:+.2f} |")
        # best lambda: highest BWT, tie on accuracy (exp6's rule)
        bl = max(lams, key=lambda l: (cells[l]["bwt"], cells[l]["acc"]))
        best[arm] = (bl, cells[bl])
    L.append("")
    L.append("## argmax-lambda (test-set rule, same as exp6/exp12)\n")
    L.append("| arm | best lambda | BWT | acc |")
    L.append("|---|---|---|---|")
    for arm, (bl, c) in sorted(best.items()):
        L.append(f"| {arm} | {bl:g} | {c['bwt']:+.2f} ± {c['bwt_sd']:.2f} | "
                 f"{c['acc']:.2f} |")
    L.append("")

    # ---------------- S2: same-lambda head-to-head ----------------
    L.append("# S2  old vs fixed at the SAME lambda  (isolates the defect)\n")
    L.append("| family | lambda | BWT old | BWT fixed | delta | acc old | acc fixed |")
    L.append("|---|---|---|---|---|---|---|")
    for fam in ("si", "mas"):
        old, new = by_arm.get(f"{fam}_old", {}), by_arm.get(f"{fam}_new", {})
        common = sorted(set(old) & set(new))
        for l in common:
            o, n = old[l], new[l]
            L.append(f"| {fam} | {l:g} | {o['bwt']:+.2f} | {n['bwt']:+.2f} | "
                     f"{n['bwt'] - o['bwt']:+.2f} | {o['acc']:.2f} | {n['acc']:.2f} |")
    L.append("")

    # ---------------- S3: main table ----------------
    L.append("# S3  main table at each arm's own best lambda\n")
    for T in (5, 10):
        L.append(f"\n## T = {T}\n")
        L.append("| arm | n | BWT | acc |")
        L.append("|---|---|---|---|")
        rows_for = {}
        for key, rows in mainres.items():
            arm, nt = key.split("|")
            if int(nt) != T:
                continue
            a = agg(rows)
            if a:
                rows_for[arm] = a
                L.append(f"| {arm} | {a['n']} | {a['bwt']:+.2f} ± {a['bwt_sd']:.2f} | "
                         f"{a['acc']:.2f} ± {a['acc_sd']:.2f} |")
        # paired t-test ours vs the rest, as exp12 did
        ours = [r["bwt"] for r in mainres.get(f"ours|{T}", []) if "bwt" in r]
        if len(ours) >= 3:
            try:
                from scipy import stats as sps
                L.append("\n")
                L.append("| comparison | dBWT | p (BWT) | dACC | p (acc) |")
                L.append("|---|---|---|---|---|")
                for arm in sorted(rows_for):
                    if arm == "ours":
                        continue
                    other = [r["bwt"] for r in mainres.get(f"{arm}|{T}", [])
                             if "bwt" in r]
                    oacc = [r["final_avg"] for r in mainres.get(f"{arm}|{T}", [])
                            if "final_avg" in r]
                    ouracc = [r["final_avg"] for r in mainres.get(f"ours|{T}", [])
                              if "final_avg" in r]
                    if len(other) != len(ours):
                        continue
                    _, p1 = sps.ttest_rel(ours, other)
                    _, p2 = sps.ttest_rel(ouracc, oacc)
                    L.append(f"| ours vs {arm} | {st.mean(ours) - st.mean(other):+.2f} | "
                             f"{p1:.4f} | {st.mean(ouracc) - st.mean(oacc):+.2f} | "
                             f"{p2:.4f} |")
            except ImportError:
                L.append("\n(scipy unavailable, skipped tests)")
    out = "\n".join(L) + "\n"
    open("ei_verdict.md", "w").write(out)
    print(out)
    # machine-readable best lambdas for the main stage
    json.dump({a: b[0] for a, b in best.items()}, open("best_lams.json", "w"), indent=1)


if __name__ == "__main__":
    main()

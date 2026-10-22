"""cmp_ns8.py -- capped vs uncapped dendrite budget, per rung and task count.

Reads the original ladder sweep (nseg = ntask) and the capped one
(nseg = min(8, T)), selects each rung's own lambda* by held-out-free mean
final_avg (the same rule the ladder report uses), and prints the paired
comparison.  Seed pairing is preserved: both runs use the same permutation
order and the same seeds, so the difference is measured within seed.

usage:  python cmp_ns8.py "results_lad/*.json" "results_ns8/*.json"
"""
import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict

LABEL = {
    "mlp_ewc": "R0   PlainMLP + EWC (no dendrites)",
    "mlp_nosig": "R0b  activation control",
    "ad_nocx": "R1   dendrite params, ctx=0",
    "ad_ctx": "R2   + context modulation",
    "ad_prot": "R3   + branch protection",
    "ad_usage": "R4   + usage weighting (= ours)",
    "ad_usage_ce": "R5   R4 with a CE Fisher",
}


def load(pats):
    cells = defaultdict(list)
    for pt in pats:
        for f in sorted(glob.glob(pt)):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            for block in ("sweep", "main"):
                for _, recs in d.get(block, {}).items():
                    for r in recs:
                        if "bwt" in r and "final_avg" in r:
                            cells[(r["arm"], r["ntask"], float(r["lam"]))].append(r)
    return cells


def mean(v):
    v = [x for x in v if x == x]
    return st.mean(v) if v else float("nan")


def paired_p(a, b):
    """two-sided paired t-test p-value, no scipy."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return float("nan")
    m = st.mean(d)
    s = st.stdev(d)
    if s == 0:
        return 0.0 if m != 0 else 1.0
    t = m / (s / math.sqrt(n))
    return math.erfc(abs(t) / math.sqrt(2))


def best(cells, arm, nt):
    """lambda* = argmax of mean final_avg, plus the per-seed records there."""
    pts = [(lam, recs) for (a, t, lam), recs in cells.items() if a == arm and t == nt]
    if not pts:
        return None
    lam, recs = max(pts, key=lambda kv: mean([r["final_avg"] for r in kv[1]]))
    recs = sorted(recs, key=lambda r: r["seed"])
    return {"lam": lam, "recs": recs,
            "acc": mean([r["final_avg"] for r in recs]),
            "bwt": mean([r["bwt"] for r in recs]),
            "npar": recs[0].get("n_param"), "nseg": recs[0].get("nseg")}


def main():
    base = load([sys.argv[1]])
    cap = load([sys.argv[2]])
    tasks = sorted({t for (_, t, _) in cap})
    print("capped-vs-uncapped dendrite budget, own lambda* per rung\n")
    print("%-38s %5s %18s %18s %9s %8s"
          % ("rung", "T", "uncapped (acc/bwt)", "capped (acc/bwt)", "dAcc", "p"))
    for nt in tasks:
        for arm in LABEL:
            b, c = best(base, arm, nt), best(cap, arm, nt)
            if not b or not c:
                continue
            a1 = [r["final_avg"] for r in b["recs"]]
            a2 = [r["final_avg"] for r in c["recs"]]
            n = min(len(a1), len(a2))
            p = paired_p(a1[:n], a2[:n])
            print("%-38s %5d %8.2f / %+7.2f %8.2f / %+7.2f %+9.2f %8.3f"
                  % (LABEL[arm], nt, b["acc"], b["bwt"], c["acc"], c["bwt"],
                     c["acc"] - b["acc"], p))
        print()

    print("budget actually used (params at lambda*):")
    for nt in tasks:
        for arm in ("ad_ctx", "ad_prot", "ad_usage"):
            b, c = best(base, arm, nt), best(cap, arm, nt)
            if b and c:
                print("  T=%-3d %-12s nseg %2s -> %2s   params %7s -> %7s"
                      % (nt, arm, b["nseg"], c["nseg"], b["npar"], c["npar"]))

    print("\nthe claim that has to survive the cap: ad_usage vs mlp_nosig")
    for nt in tasks:
        b, c = best(base, "mlp_nosig", nt), best(cap, "ad_usage", nt)
        if not b or not c:
            continue
        a1 = [r["final_avg"] for r in b["recs"]]
        a2 = [r["final_avg"] for r in c["recs"]]
        n = min(len(a1), len(a2))
        print("  T=%-3d capped ad_usage %.2f vs uncapped-base mlp_nosig %.2f  diff %+.2f  p=%.3f"
              % (nt, c["acc"], b["acc"], c["acc"] - b["acc"], paired_p(a1[:n], a2[:n])))


if __name__ == "__main__":
    main()

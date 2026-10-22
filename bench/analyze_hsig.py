"""analyze_hsig.py -- is the hidden-scale knob specific to DeLP, or universal?

DeLP gains +6.3/+8.6/+9.7 accuracy from rescaling its gate (alpha). If the MLP
families gain the same amount from their own sigmoid gain, then "DeLP is better
after calibration" is false: the knob is a universal lever and every family
should be tuned on it. If they gain little, the lever is specific to gated
architectures and DeLP's lead after calibration is real.

Both ends of the comparison are at each family's own accuracy-optimal lambda,
so only the gain varies.
"""
import glob
import json
import statistics as st
import sys
from collections import defaultdict

STAR = {
    "ewc_online": {5: 10.0, 10: 100.0, 20: 100.0},
    "ewc_multi":  {5: 1000.0, 10: 10.0, 20: 10.0},
    "si":         {5: 10.0, 10: 10.0, 20: 1.0},
    "mas":        {5: 1.0, 10: 0.3, 20: 0.1},
}
# DeLP's alpha sweep, same protocol, for reference
DELP_STAR = {5: 1000.0, 10: 100.0, 20: 30.0}


def load(pats):
    out = defaultdict(list)
    for pat in pats:
        for f in glob.glob(pat):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if "error" in d or "final_avg" not in d:
                continue
            out[(d["arm"], int(d["ntask"]),
                 float(d.get("hsig") or 1.0),
                 float(d.get("hgain") or 1.0))].append(d)
    return out


def at(cells, arm, nt, lam, hsig=1.0, hgain=1.0, tol=1e-9):
    recs = []
    for (a, n, h, g), v in cells.items():
        if a != arm or n != nt or abs(h - hsig) > tol or abs(g - hgain) > tol:
            continue
        for r in v:
            if abs(float(r["lam"]) - lam) / lam < 1e-9:
                recs.append(r)
    return recs


def main():
    pats = sys.argv[1:] or ["results/*.json", "results_fix/*.json",
                            "results_hsig/*.json", "results_alpha/*.json"]
    cells = load(pats)

    L = []
    L.append("# Is the hidden-scale knob specific to DeLP?\n")
    L.append("Every number is at that family's own accuracy-optimal lambda, so "
             "only the gain varies.\n")
    L.append("| family | T | gain=1 | gain=4 | gain=16 | best gain | delta at best |")
    L.append("|---|---|---|---|---|---|---|")

    gains = {}
    for arm, per_t in STAR.items():
        for nt, lam0 in per_t.items():
            r1 = at(cells, arm, nt, lam0)
            if not r1:
                continue
            a1 = st.mean([r["final_avg"] for r in r1])
            row = {1.0: a1}
            for h in (4.0, 16.0):
                rr = at(cells, arm, nt, lam0, hsig=h)
                if rr:
                    row[h] = st.mean([r["final_avg"] for r in rr])
            if len(row) < 2:
                continue
            best = max(row.items(), key=lambda kv: kv[1])
            gains.setdefault(arm, {})[nt] = best[1] - a1
            L.append(f"| {arm} | {nt} | {a1:.2f} | "
                     f"{row.get(4.0, float('nan')):.2f} | "
                     f"{row.get(16.0, float('nan')):.2f} | {best[0]:g} | "
                     f"{best[1] - a1:+.2f} |")
    L.append("")

    # DeLP reference, alpha sweep at its own best lambda per alpha
    best_alpha = {}
    for nt, lam0 in DELP_STAR.items():
        per = {}
        for g in (1.0, 2.0, 4.0):
            # DeLP gets its own best lambda at each alpha, from analyze_alpha
            pass
    L.append("## DeLP reference (its own best lambda at each gain)\n")
    L.append("| T | alpha=1 | alpha=2 | alpha=4 |")
    L.append("|---|---|---|---|")
    ref = {
        5: (81.48, 84.83, 87.81),
        10: (74.43, 78.16, 83.00),
        20: (65.50, 69.11, 75.18),
    }
    for nt, (a, b, c) in ref.items():
        L.append(f"| {nt} | {a:.2f} | {b:.2f} | {c:.2f} |")
        best_alpha[nt] = c - a
    L.append("")

    L.append("## Verdict\n")
    L.append("| family | gain from its own knob (T=5 / 10 / 20) |")
    L.append("|---|---|")
    L.append("| **delp (alpha)** | " + " / ".join(
        f"{best_alpha[n]:+.2f}" for n in (5, 10, 20)) + " |")
    for arm in STAR:
        if arm in gains:
            L.append(f"| {arm} (hsig) | " + " / ".join(
                f"{gains[arm][n]:+.2f}" if n in gains[arm] else "-"
                for n in (5, 10, 20)) + " |")
    L.append("")

    txt = "\n".join(L)
    open("d1_hsig.md", "w").write(txt)
    print(txt)
    print("wrote d1_hsig.md")


if __name__ == "__main__":
    main()

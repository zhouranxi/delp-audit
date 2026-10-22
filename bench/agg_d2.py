"""Aggregate the D2 (dual-compartment) sweep.  Reads d2_summary.csv and prints
one row per (arm, coupling mode), which is the granularity the paper needs:
the whole point of D2 is whether the *compartment* or the *slow current* pays.
"""
import csv, statistics as st
from collections import defaultdict

rows = list(csv.DictReader(open("d2_summary.csv")))
print("cols:", list(rows[0].keys()), "n=", len(rows))

g = defaultdict(list)
for r in rows:
    g[(r["arm"], r["couple"])].append(r)

hdr = "%-20s %-10s %6s %6s %8s %3s"
print(hdr % ("arm", "couple", "acc", "probe", "spikes", "n"))
best = []
for (arm, cp), rs in sorted(g.items()):
    acc = st.mean(float(x["acc"]) for x in rs)
    pb = st.mean(float(x["probe"]) for x in rs)
    sp = st.mean(float(x["spikes"]) for x in rs)
    npar = st.mean(int(x["nparam"]) for x in rs)
    print(hdr % (arm, cp, "%.2f" % acc, "%.1f" % pb, "%.0f" % sp, len(rs)))
    best.append((acc, arm, cp, npar))

print()
print("ranked by accuracy (mean over every gap / seed in the cell):")
for acc, arm, cp, npar in sorted(best, reverse=True):
    print("  %6.2f  %-16s %-5s  params=%d" % (acc, arm, cp, npar))

# The comparison the claim rests on: does the compartment add anything over a
# single membrane carrying the same slow signal?  Accuracy is saturated for most
# gaps, so the discriminating columns are the longest gap and the context probe
# (same sensory input, different context -> different output).
ARMS = [("point", "thr"), ("point_gate", "thr"), ("point_slow", "thr"),
        ("alif", "thr"), ("dual", "add"), ("dual", "mul"), ("dual", "thr")]
print()
print("per gap -- acc / probe  (mean over seeds)")
print("%-6s" % "gap" + "".join("%22s" % (a + "|" + c) for a, c in ARMS))
for gap in sorted({int(r["gap"]) for r in rows}):
    line = "%-6d" % gap
    for arm, cp in ARMS:
        sel = [(float(x["acc"]), float(x["probe"])) for x in rows
               if x["arm"] == arm and x["couple"] == cp and int(x["gap"]) == gap]
        line += "%22s" % ("%.1f / %.0f" % (st.mean(a for a, _ in sel), st.mean(p for _, p in sel))
                          if sel else "-")
    print(line)

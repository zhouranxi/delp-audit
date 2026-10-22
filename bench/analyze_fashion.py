"""analyze_fashion.py -- 8-arm corrected comparison table on Permuted-Fashion-MNIST.

Aggregates results_fashion/main_*.json shards, prints the T=5/T=10 table
(mean +- std over 5 seeds) and paired t-tests of ours vs each arm.
"""
import collections
import glob
import json

import numpy as np
from scipy import stats

recs = collections.defaultdict(list)
for f in sorted(glob.glob("results_fashion/main_*.json")):
    d = json.load(open(f))
    for key, rs in d.get("main", {}).items():
        recs[key].extend(rs)

ARMS = ["ours", "si_new", "mas_new", "ewc", "pack_new",
        "si_old", "mas_old", "pack_old"]
LABEL = {"ours": "DeLP", "si_new": "SI (fixed)", "mas_new": "MAS (fixed)",
         "ewc": "EWC", "pack_new": "PackNet (fixed)", "si_old": "SI (orig.)",
         "mas_old": "MAS (orig.)", "pack_old": "PackNet (orig.)"}

for nt in (5, 10):
    print(f"\n=== T = {nt} ===")
    print(f"{'arm':20s} {'final_avg':>16s} {'BWT':>16s}")
    data = {}
    for arm in ARMS:
        key = f"{arm}|{nt}"
        rs = [r for r in recs.get(key, []) if r.get("error") is None]
        if not rs:
            print(f"{LABEL[arm]:20s}  MISSING")
            continue
        fa = [r["final_avg"] for r in rs]
        bw = [r["bwt"] for r in rs]
        data[arm] = (fa, bw)
        print(f"{LABEL[arm]:20s} {np.mean(fa):7.2f} +- {np.std(fa, ddof=1):4.2f} "
              f"(n={len(rs)}) {np.mean(bw):7.2f} +- {np.std(bw, ddof=1):4.2f}")
    if "ours" in data:
        fa_o = np.array(data["ours"][0])
        bw_o = np.array(data["ours"][1])
        for arm in ARMS:
            if arm == "ours" or arm not in data:
                continue
            fa_b = np.array(data[arm][0])
            bw_b = np.array(data[arm][1])
            t_acc, p_acc = stats.ttest_rel(fa_o, fa_b)
            t_bwt, p_bwt = stats.ttest_rel(bw_o, bw_b)
            print(f"  ours vs {LABEL[arm]:18s} acc d={np.mean(fa_o-fa_b):+6.2f} p={p_acc:.4f}   "
                  f"bwt d={np.mean(bw_o-bw_b):+6.2f} p={p_bwt:.4f}")

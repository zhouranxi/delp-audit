"""aggregate_fashion.py -- pick per-arm lambda from exp6_fixed_fashion sweep shards.

Same selection semantics as exp6_fixed.pick_lambda with rule='val':
maximize mean val_bwt, ties broken on mean val_final_avg.
Usage: python aggregate_fashion.py results_fashion/sweep*.json
Writes lam_map_fashion.json and prints the table.
"""
import collections
import glob
import json
import sys

import numpy as np

shards = []
for pat in sys.argv[1:]:
    shards.extend(sorted(glob.glob(pat)))
assert shards, "no shard files given"

blocks = collections.defaultdict(list)
for s in shards:
    d = json.load(open(s))
    for key, recs in d.get("sweep", {}).items():
        blocks[key].extend(recs)

by_arm = collections.defaultdict(dict)
for key, recs in blocks.items():
    arm, lam = key.rsplit("@", 1)
    ok = [r for r in recs if r.get("error") is None and "val_bwt" in r]
    if len(ok) < len(recs):
        print(f"WARNING {key}: {len(recs)-len(ok)} failed records")
    by_arm[arm][float(lam)] = {
        "val_bwt_mean": float(np.mean([r["val_bwt"] for r in ok])),
        "val_final_avg_mean": float(np.mean([r["val_final_avg"] for r in ok])),
        "n": len(ok),
    }

lam_map = {}
for arm in sorted(by_arm):
    grid = by_arm[arm]
    best = max(grid.items(),
               key=lambda kv: (kv[1]["val_bwt_mean"], kv[1]["val_final_avg_mean"]))
    lam_map[arm] = best[0]
    print(f"{arm:10s} -> lam={best[0]:<8g} val_bwt={best[1]['val_bwt_mean']:.2f} "
          f"val_acc={best[1]['val_final_avg_mean']:.2f} (n={best[1]['n']})")
    for lam in sorted(grid):
        mark = " <-- selected" if lam == best[0] else ""
        print(f"    lam={lam:<8g} val_bwt={grid[lam]['val_bwt_mean']:.2f} "
              f"val_acc={grid[lam]['val_final_avg_mean']:.2f}{mark}")

json.dump(lam_map, open("lam_map_fashion.json", "w"), indent=1)
print("WROTE lam_map_fashion.json")

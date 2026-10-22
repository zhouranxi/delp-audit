"""d2_readout.py -- is the long-gap accuracy real memory, or a readout leak?

The task: a context cue fires at t < ncue, then a silent gap, then a sensory
template. The label is (context, template), so answering needs the cue.

The readout (`rate`) is the firing rate averaged over the WHOLE trial. That
average contains the spikes the cue itself elicited at t < ncue, so a model can
predict the label from the time average without carrying any state across the
gap. Evidence that this is not hypothetical: at gap=320 `point_slow` has a
context probe of 28 (chance is 33) yet 97% task accuracy -- the state at the end
of the gap is empty, so the cue must be arriving through the average.

This trains each arm twice, once with the historical whole-trial readout
(RATE_WINDOW=0) and once reading only the last `win` steps. If the long-gap
advantage survives a restricted readout it is memory; if it collapses to chance
it was the average, and the task (not the model) needs fixing before D2 can
claim anything about holding context across a delay.
"""
import argparse
import json
import os
import sys
import time

import torch

import d2_comp_snn as M

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cfg_for(couple, nh=96, ntrain=4000, ntest=1000):
    return dict(C=3, M=3, nh=nh, ntrain=ntrain, ntest=ntest, epochs=20, bs=128,
                lr=0.004, gamma=1.0, tau_s=0.85, tau_a=0.97, kappa=0.8,
                alif_rho=0.9, alif_beta=0.5, eval_every=20, data_seed=1,
                couple=couple, ncue=3)


ARMS = [("point", "-"), ("point_slow", "thr"), ("dual", "add"), ("alif", "thr")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", default="80,320")
    ap.add_argument("--wins", default="0,15",
                    help="0 = whole trial (historical), else last k steps")
    ap.add_argument("--arms", default="", help="comma list of arm names")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gaps = [int(x) for x in a.gaps.split(",") if x.strip()]
    wins = [int(x) for x in a.wins.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    want = [x for x in a.arms.split(",") if x.strip()]
    arms = [p for p in ARMS if not want or p[0] in want]

    rows = []
    for arm, couple in arms:
        for gap in gaps:
            for win in wins:
                for seed in seeds:
                    M.RATE_WINDOW = win
                    t0 = time.time()
                    rec = {"arm": arm, "couple": couple, "gap": gap,
                           "rate_window": win, "seed": seed}
                    try:
                        r = M.run(arm, gap, seed, cfg_for(couple))
                        rec.update(acc=r["acc"], ctx_probe=r["ctx_probe"],
                                   spikes=r["spikes"], T=r["T"])
                    except Exception as exc:                    # noqa: BLE001
                        rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["elapsed"] = round(time.time() - t0, 1)
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
                    tmp = a.out + ".part"
                    with open(tmp, "w") as fh:
                        json.dump({"rows": rows}, fh, indent=1)
                    os.replace(tmp, a.out)

    # summary: does the restricted readout kill the long-gap advantage?
    print("\n=== acc by readout ===")
    print(f"{'arm':12s} {'gap':>4s} {'whole-trial':>12s} {'last-window':>12s}")
    by = {}
    for r in rows:
        if "acc" in r:
            by[(r["arm"], r["gap"], r["rate_window"])] = by.get(
                (r["arm"], r["gap"], r["rate_window"]), [])
            by[(r["arm"], r["gap"], r["rate_window"])].append(r["acc"])
    for arm, _ in arms:
        for gap in gaps:
            full = by.get((arm, gap, 0))
            last = by.get((arm, gap, 15))
            print(f"{arm:12s} {gap:4d} "
                  f"{(sum(full)/len(full)) if full else float('nan'):12.1f} "
                  f"{(sum(last)/len(last)) if last else float('nan'):12.1f}")


if __name__ == "__main__":
    main()

"""gen_ladgain.py -- build the joint (gain x lambda) grid for the ladder.

Why a joint grid
----------------
The ladder's first pass tuned every rung on lambda only, holding each rung's own
free scale at its default. That is not an equal-effort comparison: the MLP rungs
carry an untuned gain inside the sigmoid and the dendritic rungs carry an untuned
scale on the pre-activation, and EWC's penalty (theta - theta_bar)^2 is
scale-sensitive even though kWTA's function class is not. The peer-gain probe
(665 cells, audit section 5.17) showed tuning that scale is worth +3 to +10
points and is worth MORE to the dendritic arm than to any baseline -- so any
step in the ladder measured without it is a lower bound of unknown tightness.

This grid therefore gives every rung the same two-dimensional budget:
  gain in {0.25, 1, 4, 16}   x   lambda in {1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4}
at T in {5, 10, 20}, 5 seeds, and selection is on held-out accuracy (same rule as
before). Gain stops at 16 because sigmoid(16t) is already a step function to
within float32 for |t| > 0.3, i.e. 64 would be a no-op for the mlp rungs and a
purely destructive logit-temperature for the rest.

Lambda stops at 1e4 on purpose: the old grid ran to 1e5 and its top edge was
where the degenerate "select on BWT" rule always landed, so the edge point
measured truncation, not a peak.

usage
    python gen_ladgain.py > jobs_ladgain.txt
"""
RUNGS = ["mlp_ewc", "mlp_nosig", "ad_nocx", "ad_ctx", "ad_prot", "ad_usage",
         "ad_usage_ce"]
TS = [5, 10, 20]
GAINS = [0.25, 1.0, 4.0, 16.0]
LAMS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
SEEDS = [0, 1, 2, 3, 4]
NLANE = 4


def g(x):
    """0.25 -> '0.25', 16.0 -> '16' (filenames stay short but unambiguous)."""
    return ("%g" % x)


def main():
    rows = []
    for t in TS:                       # T outermost so lanes mix cheap and dear
        for r in RUNGS:
            for gain in GAINS:
                for lam in LAMS:
                    for s in SEEDS:
                        tag = f"G_{r}_T{t}_g{g(gain)}_l{g(lam)}_s{s}"
                        out = f"results_ladgain/{tag}.json"
                        rows.append((tag, f"ladder.py --rung {r} --nt {t} "
                                          f"--seed {s} --lam {g(lam)} "
                                          f"--hgain {g(gain)} --hsig {g(gain)} "
                                          f"--out {out}"))
    for i, (tag, args) in enumerate(rows):
        print(f"{i % NLANE}\t{tag}\t{args}")
    import sys
    print(f"# {len(rows)} cells, {NLANE} lanes", file=sys.stderr)


if __name__ == "__main__":
    main()

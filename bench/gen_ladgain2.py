"""gen_ladgain2.py -- extend the joint grid along the gain axis.

The first joint grid (results_ladgain, 3780 cells) put gain* on the TOP EDGE
of the grid in 19 of 21 (rung, T) blocks: the optimiser wanted gain > 16 and
the grid stopped at 16.  An argmax on an edge is a truncation artefact, not a
measurement -- exactly the rule that invalidated the old lambda grids -- so
none of the section 1/2/3 numbers in ladgain.md can be used until the axis is
extended.

This grid:

    gain  16, 64, 256, 1024          (16 is the overlap row, kept so the two
                                      grids can be spliced at a shared point)
    lam   0.1 .. 1e5, half-decade    (lambda* moves DOWN as gain goes up, so
                                      the low end has to be populated too;
                                      mlp_ewc at T=5 already sat on the 1e4
                                      ceiling, hence the extension upward)
    seeds 0..4                       (paired tests need the same seed sets)

Cells already present are skipped by ladder.py itself, so re-running this is
idempotent and the 180 cells of the gain=16 overlap row are not recomputed.

usage
    python gen_ladgain2.py > jobs_ladgain2.txt
"""
import sys

RUNGS = ["mlp_ewc", "mlp_nosig", "ad_nocx", "ad_ctx", "ad_prot", "ad_usage",
         "ad_usage_ce"]
TS = [5, 10, 20]
GAINS = [16, 64, 256, 1024]
LAMS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
NSEED = 5

# seconds per cell on this box, measured (T=5 ~6.5 s, T=20 ~27 s), and scaled
# by the number of cells actually queued so LPT balances the real work rather
# than the wall-clock of the first grid.
COST = {5: 6.5, 10: 13.0, 20: 27.0}
NLANE = 4

# the gain=16 row of the previous grid already covers these lambdas
HAVE16 = {1, 3, 10, 30, 100, 300, 1000, 3000, 10000}


def main():
    nnew = sum(1 for g in GAINS for l in LAMS
               if not (g == 16 and (l in HAVE16 or float(int(l)) in HAVE16)))
    ncell = nnew * NSEED
    blocks = [(r, t, ncell * COST[t]) for t in TS for r in RUNGS]
    blocks.sort(key=lambda b: -b[2])
    lanes, load = [[] for _ in range(NLANE)], [0.0] * NLANE
    for r, t, c in blocks:
        i = load.index(min(load))
        lanes[i].append((r, t, c))
        load[i] += c

    gs = ",".join(f"{g:g}" for g in GAINS)
    ls = ",".join(f"{l:g}" for l in LAMS)
    for i, lane in enumerate(lanes):
        for r, t, _ in lane:
            tag = f"X_{r}_T{t}"
            args = (f"ladder.py --rung {r} --nt {t} --outdir results_ladgain "
                    f"--gains {gs} --lams {ls} --seeds 0,1,2,3,4")
            print(f"{i}\t{tag}\t{args}")
    for i, l in enumerate(load):
        print(f"# lane{i} ~{l/3600:.1f} h", file=sys.stderr)
    print(f"# {len(blocks)} blocks, {ncell} cells/block, {len(blocks)*ncell} total",
          file=sys.stderr)


if __name__ == "__main__":
    main()

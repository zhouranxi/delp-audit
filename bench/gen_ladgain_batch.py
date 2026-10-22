"""gen_ladgain_batch.py -- schedule the joint grid as (rung, T) blocks.

Per-cell processes waste ~22 s of torch/CUDA startup on 4-6 s of T=5 work, so
the grid is cut into 21 blocks (7 rungs x 3 T) of 180 cells each and the blocks
are packed onto 4 lanes with LPT (longest processing time first). Lanes are then
reasonably balanced even though a T=20 block costs ~5x a T=5 block.

usage
    python gen_ladgain_batch.py > jobs_ladgain_batch.txt
"""
import sys

RUNGS = ["mlp_ewc", "mlp_nosig", "ad_nocx", "ad_ctx", "ad_prot", "ad_usage",
         "ad_usage_ce"]
TS = [5, 10, 20]
# seconds per cell, measured on this box (T=5: 6.7 s, T=20: 29 s, incl. val load)
COST = {5: 6.5, 10: 13.0, 27: 0, 20: 27.0}
NCELL = 4 * 9 * 5          # gains x lambdas x seeds
NLANE = 4


def main():
    blocks = [(r, t, NCELL * COST[t]) for t in TS for r in RUNGS]
    blocks.sort(key=lambda b: -b[2])          # LPT: biggest blocks first
    lanes, load = [[] for _ in range(NLANE)], [0.0] * NLANE
    for r, t, c in blocks:
        i = load.index(min(load))             # least loaded lane
        lanes[i].append((r, t, c))
        load[i] += c
    for i, lane in enumerate(lanes):
        for r, t, _ in lane:
            tag = f"B_{r}_T{t}"
            args = f"ladder.py --rung {r} --nt {t} --outdir results_ladgain"
            print(f"{i}\t{tag}\t{args}")
    for i, l in enumerate(load):
        print(f"# lane{i} load ~{l/60:.0f} min", file=sys.stderr)
    print(f"# {len(blocks)} blocks, {len(blocks)*NCELL} cells, {NLANE} lanes",
          file=sys.stderr)


if __name__ == "__main__":
    main()

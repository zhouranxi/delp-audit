"""gen_jobs.py -- emit the job list for the D1 benchmark grid.

One line per run:   <gpu> <module> <args...>
Run with:  xargs -P 4 -L 1 bash -c '<line>'
"""
import argparse, itertools, json

LAM_EWC = [10.0, 100.0, 1000.0, 10000.0]
LAM_SI = [0.1, 1.0, 10.0, 100.0]
LAM_MAS = [0.1, 1.0, 10.0, 100.0]
LAM_LWF = [0.1, 1.0, 10.0, 100.0]
LAM_DELP = [100.0, 1000.0, 5000.0, 20000.0]

GRID = {
    "naive": [0.0],
    "ad_only": [0.0],
    "ewc_online": LAM_EWC,
    "ewc_multi": LAM_EWC,
    "si": LAM_SI,
    "mas": LAM_MAS,
    "lwf": LAM_LWF,
    "delp": LAM_DELP,
    "packnet": [0.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="full")
    ap.add_argument("--ngpu", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ntasks", default="5,10,20")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    nts = [int(s) for s in a.ntasks.split(",")]

    jobs = []
    if a.phase == "smoke":
        for arm in ("naive", "ewc_online", "si", "mas", "delp", "packnet", "lwf"):
            jobs.append((arm, 5, GRID[arm][1] if len(GRID[arm]) > 1 else 0.0, 0, 0.5))
        jobs.append(("ewc_multi", 5, 1000.0, 0, 0.5))
        jobs.append(("ewc_online", 20, 1000.0, 0, 0.5))
        jobs.append(("delp", 20, 5000.0, 0, 0.5))
    else:
        for arm, lams in GRID.items():
            # lam=0 is emitted for every regularised arm: it must reproduce `naive`
            # bit-for-bit, which is the harness' correctness unit test.
            lams = sorted(set(list(lams) + [0.0])) if lams != [0.0] else lams
            for nt in nts:
                for lam in lams:
                    prunes = [0.2, 0.5] if arm == "packnet" else [0.5]
                    for pr in prunes:
                        for s in seeds:
                            jobs.append((arm, nt, lam, s, pr))

    i = 0
    for arm, nt, lam, s, pr in jobs:
        gpu = i % a.ngpu
        tag = f"{arm}_T{nt}_lam{lam:g}_p{pr}_s{s}"
        print(f"{gpu}\t{tag}\tbench_cl.py --arm {arm} --ntask {nt} --lam {lam:g} "
              f"--prune {pr} --seed {s} --out results/{tag}.json")
        i += 1
    print(f"# {len(jobs)} jobs, {a.ngpu} gpus", flush=True)


if __name__ == "__main__":
    main()

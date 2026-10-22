"""exp12: seeded re-run of the whole comparison table (reproducibility pass).

exp6 produced the baseline arms without an explicit torch.manual_seed, so model
initialisation depended on ambient global RNG state: a re-run gave slightly
different numbers. exp6 now seeds each arm; this script re-runs (i) the lambda
screening and (ii) both protocols for all four arms so that Tables 1-2 and the
hyperparameter table come from one seeded pipeline.

Arms: DeLP (ours), Synaptic Intelligence, MAS, PackNet. 5 seeds, 5 and 10 tasks.
Lambda grids unchanged: DeLP {100, 1000, 5000}, SI {0.1, 1, 10}, MAS {0.1, 1, 10}.
"""
import json, time, statistics as st
import torch
from torch.utils.data import DataLoader, TensorDataset
from exp6_baselines import (SEEDS, load_mnist, get_perms, summarize,
                            run_ours, run_si_impl, run_mas_impl, run_packnet)

SCREEN_SEEDS = [0, 1]
LAM_DELP = [100.0, 1000.0, 5000.0]
LAM_REF = [0.1, 1.0, 10.0]


def main():
    t0 = time.time()
    data = load_mnist()
    res = {}

    # ---- lambda screening (5 tasks) ----
    screen = {}
    for lam in LAM_DELP:
        rs = [run_ours(s, data, get_perms(s, 5), 5, lam) for s in SCREEN_SEEDS]
        screen[f"delp_{int(lam)}"] = summarize(rs)
        print(f"[screen delp {int(lam)}] {screen[f'delp_{int(lam)}']}", flush=True)
    for arm, fn in [("si", run_si_impl), ("mas", run_mas_impl)]:
        for lam in LAM_REF:
            rs = [fn(s, data, get_perms(s, 5), 5, lam) for s in SCREEN_SEEDS]
            screen[f"{arm}_{lam}"] = summarize(rs)
            print(f"[screen {arm} {lam}] {screen[f'{arm}_{lam}']}", flush=True)
    res["screen"] = screen
    best = {"delp": max(LAM_DELP, key=lambda l: (screen[f"delp_{int(l)}"]["bwt_mean"],
                                                 screen[f"delp_{int(l)}"]["final_avg_mean"]))}
    for arm in ("si", "mas"):
        best[arm] = max(LAM_REF, key=lambda l: (screen[f"{arm}_{l}"]["bwt_mean"],
                                                screen[f"{arm}_{l}"]["final_avg_mean"]))
    res["best_lam"] = best
    print("[best]", best, flush=True)

    # ---- main runs, both protocols ----
    for ntask in (5, 10):
        res[f"{ntask}tasks"] = {}
        for name, fn in ((("ours", lambda s, p, n=ntask: run_ours(s, data, p, n, best["delp"]))),
                         (("si", lambda s, p, n=ntask: run_si_impl(s, data, p, n, best["si"]))),
                         (("mas", lambda s, p, n=ntask: run_mas_impl(s, data, p, n, best["mas"]))),
                         (("packnet", lambda s, p: run_packnet(s, data, p, ntask)))):
            runs = [fn(s, get_perms(s, ntask)) for s in SEEDS]
            res[f"{ntask}tasks"][name] = {**summarize(runs), "runs": runs}
            v = res[f"{ntask}tasks"][name]
            print(f"[{ntask}T {name}] BWT {v['bwt_mean']}±{v['bwt_std']} "
                  f"final {v['final_avg_mean']}±{v['final_avg_std']}", flush=True)

    from scipy import stats as sps
    for ntask in (5, 10):
        block = res[f"{ntask}tasks"]
        so = {}
        for other in ("si", "mas", "packnet"):
            _, p1 = sps.ttest_rel([r["bwt"] for r in block["ours"]["runs"]],
                                  [r["bwt"] for r in block[other]["runs"]])
            _, p2 = sps.ttest_rel([r["final_avg"] for r in block["ours"]["runs"]],
                                  [r["final_avg"] for r in block[other]["runs"]])
            so[f"ours_vs_{other}"] = {"bwt_p": round(float(p1), 5),
                                      "finalavg_p": round(float(p2), 5)}
        block["stats"] = so
        print(f"[{ntask}T stats] {so}", flush=True)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(res, open("exp12_result.json", "w"), indent=2, default=str)
    print("DONE", res["elapsed_sec"])


if __name__ == "__main__":
    main()

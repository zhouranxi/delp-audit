"""exp11: is the 10-task accuracy ceiling a capacity wall or an operating point?

exp10 showed the loss is in the LEARNING term (mean_t a[t][t] collapses from 90%
to 19% across ten tasks) while retention stays excellent (BWT ~ -3.3). Widening
256 -> 1024 recovers only +3.4 points of learning, so the binding constraint is
not width but the growing stack of anchors pinning the shared projection.

This experiment maps the stability-plasticity frontier:
  * lambda in {0, 100, 1000, 5000, 10000}  (10 tasks, N=256, 5 seeds)
  * truncated anchor sets (keep only the M most recent anchors) at lambda=10000
    and 5000 -- a direct mechanistic probe of cumulative anchoring.
Reports BWT, final accuracy, learning term and the per-task just-learned curve,
so the result can be plotted as a Pareto front against SI.

Protocol matches exp6 (per-seed permutations) so the SI/MAS points are directly
comparable. Uses GPU 1 (GPU 2/3 host llama-server; do not touch).
"""
import json, time, statistics as st
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from exp6_baselines import (SEEDS, EPOCHS, BS, NTRAIN, LR, DEV, load_mnist,
                            permute, get_perms, eval_model, finish,
                            summarize, flat_params, kwta)
from exp4_branch_protection import ADNet
from exp10_capacity import fisher_usage

NTASK, NH = 10, 256
LAMS = [0.0, 100.0, 1000.0, 5000.0, 10000.0]


def run_delp(seed, data, perms, ntask, lam, max_anchors=None, nh=NH):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = ADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV), ntask).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors and lam > 0:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        for k, v in P.items():
                            pen = pen + (a["fisher"][k] * (v - a["theta"][k]) ** 2).sum()
                    loss = loss + lam * pen
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        anchors.append(fisher_usage(model, xs, phase, ntask))
        if max_anchors is not None and len(anchors) > max_anchors:
            anchors = anchors[-max_anchors:]
        acc.update(eval_model(model, xte, yte, perms, phase, ntask, True))
    bwt, favg = finish(acc, ntask)
    just = [acc[(t, t)] for t in range(ntask)]
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "learning_term": round(st.mean(just), 2),
            "justlearn": [round(v, 2) for v in just],
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)] for p in range(ntask)]}


def main():
    t0 = time.time()
    data = load_mnist()
    arms = {}
    for lam in LAMS:
        arms[f"lam{int(lam)}"] = dict(lam=lam, max_anchors=None)
    arms["lam10000_M5"] = dict(lam=10000.0, max_anchors=5)
    arms["lam5000_M5"] = dict(lam=5000.0, max_anchors=5)

    res = {"protocol": {"ntask": NTASK, "nh": NH, "seeds": SEEDS, "lams": LAMS}, "arms": {}}
    for name, cfg in arms.items():
        runs = [run_delp(s, data, get_perms(s, NTASK), NTASK,
                         cfg["lam"], cfg["max_anchors"]) for s in SEEDS]
        agg = summarize(runs)
        lt = round(st.mean([r["learning_term"] for r in runs]), 2)
        res["arms"][name] = {**cfg, **agg, "learning_term": lt,
                             "justlearn_mean": [round(st.mean([r["justlearn"][t] for r in runs]), 2)
                                                for t in range(NTASK)],
                             "runs": runs}
        print(f"[{name}] BWT {agg['bwt_mean']}±{agg['bwt_std']} "
              f"final {agg['final_avg_mean']}±{agg['final_avg_std']} learn {lt}", flush=True)

    # include SI / MAS 10-task points from exp6 for the Pareto comparison
    try:
        e6 = json.load(open("exp6_result.json"))
        res["reference_10t"] = {k: {kk: e6["10tasks"][k][kk]
                                    for kk in ("bwt_mean", "bwt_std", "final_avg_mean", "final_avg_std")}
                                for k in ("si", "mas")}
    except FileNotFoundError:
        res["reference_10t"] = {}

    res["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(res, open("exp11_result.json", "w"), indent=2, default=str)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "runs"})
                      for k, v in res["arms"].items()}, indent=2, default=str))
    print("FRONTIER:")
    for name, v in res["arms"].items():
        print(f"  {name:14s} BWT {v['bwt_mean']:7.2f}  acc {v['final_avg_mean']:6.2f}  "
              f"learn {v['learning_term']:6.2f}")
    for k, v in res["reference_10t"].items():
        print(f"  {k:14s} BWT {v['bwt_mean']:7.2f}  acc {v['final_avg_mean']:6.2f}")
    print("elapsed", res["elapsed_sec"])


if __name__ == "__main__":
    main()

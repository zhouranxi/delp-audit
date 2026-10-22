"""exp9: consistency run — MLP / AD / v1 under the EXACT exp6 pipeline so that
Table 1 of the paper reports one pipeline for all rows. Uses exp6 helpers
(eval_model, finish, summarize, get_perms, PlainMLP) and the DeLP runs already
stored in exp6_result.json for paired t-tests.
"""
import json, time, statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from exp6_baselines import (SEEDS, EPOCHS, BS, NTRAIN, LR, DEV, load_mnist,
                            permute, get_perms, eval_model, finish, summarize,
                            PlainMLP)
from exp4_branch_protection import ADNet

def run_arm(seed, data, perms, ntask, kind):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    if kind == "mlp":
        model = PlainMLP().to(DEV); needs_ctx = False
    else:
        model = ADNet(protected=(kind == "v1"), nseg=ntask, ntask=ntask).to(DEV)
        needs_ctx = True
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    acc = {}
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                if needs_ctx:
                    ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV), ntask).float()
                    out, _ = model(xb, ctx)
                else:
                    out, _ = model(xb)
                F.cross_entropy(out, yb).backward()
                if kind == "v1":
                    model.protect_grads(phase)
                opt.step()
        a = eval_model(model, xte, yte, perms, phase, ntask, needs_ctx)
        acc.update(a)
    bwt, favg = finish(acc, ntask)
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(ntask)]}

def main():
    t0 = time.time()
    data = load_mnist()
    ntask = 5
    res = {}
    for kind in ["mlp", "ad", "v1"]:
        runs = [run_arm(s, data, get_perms(s, ntask), ntask, kind) for s in SEEDS]
        res[kind] = {**summarize(runs), "runs": runs}
        print(f"[5T {kind}] BWT {res[kind]['bwt_mean']}±{res[kind]['bwt_std']} "
              f"final {res[kind]['final_avg_mean']}±{res[kind]['final_avg_std']}", flush=True)
    # paired tests vs DeLP runs stored in exp6
    from scipy import stats as sps
    e6 = json.load(open("exp6_result.json"))
    ours = e6["5tasks"]["ours"]["runs"]
    stats_out = {}
    for kind in res:
        _, p1 = sps.ttest_rel([r["bwt"] for r in ours], [r["bwt"] for r in res[kind]["runs"]])
        _, p2 = sps.ttest_rel([r["final_avg"] for r in ours], [r["final_avg"] for r in res[kind]["runs"]])
        stats_out[f"delp_vs_{kind}"] = {"bwt_p": round(float(p1), 5),
                                        "finalavg_p": round(float(p2), 5)}
        # also this-kind vs si/mas for the table's internal reference
    res["stats_vs_delp"] = stats_out
    res["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=2, default=str)
          if False else json.dumps({k: (v if k in ("stats_vs_delp", "elapsed_sec")
                                        else {kk: vv for kk, vv in v.items() if kk != "runs"})
                                    for k, v in res.items()}, indent=2, default=str))
    json.dump(res, open("exp9_result.json", "w"), indent=2, default=str)

if __name__ == "__main__":
    main()

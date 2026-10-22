"""exp7: fix exp6 finding — lambda must scale with number of EWC anchors.
exp6 10T with lambda=5000 (tuned on 5T) showed accelerating forgetting on later
tasks (anchor sum doubles, protection per-anchor diluted ... actually the sum
DOUBLES protection; the failure is more subtle: capacity saturation of nseg=10
segments on NH=256, and Fisher staleness). Here we test three fixes on 10T:
  A) lam scaled by ntask/5 (=10000): stronger global penalty
  B) lam=10000 + NH=512: more neurons per segment
  C) lam=10000 + NH=512 + nseg stays 10
(A vs exp6-10T isolates lambda effect; B adds capacity.)
Same protocol/seeds/perms as exp6. Paired t-tests vs exp6 10T ours.
"""
import json, time, statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from exp6_baselines import (SEEDS, EPOCHS, BS, NTRAIN, LR, DEV, load_mnist,
                            permute, kwta, get_perms, fisher_usage, eval_model,
                            finish, summarize, PlainMLP)
from exp4_branch_protection import ADNet

def run_arm(seed, data, perms, ntask, lam, nh):
    xtr, ytr, xte, yte = data
    model = ADNet(protected=True, nseg=ntask, ntask=ntask, nh=nh).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    P = {"ff.w": model.ff.weight, "ff.b": model.ff.bias,
         "out.w": model.out.weight, "out.b": model.out.bias}
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
                if anchors:
                    pen = 0.0
                    for a in anchors:
                        for k, v in P.items():
                            pen = pen + (a["fisher"][k] * (v - a["theta"][k]) ** 2).sum()
                    loss = loss + lam * pen
                loss.backward()
                model.protect_grads(phase)
                opt.step()
        anchors.append(fisher_usage(model, xs, phase, ntask))
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
        print(f"  s{seed} lam{int(lam)} nh{nh} T{phase}: "
              + " ".join(f"t{c}={acc[(phase,c)]:.0f}" for c in range(phase+1)), flush=True)
    bwt, favg = finish(acc, ntask)
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(ntask)]}

def main():
    t0 = time.time()
    data = load_mnist()
    ntask = 10
    arms = {
        "lam10k_nh256": (10000.0, 256),
        "lam10k_nh512": (10000.0, 512),
    }
    res = {}
    for name, (lam, nh) in arms.items():
        runs = [run_arm(s, data, get_perms(s, ntask), ntask, lam, nh) for s in SEEDS]
        res[name] = {**summarize(runs), "runs": runs}
        print(f"[10T {name}] BWT {res[name]['bwt_mean']}±{res[name]['bwt_std']} "
              f"final {res[name]['final_avg_mean']}±{res[name]['final_avg_std']}", flush=True)
    from scipy import stats as sps
    prev = json.load(open("exp6_result.json"))
    ours10 = prev["10tasks"]["ours"]["runs"]
    stats_out = {}
    for name in res:
        _, p1 = sps.ttest_rel([r["bwt"] for r in res[name]["runs"]],
                              [r["bwt"] for r in ours10])
        _, p2 = sps.ttest_rel([r["final_avg"] for r in res[name]["runs"]],
                              [r["final_avg"] for r in ours10])
        stats_out[f"{name}_vs_exp6_10T"] = {"bwt_p": round(float(p1), 5),
                                            "finalavg_p": round(float(p2), 5)}
    res["stats"] = stats_out
    res["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps({k: (v if k not in arms else {kk: (vv if kk != "runs" else "...") for kk, vv in v.items()}) for k, v in res.items()}, default=str))
    json.dump(res, open("exp7_result.json", "w"), indent=2, default=str)

if __name__ == "__main__":
    main()

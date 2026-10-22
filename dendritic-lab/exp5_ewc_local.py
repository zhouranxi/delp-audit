"""exp5: dendrite-localized EWC on the shared ff layer (fix for v2/v3 failures).

Diagnosis from exp4: main forgetting channel = shared feedforward weights.
v2 (hard neuron gating) collapsed capacity; v3 (per-task heads) shifted
forgetting into ff. exp5 combines:
  * v1 segment freeze (dendrite weights of non-current segments frozen)
  * EWC soft penalty on ff/out weights, with Fisher weighted by each neuron's
    ACTIVATION USAGE under old tasks (o_j = fraction of old-task samples where
    neuron j is within k-WTA winners). Heavily-used-by-old-task neurons get
    strong protection; rarely-used stay plastic -> no capacity collapse.
Arms:
  ewc_local  : Fisher * usage (localized)
  ewc_global : plain Fisher (reference, same lambda screening)
Protocol: Permuted-MNIST 5 tasks; lambda screened on seeds {0,1}; final 5 seeds;
paired t-tests vs mlp/ad/ad_prot (exp4).
"""
import json, time, statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from exp4_branch_protection import (SEEDS, NTASKS, EPOCHS, BS, NTRAIN, NH, KSEG,
                                    LR, DEV, load_mnist, permute, kwta, ADNet)

NFISHER, NUSAGE = 300, 500

def fisher_and_usage(model, xs, ys, task_id):
    """Diagonal Fisher (per-sample grads, true labels) + activation usage o_j."""
    model.eval()
    params = {"ff.w": model.ff.weight, "ff.b": model.ff.bias,
              "out.w": model.out.weight, "out.b": model.out.bias}
    fisher = {k: torch.zeros_like(v) for k, v in params.items()}
    ctx1 = F.one_hot(torch.tensor([task_id], device=DEV), NTASKS).float()
    # usage
    o = torch.zeros(NH, device=DEV)
    with torch.no_grad():
        for i in range(0, NUSAGE, 250):
            xb = xs[i:i+250].to(DEV)
            ctx = ctx1.expand(xb.size(0), -1)
            t = model.ff(xb)
            u = model.dend(ctx).view(-1, KSEG, NH)
            d, _ = u.max(dim=1)
            h = kwta(t * torch.sigmoid(d))
            o += (h > 0).float().sum(0)
    o /= NUSAGE
    # fisher: per-sample gradient of log p(y_true)
    for i in range(NFISHER):
        xb = xs[i:i+1].to(DEV); yb = ys[i:i+1].to(DEV)
        ctx = ctx1
        out, _ = model(xb, ctx)
        model.zero_grad()
        F.log_softmax(out, 1).gather(1, yb[:, None]).squeeze(1).sum().backward()
        for k, v in params.items():
            fisher[k] += v.grad.detach() ** 2
    for k in fisher:
        fisher[k] /= NFISHER
    # localize: scale ff rows / out cols by usage
    fisher["ff.w"] = fisher["ff.w"] * o.unsqueeze(1)
    fisher["ff.b"] = fisher["ff.b"] * o
    fisher["out.w"] = fisher["out.w"] * o.unsqueeze(0)
    return {"theta": {k: v.detach().clone() for k, v in params.items()},
            "fisher": fisher, "usage_mean": float(o.mean().item())}

def penalty(anchors, model, arm):
    params = {"ff.w": model.ff.weight, "ff.b": model.ff.bias,
              "out.w": model.out.weight, "out.b": model.out.bias}
    pen = 0.0
    for a in anchors:
        for k, v in params.items():
            f = a["fisher"][k] if arm == "ewc_local" else a["fisher"][k] / (a["usage_mean"] + 1e-8) * a["fisher"][k].gt(0).float().mean()
            # global arm: plain fisher (no usage scaling) -> recompute: store raw too
            pen = pen + (f * (v - a["theta"][k]) ** 2).sum()
    return pen

def run_seed(seed, arm, lam, data, perms):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = data
    model = ADNet(protected=True).to(DEV)      # v1 segment freeze included
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors = []
    acc = {}
    for phase in range(NTASKS):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV), NTASKS).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    loss = loss + lam * penalty(anchors, model, arm)
                loss.backward()
                model.protect_grads(phase)     # v1 dendrite segment freeze
                opt.step()
        if arm.startswith("ewc"):
            anchors.append(fisher_and_usage(model, xs, ys, phase))
        model.eval()
        with torch.no_grad():
            for cc in range(phase + 1):
                xt = permute(xte, perms[cc]).to(DEV)
                correct = 0
                for i in range(0, xt.size(0), 2000):
                    n = min(2000, xt.size(0) - i)
                    ctx = F.one_hot(torch.full((n,), cc, device=DEV), NTASKS).float()
                    out, _ = model(xt[i:i+2000], ctx)
                    correct += (out.argmax(1) == yte[i:i+2000].to(DEV)).sum().item()
                acc[(phase, cc)] = correct / xt.size(0) * 100
        print(f"  seed{seed} {arm} lam{int(lam)} task{phase}: " +
              " ".join(f"t{cc}={acc[(phase,cc)]:.1f}" for cc in range(phase+1)), flush=True)
    bwt = sum(acc[(NTASKS-1, c)] - acc[(c, c)] for c in range(NTASKS-1)) / (NTASKS-1)
    final_avg = sum(acc[(NTASKS-1, c)] for c in range(NTASKS)) / NTASKS
    return {"seed": seed, "arm": arm, "lam": lam, "bwt": round(bwt, 2),
            "final_avg": round(final_avg, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(NTASKS)]}

def summarize(runs):
    b = [r["bwt"] for r in runs]; f = [r["final_avg"] for r in runs]
    return {"bwt_mean": round(st.mean(b), 2), "bwt_std": round(st.pstdev(b), 2),
            "final_avg_mean": round(st.mean(f), 2), "final_avg_std": round(st.pstdev(f), 2)}

def main():
    t0 = time.time()
    data = load_mnist()
    perms_by_seed = {}
    for seed in SEEDS:
        g = torch.Generator().manual_seed(100 + seed)
        perms_by_seed[seed] = [torch.randperm(784, generator=g) for _ in range(NTASKS)]
    # ---- lambda screening on seeds {0,1}
    LAMS = [100.0, 1000.0, 5000.0]
    screen = {}
    best_lam = {}
    for arm in ["ewc_local", "ewc_global"]:
        for lam in LAMS:
            runs = [run_seed(s, arm, lam, data, perms_by_seed[s]) for s in [0, 1]]
            m = summarize(runs)
            screen[f"{arm}_lam{int(lam)}"] = m
            print(f"[screen] {arm} lam={int(lam)}: BWT {m['bwt_mean']} final {m['final_avg_mean']}", flush=True)
        best = max(LAMS, key=lambda l: (screen[f"{arm}_lam{int(l)}"]["bwt_mean"],
                                        screen[f"{arm}_lam{int(l)}"]["final_avg_mean"]))
        best_lam[arm] = best
    # ---- final 5 seeds at best lambda
    final = {}
    for arm in ["ewc_local", "ewc_global"]:
        runs = [run_seed(s, arm, best_lam[arm], data, perms_by_seed[s]) for s in SEEDS]
        final[arm] = {"lam": best_lam[arm], **summarize(runs), "runs": runs}
        m = final[arm]
        print(f"[final] {arm} lam={int(best_lam[arm])}: BWT {m['bwt_mean']}±{m['bwt_std']} "
              f"final {m['final_avg_mean']}±{m['final_avg_std']}", flush=True)
    # ---- stats vs exp4
    from scipy import stats as sps
    prev = json.load(open("exp4_result.json"))
    prev["ad_prot2"] = json.load(open("exp4b_result.json"))["ad_prot2"]
    stats_out = {"best_lam": {k: int(v) for k, v in best_lam.items()}}
    for arm in final:
        fa = [r["final_avg"] for r in final[arm]["runs"]]
        ba = [r["bwt"] for r in final[arm]["runs"]]
        for other in ["mlp", "ad", "ad_prot"]:
            db = [r["bwt"] for r in prev[other]["runs"]]
            _, p = sps.ttest_rel(ba, db)
            stats_out[f"{arm}_vs_{other}_bwt_p"] = round(float(p), 5)
            fo = [r["final_avg"] for r in prev[other]["runs"]]
            _, p2 = sps.ttest_rel(fa, fo)
            stats_out[f"{arm}_vs_{other}_finalavg_p"] = round(float(p2), 5)
    res = {"screen": screen, "final": final, "stats": stats_out,
           "elapsed_sec": round(time.time() - t0, 1)}
    print(json.dumps({k: v for k, v in res.items() if k != "final"}, indent=2, default=str))
    json.dump(res, open("exp5_result.json", "w"), indent=2, default=str)

if __name__ == "__main__":
    main()

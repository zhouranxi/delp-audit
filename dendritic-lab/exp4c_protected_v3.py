"""exp4c: v3 = v1 segment freeze + per-task output heads (context-selected).
Ablation chain: v1 isolated dendrites (small gain), v2 added neuron gating
(capacity collapse), v3 isolates the output layer channel. Permuted-MNIST
domain-IL; context one-hot available at test (AD assumption), so per-task
heads are legitimate (same 10 classes, chosen by context).
"""
import json, time, statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from exp4_branch_protection import (SEEDS, NTASKS, EPOCHS, BS, NTRAIN, NH, KSEG,
                                    LR, DEV, load_mnist, permute, kwta)

class ADNet3(nn.Module):
    def __init__(self, nh=NH, nseg=KSEG, ntask=NTASKS):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.dend = nn.Linear(ntask, nh * nseg, bias=False)
        self.heads = nn.ModuleList([nn.Linear(nh, 10) for _ in range(ntask)])
        self.nseg, self.nh = nseg, nh
    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d, _ = u.max(dim=1)
        h = kwta(t * torch.sigmoid(d))
        task_id = int(ctx[0].argmax().item())
        return self.heads[task_id](h), None
    def protect_grads(self, ctx_label):
        if self.dend.weight.grad is None:
            return
        W = self.dend.weight.grad.view(self.nseg, self.nh, -1)
        seg_mask = torch.zeros(self.nseg, 1, 1, device=W.device)
        seg_mask[ctx_label] = 1.0
        W.mul_(seg_mask)

def run_seed(seed, data, perms):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = data
    model = ADNet3().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
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
                F.cross_entropy(out, yb).backward()
                model.protect_grads(phase)
                opt.step()
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
        print(f"  seed{seed} ad_prot3 after task{phase}: " +
              " ".join(f"t{cc}={acc[(phase,cc)]:.1f}" for cc in range(phase+1)), flush=True)
    bwt = sum(acc[(NTASKS-1, c)] - acc[(c, c)] for c in range(NTASKS-1)) / (NTASKS-1)
    final_avg = sum(acc[(NTASKS-1, c)] for c in range(NTASKS)) / NTASKS
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(final_avg, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(NTASKS)]}

def main():
    t0 = time.time()
    data = load_mnist()
    runs = []
    for seed in SEEDS:
        g = torch.Generator().manual_seed(100 + seed)
        perms = [torch.randperm(784, generator=g) for _ in range(NTASKS)]
        runs.append(run_seed(seed, data, perms))
    bwts = [r["bwt"] for r in runs]; favgs = [r["final_avg"] for r in runs]
    res = {"ad_prot3": {"bwt_mean": round(st.mean(bwts), 2), "bwt_std": round(st.pstdev(bwts), 2),
                        "final_avg_mean": round(st.mean(favgs), 2),
                        "final_avg_std": round(st.pstdev(favgs), 2), "runs": runs}}
    print(f"[ad_prot3] BWT {st.mean(bwts):.2f}±{st.pstdev(bwts):.2f}  "
          f"finalAvg {st.mean(favgs):.2f}±{st.pstdev(favgs):.2f}", flush=True)
    from scipy import stats as sps
    prev = json.load(open("exp4_result.json"))
    prev["ad_prot2"] = json.load(open("exp4b_result.json"))["ad_prot2"]
    stats_out = {}
    for other in ["mlp", "ad", "ad_prot", "ad_prot2"]:
        db = [r["bwt"] for r in prev[other]["runs"]]
        _, p = sps.ttest_rel(bwts, db)
        stats_out[f"vs_{other}_bwt_p"] = round(float(p), 5)
        fa = [r["final_avg"] for r in prev[other]["runs"]]
        _, p2 = sps.ttest_rel(favgs, fa)
        stats_out[f"vs_{other}_finalavg_p"] = round(float(p2), 5)
    res["stats"] = stats_out
    res["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in res.items() if k != "ad_prot3"}, indent=2))
    json.dump(res, open("exp4c_result.json", "w"), indent=2)

if __name__ == "__main__":
    main()

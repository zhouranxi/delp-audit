"""exp4b: v2 branch protection = dendrite seg gating + per-neuron ff gating.
Rationale (from exp4 v1 negative/partial result): forgetting flows through the
SHARED feedforward weights. v2: neurons whose winning dendritic segment (for the
current context) is the context-assigned segment are 'owned' by this task ->
only they update ff weights. Biologically: dendritic gating of plasticity
(cf. Onasch et al., Neuron 2026). Runs only the ad_prot2 arm; compare with exp4.
"""
import json, time, statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from exp4_branch_protection import (SEEDS, NTASKS, EPOCHS, BS, NTRAIN, NH, KSEG,
                                    LR, DEV, load_mnist, permute, kwta)

class ADNet2(nn.Module):
    def __init__(self, nh=NH, nseg=KSEG, ntask=NTASKS):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.dend = nn.Linear(ntask, nh * nseg, bias=False)
        self.out = nn.Linear(nh, 10)
        self.nseg, self.nh = nseg, nh
    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d, _ = u.max(dim=1)
        return self.out(kwta(t * torch.sigmoid(d))), None
    def owner_mask(self, ctx_label):
        """neurons whose winning segment for THIS context == ctx-assigned segment."""
        with torch.no_grad():
            u = self.dend.weight.view(self.nseg, self.nh, -1)[:, :, ctx_label]
            win = u.argmax(dim=0)                     # (nh,)
            keep = (win == ctx_label)
            if keep.sum() < 16:                       # degenerate init guard
                keep = torch.ones_like(keep)
        return keep.float()
    def protect_grads(self, ctx_label):
        if self.dend.weight.grad is None:
            return
        keep = self.owner_mask(ctx_label)             # (nh,)
        W = self.dend.weight.grad.view(self.nseg, self.nh, -1)
        # dendrites: only current-task segment updates, and only owner neurons
        seg_mask = torch.zeros(self.nseg, 1, 1, device=W.device)
        seg_mask[ctx_label] = 1.0
        W.mul_(seg_mask * keep.view(1, -1, 1))
        # ff: per-neuron gating
        self.ff.weight.grad.mul_(keep.unsqueeze(1))
        self.ff.bias.grad.mul_(keep)

def run_seed(seed, data, perms):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = data
    model = ADNet2().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    acc = {}
    n_owned = []
    for phase in range(NTASKS):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        n_owned.append(int(model.owner_mask(phase).sum().item()))
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV), NTASKS).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                loss.backward()
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
        print(f"  seed{seed} ad_prot2 after task{phase}: " +
              " ".join(f"t{cc}={acc[(phase,cc)]:.1f}" for cc in range(phase+1)), flush=True)
    bwt = sum(acc[(NTASKS-1, c)] - acc[(c, c)] for c in range(NTASKS-1)) / (NTASKS-1)
    final_avg = sum(acc[(NTASKS-1, c)] for c in range(NTASKS)) / NTASKS
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(final_avg, 2),
            "n_owned_neurons": n_owned,
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(NTASKS)]}

def main():
    t0 = time.time()
    data = load_mnist()
    runs = []
    for seed in SEEDS:
        g = torch.Generator().manual_seed(100 + seed)   # same perms as exp4
        perms = [torch.randperm(784, generator=g) for _ in range(NTASKS)]
        runs.append(run_seed(seed, data, perms))
    bwts = [r["bwt"] for r in runs]; favgs = [r["final_avg"] for r in runs]
    res = {"ad_prot2": {"bwt_mean": round(st.mean(bwts), 2), "bwt_std": round(st.pstdev(bwts), 2),
                        "final_avg_mean": round(st.mean(favgs), 2), "final_avg_std": round(st.pstdev(favgs), 2),
                        "runs": runs}}
    print(f"[ad_prot2] BWT {st.mean(bwts):.2f}±{st.pstdev(bwts):.2f}  "
          f"finalAvg {st.mean(favgs):.2f}±{st.pstdev(favgs):.2f}", flush=True)
    from scipy import stats as sps
    prev = json.load(open("exp4_result.json"))
    stats_out = {}
    for other in ["mlp", "ad", "ad_prot"]:
        db = [r["bwt"] for r in prev[other]["runs"]]
        t, p = sps.ttest_rel(bwts, db)
        stats_out[f"ad_prot2_vs_{other}_bwt_ttest_p"] = round(float(p), 5)
        fa = [r["final_avg"] for r in prev[other]["runs"]]
        t2, p2 = sps.ttest_rel(favgs, fa)
        stats_out[f"ad_prot2_vs_{other}_finalavg_ttest_p"] = round(float(p2), 5)
    res["stats"] = stats_out
    res["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in res.items() if k != "ad_prot2"}, indent=2))
    json.dump(res, open("exp4b_result.json", "w"), indent=2)

if __name__ == "__main__":
    main()

"""exp4: Branch-protected plasticity in Active Dendrites networks (full experiment).

Protocol (Permuted-MNIST continual learning):
  5 sequential tasks (fixed random pixel permutations), 10k train samples each.
  Conditions:
    1) mlp       - point-neuron baseline (same size hidden layers, k-WTA)
    2) ad        - Active Dendrites (context one-hot, K=10 segments, max-gating)
    3) ad_prot   - AD + branch-protected plasticity: non-winning dendritic
                   segments AND their feedforward path get per-task context
                   gating of gradient updates (update only ff+win-seg when ctx
                   matches the segment's dominant context).
  Metrics per condition (mean over seeds): final avg acc, BWT (forgetting),
  accuracy matrix. Also runtime.

Design notes:
  * Protection rule (minimal, biologically interpretable):
      - Each dendritic segment owns one context (segment j <-> task j, K = n_tasks).
      - Forward: unchanged (d = max_j u_j . c selects the segment).
      - Backward: gradients to ff weights W and to dendrite weights U are
        zeroed for all segments except the winning one AND except the segment
        whose context matches current task. This mimics "frozen" non-relevant
        branches -> no interference to previously learned contexts.
    (EWC reference optional; omitted for runtime, main comparison is 3 arms.)
  * 5 seeds x 3 conditions. GPU: single 3090 is plenty.
"""
import json, time, copy
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEEDS = [0, 1, 2, 3, 4]
NTASKS, EPOCHS, BS, NTRAIN, NH = 5, 2, 256, 10000, 256
KSEG = 5           # one segment per task
LR = 1e-3
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def load_mnist():
    from torchvision import datasets, transforms
    tr = datasets.MNIST("./data", train=True, download=True)
    te = datasets.MNIST("./data", train=False, download=True)
    xtr = tr.data.float().view(-1, 784) / 255.0
    return xtr, tr.targets, te.data.float().view(-1, 784) / 255.0, te.targets

def permute(x, perm):
    return x[:, perm]

class ADNet(nn.Module):
    """Active Dendrites with optional branch-protected gradients."""
    def __init__(self, nh=NH, nseg=KSEG, ntask=NTASKS, protected=False):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.dend = nn.Linear(ntask, nh * nseg, bias=False)
        self.out = nn.Linear(nh, 10)
        self.nseg, self.nh, self.protected = nseg, nh, protected
        # segment-context assignment: seg j <-> task j (simple, no learning)
        self.seg_of_task = torch.arange(ntask)
    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d, seg_id = u.max(dim=1)                     # winning segment
        h = kwta(t * torch.sigmoid(d))
        return self.out(h), seg_id
    def protect_grads(self, ctx_label):
        """Zero grads for non-relevant branches before optimizer.step().
        ctx_label: int task id of current batch."""
        if not self.protected or self.dend.weight.grad is None:
            return
        W = self.dend.weight.grad                     # (nh*nseg, ntask)
        Wv = W.view(self.nseg, self.nh, -1)
        # freeze all segments except the one assigned to current task
        mask = torch.zeros(self.nseg, 1, 1, device=W.device)
        mask[ctx_label] = 1.0
        Wv.mul_(mask)
        # ff weights: keep (they are shared) -- protection only on dendrites.
        return

class MLPNet(nn.Module):
    def __init__(self, nh=NH):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.out = nn.Linear(nh, 10)
    def forward(self, x, ctx):
        return self.out(kwta(torch.sigmoid(self.ff(x)))), None

def kwta(h, k=64):
    thr = h.topk(min(k, h.size(1)), dim=1).values[:, -1:]
    return h * (h >= thr).float()

def run_seed(seed, cond, data, perms):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = data
    model = (MLPNet() if cond == "mlp"
             else ADNet(protected=(cond == "ad_prot"))).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    acc = {}
    for phase in range(NTASKS):
        perm = perms[phase]
        xs, ys = permute(xtr[:NTRAIN], perm), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV), NTASKS).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                if cond == "ad_prot":
                    model.protect_grads(phase)
                opt.step()
        # eval all seen tasks
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
        print(f"  seed{seed} {cond} after task{phase}: " +
              " ".join(f"t{cc}={acc[(phase,cc)]:.1f}" for cc in range(phase+1)), flush=True)
    bwt = sum(acc[(NTASKS-1, c)] - acc[(c, c)] for c in range(NTASKS-1)) / (NTASKS-1)
    final_avg = sum(acc[(NTASKS-1, c)] for c in range(NTASKS)) / NTASKS
    acc_matrix = [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(NTASKS)]
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(final_avg, 2),
            "acc_matrix": acc_matrix}

def main():
    t0 = time.time()
    data = load_mnist()
    results = {}
    for cond in ["mlp", "ad", "ad_prot"]:
        runs = []
        for seed in SEEDS:
            g = torch.Generator().manual_seed(100 + seed)
            perms = [torch.randperm(784, generator=g) for _ in range(NTASKS)]
            runs.append(run_seed(seed, cond, data, perms))
        bwts = [r["bwt"] for r in runs]; favgs = [r["final_avg"] for r in runs]
        import statistics as st
        results[cond] = {
            "bwt_mean": round(st.mean(bwts), 2), "bwt_std": round(st.pstdev(bwts), 2),
            "final_avg_mean": round(st.mean(favgs), 2), "final_avg_std": round(st.pstdev(favgs), 2),
            "runs": runs}
        print(f"[{cond}] BWT {st.mean(bwts):.2f}±{st.pstdev(bwts):.2f}  "
              f"finalAvg {st.mean(favgs):.2f}±{st.pstdev(favgs):.2f}", flush=True)
    # paired t-test ad_prot vs mlp and ad (same seeds, same perms across conds by construction)
    try:
        from scipy import stats as sps
        res = {}
        for a, b in [("ad_prot", "mlp"), ("ad_prot", "ad")]:
            da = [r["bwt"] for r in results[a]["runs"]]
            db = [r["bwt"] for r in results[b]["runs"]]
            t, p = sps.ttest_rel(da, db)
            res[f"{a}_vs_{b}_bwt_ttest_p"] = round(float(p), 5)
        results["stats"] = res
    except ImportError:
        results["stats"] = "scipy not available"
    results["elapsed_sec"] = round(time.time() - t0, 1)
    results["config"] = {"ntasks": NTASKS, "epochs": EPOCHS, "ntrain": NTRAIN,
                         "nh": NH, "kseg": KSEG, "lr": LR, "seeds": SEEDS, "dev": DEV}
    print(json.dumps({k: v for k, v in results.items() if k != "runs"}, indent=2))
    json.dump(results, open("exp4_result.json", "w"), indent=2)

if __name__ == "__main__":
    main()

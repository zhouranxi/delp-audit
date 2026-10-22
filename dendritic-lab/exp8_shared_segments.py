"""exp8: shared dendritic segment pool (task-scalable AD).
Diagnosis from exp6/exp7: with nseg=ntask, per-segment capacity shrinks as
tasks grow (10 tasks -> 10 segments -> few neurons each) => underfitting of
later tasks (t9 train acc ~21%), NOT forgetting. Fix: segments are a SHARED
pool (KSEG fixed, e.g. 5); context selects a COMBINATION of segments via
top-k gating, so features are reused across tasks. Evaluated on 10T with
lam=10000, nh=512, plus oracle-context and inferred-context variants.
Also includes our final 5T check with same arch (nseg=5, lam=5000).
"""
import json, time, statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from exp6_baselines import (SEEDS, EPOCHS, BS, NTRAIN, LR, DEV, load_mnist,
                            permute, kwta, get_perms, eval_model, finish,
                            summarize)
from exp4_branch_protection import ADNet

TOPK = 3          # segments active per context
NSEG = 8          # shared pool size (independent of ntasks)
REPLAY = 500      # samples per old task for context inference
NH10 = 512

class ADNetShared(nn.Module):
    """Shared segment pool; context -> top-k segment combination (fixed map)."""
    def __init__(self, nh, nseg=NSEG, ntask=10):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.dend = nn.Linear(ntask, nh * nseg, bias=False)
        self.out = nn.Linear(nh, 10)
        self.nseg, self.nh, self.ntask = nseg, nh, ntask
        # fixed pseudo-random context->segment affinity (hash-like, shared use)
        g = torch.Generator().manual_seed(7)
        self.register_buffer("affinity", (torch.rand(ntask, nseg, generator=g) * 2 - 1))
    def seg_weights(self, ctx):
        """ctx (B, ntask) -> (B, nseg) gate in [0,1]: top-k of affinity rows."""
        a = ctx @ self.affinity                    # (B, nseg)
        k = TOPK
        thr = a.topk(k, dim=1).values[:, -1:]
        gate = torch.sigmoid((a - thr) * 8)        # soft top-k
        return gate
    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        gate = self.seg_weights(ctx).unsqueeze(-1)  # (B, nseg, 1)
        d = (u * gate).max(dim=1).values
        h = kwta(t * torch.sigmoid(d))
        return self.out(h), None
    def protect_grads(self, ctx_label):
        """Freeze segments not gated-on by current context (rows with gate~0)."""
        if self.dend.weight.grad is None:
            return
        with torch.no_grad():
            c = F.one_hot(torch.tensor([ctx_label], device=DEV), self.ntask).float()
            gate = self.seg_weights(c).mean(0)      # (nseg,)
        W = self.dend.weight.grad.view(self.nseg, self.nh, -1)
        W.mul_((gate > 0.5).float().view(-1, 1, 1))

def fisher_usage_shared(model, xs, task_id, ntask):
    model.eval()
    P = {"ff.w": model.ff.weight, "ff.b": model.ff.bias,
         "out.w": model.out.weight, "out.b": model.out.bias}
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    ctx1 = F.one_hot(torch.tensor([task_id], device=DEV), ntask).float()
    o = torch.zeros(model.nh, device=DEV)
    ns, nh = model.nseg, model.nh
    with torch.no_grad():
        for i in range(0, 500, 250):
            xb = xs[i:i+250].to(DEV)
            ctx = ctx1.expand(xb.size(0), -1)
            t = model.ff(xb)
            u = model.dend(ctx).view(-1, ns, nh)
            gate = model.seg_weights(ctx).unsqueeze(-1)
            d = (u * gate).max(dim=1).values
            h = kwta(t * torch.sigmoid(d))
            o += (h > 0).float().sum(0)
    o /= 500
    for i in range(300):
        xb = xs[i:i+1].to(DEV)
        out, _ = model(xb, ctx1)
        model.zero_grad()
        F.log_softmax(out, 1).max(1).values.sum().backward()
        for k, v in P.items():
            fisher[k] += v.grad.detach() ** 2
    for k in fisher:
        fisher[k] /= 300
    fisher["ff.w"] = fisher["ff.w"] * o.unsqueeze(1)
    fisher["ff.b"] = fisher["ff.b"] * o
    fisher["out.w"] = fisher["out.w"] * o.unsqueeze(0)
    return {"theta": {k: v.detach().clone() for k, v in P.items()}, "fisher": fisher}

def train_ours(seed, data, perms, ntask, lam, nh, infer_ctx=False):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)          # reproducibility: init was ambient-RNG dependent
    model = ADNetShared(nh, ntask=ntask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    P = {"ff.w": model.ff.weight, "ff.b": model.ff.bias,
         "out.w": model.out.weight, "out.b": model.out.bias}
    anchors, acc = [], {}
    replay = []           # (x, task_id)
    ctx_clf = None        # logistic regression on model.ff features
    def infer_task(xb):
        with torch.no_grad():
            feat = model.ff(xb)
        logits = ctx_clf(feat)
        return logits  # (B, ntask) soft one-hot posterior
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
        anchors.append(fisher_usage_shared(model, xs, phase, ntask))
        replay.append((permute(xtr[:REPLAY], perms[phase]).to(DEV), phase))
        # train context classifier on replay buffer (task-inference head)
        Xr = torch.cat([x for x, _ in replay])
        Yr = torch.cat([torch.full((x.size(0),), t, device=DEV) for x, t in replay])
        ctx_clf = nn.Linear(P["ff.w"].shape[0], ntask).to(DEV)
        copt = torch.optim.Adam(ctx_clf.parameters(), lr=1e-2)
        for _ in range(300):
            copt.zero_grad()
            F.cross_entropy(ctx_clf(model.ff(Xr)), Yr).backward()
            copt.step()
        # eval
        model.eval()
        with torch.no_grad():
            for cc in range(phase + 1):
                xt = permute(xte, perms[cc]).to(DEV)
                correct = 0
                correct_inf = 0
                for i in range(0, xt.size(0), 2000):
                    n = min(2000, xt.size(0) - i)
                    xb = xt[i:i+2000]
                    ctx = F.one_hot(torch.full((n,), cc, device=DEV), ntask).float()
                    out, _ = model(xb, ctx)
                    correct += (out.argmax(1) == yte[i:i+2000].to(DEV)).sum().item()
                    if infer_ctx:
                        post = ctx_clf(model.ff(xb))          # (n, ntask)
                        # sample-free: use argmax one-hot (hard) context
                        hard = F.one_hot(post.argmax(1), ntask).float()
                        out2, _ = model(xb, hard)
                        correct_inf += (out2.argmax(1) == yte[i:i+2000].to(DEV)).sum().item()
                acc[(phase, cc)] = correct / xt.size(0) * 100
                if infer_ctx:
                    acc[("inf", phase, cc)] = correct_inf / xt.size(0) * 100
        tail = (f" inf{cc}={acc[('inf', phase, cc)]:.0f}" if infer_ctx else "")
        print(f"  s{seed} shared T{phase}: " + " ".join(f"t{c}={acc[(phase,c)]:.0f}" for c in range(phase+1)) + tail, flush=True)
    bwt, favg = finish(acc, ntask)
    out = {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
           "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(ntask)]}
    if infer_ctx:
        inf_favg = sum(acc[("inf", ntask-1, c)] for c in range(ntask)) / ntask
        out["final_avg_inferred"] = round(inf_favg, 2)
    return out

def main():
    t0 = time.time()
    data = load_mnist()
    results = {}
    # 10T: oracle context
    runs = [train_ours(s, data, get_perms(s, 10), 10, 10000.0, NH10, infer_ctx=False) for s in SEEDS]
    results["10T_shared_oracle"] = {**summarize(runs), "runs": runs}
    print(f"[10T shared oracle] BWT {results['10T_shared_oracle']['bwt_mean']} "
          f"final {results['10T_shared_oracle']['final_avg_mean']}", flush=True)
    # 10T: inferred context (2 seeds to save time — ablation trend)
    runs2 = [train_ours(s, data, get_perms(s, 10), 10, 10000.0, NH10, infer_ctx=True) for s in [0, 1]]
    results["10T_shared_inferred"] = {**summarize(runs2), "runs": runs2}
    oracle2 = [r for r in runs if r["seed"] in [0, 1]]
    drops = [round(r1["final_avg_inferred"] - r1["final_avg"], 2) for r1 in runs2]
    results["10T_shared_inferred"]["inferred_final_avg"] = [r["final_avg_inferred"] for r in runs2]
    results["10T_shared_inferred"]["drop_vs_oracle"] = drops
    # 5T regression check with shared arch (nseg=8, topk=3, lam=5000, nh=256)
    runs3 = [train_ours(s, data, get_perms(s, 5), 5, 5000.0, 256, infer_ctx=False) for s in SEEDS]
    results["5T_shared"] = {**summarize(runs3), "runs": runs3}
    print(f"[5T shared] BWT {results['5T_shared']['bwt_mean']} final {results['5T_shared']['final_avg_mean']}", flush=True)
    results["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(results, open("exp8_result.json", "w"), indent=2, default=str)
    print("SAVED exp8_result.json")

if __name__ == "__main__":
    main()

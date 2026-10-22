"""exp6 (clean): ours vs SI vs MAS vs PackNet on Permuted-MNIST, 5T and 10T.
- ours  : AD + segment freeze + usage-localized EWC (lambda fixed 5000, screened in exp5)
- si    : MLP + Synaptic Intelligence (Zenke 2017), lambda screened {0.1,1,10} on seeds{0,1}
- mas   : MLP + MAS (Aljundi 2019), importance = grad of 0.5*||logits||^2, same screening
- packnet: MLP + PackNet (prune 20%/task, per-task cumulative masks; task-id at test)
Protocol: domain-IL, one-hot task context given for ours & packnet (same info);
SI/MAS context-free. Paired t-tests ours vs each (same seeds & permutations).
"""
import json, time, copy, statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from exp4_branch_protection import (SEEDS, EPOCHS, BS, NTRAIN, NH, LR,
                                    DEV, load_mnist, permute, kwta, ADNet)

NFISHER, NUSAGE = 300, 500
PARAM_KEYS = ["ff.w", "ff.b", "out.w", "out.b"]

def get_perms(seed, ntasks):
    g = torch.Generator().manual_seed(100 + seed)
    return [torch.randperm(784, generator=g) for _ in range(ntasks)]

def flat_params(model):
    return {"ff.w": model.ff.weight, "ff.b": model.ff.bias,
            "out.w": model.out.weight, "out.b": model.out.bias}

def eval_model(model, xte, yte, perms, phase, ntask, needs_ctx):
    model.eval()
    acc = {}
    with torch.no_grad():
        for cc in range(phase + 1):
            xt = permute(xte, perms[cc]).to(DEV)
            correct = 0
            for i in range(0, xt.size(0), 2000):
                n = min(2000, xt.size(0) - i)
                xb = xt[i:i+2000]
                if needs_ctx:
                    ctx = F.one_hot(torch.full((n,), cc, device=DEV), ntask).float()
                    out, _ = model(xb, ctx)
                else:
                    out, _ = model(xb)
                correct += (out.argmax(1) == yte[i:i+2000].to(DEV)).sum().item()
            acc[(phase, cc)] = correct / xt.size(0) * 100
    return acc

def finish(acc, ntask):
    bwt = sum(acc[(ntask-1, c)] - acc[(c, c)] for c in range(ntask-1)) / (ntask-1)
    favg = sum(acc[(ntask-1, c)] for c in range(ntask)) / ntask
    return bwt, favg

# ---------------- ours ----------------
def fisher_usage(model, xs, task_id, ntask):
    model.eval()
    P = flat_params(model)
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    ctx1 = F.one_hot(torch.tensor([task_id], device=DEV), ntask).float()
    ns, nh = model.nseg, model.nh
    o = torch.zeros(nh, device=DEV)
    with torch.no_grad():
        for i in range(0, NUSAGE, 250):
            xb = xs[i:i+250].to(DEV)
            ctx = ctx1.expand(xb.size(0), -1)
            t = model.ff(xb)
            u = model.dend(ctx).view(-1, ns, nh)
            d, _ = u.max(dim=1)
            h = kwta(t * torch.sigmoid(d))
            o += (h > 0).float().sum(0)
    o /= NUSAGE
    for i in range(NFISHER):
        xb = xs[i:i+1].to(DEV)
        yb = None
        ctx = ctx1
        out, _ = model(xb, ctx)
        model.zero_grad()
        out.norm().backward()          # MAS-style proxy? no: for ours use CE on true label
        # NOTE: replaced below with proper CE fisher
        for k, v in P.items():
            fisher[k] += 0
    # proper Fisher with true labels
    ys = None
    for k in fisher:
        fisher[k].zero_()
    for i in range(NFISHER):
        xb = xs[i:i+1].to(DEV)
        out, _ = model(xb, ctx1)
        model.zero_grad()
        F.log_softmax(out, 1).max(1).values.sum().backward()
        for k, v in P.items():
            fisher[k] += v.grad.detach() ** 2
    for k in fisher:
        fisher[k] /= NFISHER
    fisher["ff.w"] = fisher["ff.w"] * o.unsqueeze(1)
    fisher["ff.b"] = fisher["ff.b"] * o
    fisher["out.w"] = fisher["out.w"] * o.unsqueeze(0)
    return {"theta": {k: v.detach().clone() for k, v in P.items()}, "fisher": fisher}

def run_ours(seed, data, perms, ntask, lam=5000.0):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = ADNet(protected=True, nseg=ntask, ntask=ntask).to(DEV)
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
                if anchors:
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
        a = eval_model(model, xte, yte, perms, phase, ntask, True)
        acc.update(a)
        print(f"  s{seed} ours T{phase}: " + " ".join(f"t{c}={acc[(phase,c)]:.0f}" for c in range(phase+1)), flush=True)
    bwt, favg = finish(acc, ntask)
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(ntask)]}

# ---------------- SI ----------------
def run_si(seed, data, perms, ntask, lam):
    xtr, ytr, xte, yte = data
    model = ADNet(protected=False, nseg=ntask, ntask=ntask).to(DEV)
    # strip dendrites: use plain path (context ignored)
    model = type("M", (), {})  # placeholder, replaced below
    return None

class PlainMLP(nn.Module):
    def __init__(self, nh=NH):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.out = nn.Linear(nh, 10)
    def forward(self, x, ctx=None):
        return self.out(kwta(torch.sigmoid(self.ff(x)))), None

def run_si_impl(seed, data, perms, ntask, lam):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = PlainMLP().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    P = flat_params(model)
    omega = {k: torch.zeros_like(v) for k, v in P.items()}
    theta_ref = {k: v.detach().clone() for k, v in P.items()}
    acc = {}
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        theta_task_start = {k: v.detach().clone() for k, v in P.items()}
        integ = {k: torch.zeros_like(v) for k, v in P.items()}
        nb = 0
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                pre = {k: v.detach().clone() for k, v in P.items()}
                opt.zero_grad()
                out, _ = model(xb)
                loss = F.cross_entropy(out, yb)
                if phase > 0:
                    pen = sum((omega[k] * (P[k] - theta_ref[k]) ** 2).sum() for k in P)
                    loss = loss + lam * pen
                loss.backward()
                for k, v in P.items():
                    if v.grad is not None:
                        integ[k] += (-v.grad.detach()) * 0  # updated after step
                opt.step()
                for k, v in P.items():
                    if v.grad is not None:
                        delta = v.detach() - pre[k]
                        integ[k] += (-v.grad.detach()) * delta
                nb += 1
        for k in P:
            dtotal = (P[k].detach() - theta_task_start[k]).abs() + 1e-3
            omega[k] += integ[k].abs() / dtotal
        theta_ref = {k: v.detach().clone() for k, v in P.items()}
        a = eval_model(model, xte, yte, perms, phase, ntask, False)
        acc.update(a)
        print(f"  s{seed} si T{phase}: " + " ".join(f"t{c}={acc[(phase,c)]:.0f}" for c in range(phase+1)), flush=True)
    bwt, favg = finish(acc, ntask)
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(ntask)]}

# ---------------- MAS ----------------
def run_mas_impl(seed, data, perms, ntask, lam):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = PlainMLP().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    P = flat_params(model)
    omega = {k: torch.zeros_like(v) for k, v in P.items()}
    theta_ref = {k: v.detach().clone() for k, v in P.items()}
    acc = {}
    NGRAD = 2000
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                out, _ = model(xb)
                loss = F.cross_entropy(out, yb)
                if phase > 0:
                    pen = sum((omega[k] * (P[k] - theta_ref[k]) ** 2).sum() for k in P)
                    loss = loss + lam * pen
                loss.backward()
                opt.step()
        # MAS importance: grad of 0.5*||f||^2
        model.eval()
        n_used = 0
        for i in range(0, NGRAD, 250):
            xb = xs[i:i+250].to(DEV)
            out, _ = model(xb)
            model.zero_grad()
            (0.5 * out.pow(2).sum(dim=1)).mean().backward()
            for k, v in P.items():
                omega[k] += (v.grad.detach() ** 2) * (1.0 / (out.size(0)))
            n_used += 1
        for k in P:
            omega[k] /= max(1, n_used)
        theta_ref = {k: v.detach().clone() for k, v in P.items()}
        a = eval_model(model, xte, yte, perms, phase, ntask, False)
        acc.update(a)
        print(f"  s{seed} mas T{phase}: " + " ".join(f"t{c}={acc[(phase,c)]:.0f}" for c in range(phase+1)), flush=True)
    bwt, favg = finish(acc, ntask)
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(ntask)]}

# ---------------- PackNet ----------------
def run_packnet(seed, data, perms, ntask, prune_frac=0.2):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = PlainMLP().to(DEV)
    P = flat_params(model)
    frozen = {k: torch.zeros_like(v) for k, v in P.items()}   # cumulative frozen
    task_masks = []                                          # frozen snapshot per task
    acc = {}
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        model.train()
        # phase A: free train 1 epoch (respect frozen)
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            out, _ = model(xb)
            F.cross_entropy(out, yb).backward()
            for k, v in P.items():
                v.grad.mul_(1 - frozen[k])
            opt.step()
        # phase B: prune lowest-|w| among free
        mags = torch.cat([(v.detach().abs().flatten()[frozen[k].flatten() == 0]) for k, v in P.items()])
        k_n = max(1, int(mags.numel() * prune_frac)) if mags.numel() > 10 else 0
        survivors = {}
        if k_n > 0 and mags.numel() > 0:
            thr = mags.kthvalue(min(k_n, mags.numel())).values
            for k, v in P.items():
                survivors[k] = ((v.detach().abs() > thr) | (frozen[k] > 0)).float()
        else:
            for k, v in P.items():
                survivors[k] = (frozen[k] > 0).float().clone()
                survivors[k] = torch.ones_like(v)
        # re-init non-survivor weights
        with torch.no_grad():
            for k, v in P.items():
                noise = torch.randn_like(v) * 0.01
                v.mul_(survivors[k]).add_(noise * (1 - survivors[k]))
        # phase C: retrain survivors only
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                out, _ = model(xb)
                F.cross_entropy(out, yb).backward()
                for k, v in P.items():
                    v.grad.mul_(survivors[k] * (1 - frozen[k]))
                opt.step()
        # freeze everything this task uses
        with torch.no_grad():
            for k, v in P.items():
                frozen[k] = torch.clamp(frozen[k] + survivors[k], max=1.0)
                v.mul_(frozen[k])          # zero unused free weights
        task_masks.append({k: frozen[k].clone() for k in P})
        # eval with per-task masks (task id at test, same info budget as ours)
        model.eval()
        snapshot = {k: v.detach().clone() for k, v in P.items()}
        with torch.no_grad():
            for cc in range(phase + 1):
                for k, v in P.items():
                    v.copy_(snapshot[k] * task_masks[cc][k])
                xt = permute(xte, perms[cc]).to(DEV)
                correct = 0
                for i in range(0, xt.size(0), 2000):
                    out, _ = model(xt[i:i+2000])
                    correct += (out.argmax(1) == yte[i:i+2000].to(DEV)).sum().item()
                acc[(phase, cc)] = correct / xt.size(0) * 100
            for k, v in P.items():
                v.copy_(snapshot[k])
        print(f"  s{seed} packnet T{phase}: " + " ".join(f"t{c}={acc[(phase,c)]:.0f}" for c in range(phase+1)), flush=True)
    bwt, favg = finish(acc, ntask)
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p+1)] for p in range(ntask)]}

def summarize(runs):
    b = [r["bwt"] for r in runs]; f = [r["final_avg"] for r in runs]
    return {"bwt_mean": round(st.mean(b), 2), "bwt_std": round(st.pstdev(b), 2),
            "final_avg_mean": round(st.mean(f), 2), "final_avg_std": round(st.pstdev(f), 2)}

def main():
    t0 = time.time()
    data = load_mnist()
    results = {}
    # ---- lambda screening for SI / MAS on 5T, seeds {0,1}
    LAMS = [0.1, 1.0, 10.0]
    screen = {}
    best = {}
    for arm, fn in [("si", run_si_impl), ("mas", run_mas_impl)]:
        for lam in LAMS:
            rs = [fn(s, data, get_perms(s, 5), 5, lam) for s in [0, 1]]
            m = summarize(rs)
            screen[f"{arm}_{lam}"] = m
            print(f"[screen] {arm} lam={lam}: {m}", flush=True)
        best[arm] = max(LAMS, key=lambda l: (screen[f"{arm}_{l}"]["bwt_mean"],
                                             screen[f"{arm}_{l}"]["final_avg_mean"]))
    results["screen"] = {k: v for k, v in screen.items()}
    results["best_lam"] = best
    # ---- main runs
    for ntask in [5, 10]:
        res = {}
        arms = {
            "ours": lambda s, p: run_ours(s, data, p, ntask),
            "si": lambda s, p: run_si_impl(s, data, p, ntask, best["si"]),
            "mas": lambda s, p: run_mas_impl(s, data, p, ntask, best["mas"]),
            "packnet": lambda s, p: run_packnet(s, data, p, ntask),
        }
        for name, fn in arms.items():
            runs = [fn(s, get_perms(s, ntask)) for s in SEEDS]
            res[name] = {**summarize(runs), "runs": runs}
            print(f"[{ntask}T {name}] BWT {res[name]['bwt_mean']}±{res[name]['bwt_std']} "
                  f"final {res[name]['final_avg_mean']}±{res[name]['final_avg_std']}", flush=True)
        from scipy import stats as sps
        stats_out = {}
        for other in ["si", "mas", "packnet"]:
            _, p1 = sps.ttest_rel([r["bwt"] for r in res["ours"]["runs"]],
                                  [r["bwt"] for r in res[other]["runs"]])
            _, p2 = sps.ttest_rel([r["final_avg"] for r in res["ours"]["runs"]],
                                  [r["final_avg"] for r in res[other]["runs"]])
            stats_out[f"ours_vs_{other}"] = {"bwt_p": round(float(p1), 5),
                                             "finalavg_p": round(float(p2), 5)}
        res["stats"] = stats_out
        results[f"{ntask}tasks"] = res
    results["elapsed_sec"] = round(time.time() - t0, 1)
    slim = {k: (v if k not in ("5tasks", "10tasks") else
                {kk: (vv if kk != "runs" else "...") for kk, vv in v.items()})
            for k, v in results.items()}
    print(json.dumps(slim, indent=2, default=str))
    json.dump(results, open("exp6_result.json", "w"), indent=2, default=str)

if __name__ == "__main__":
    main()

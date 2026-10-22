"""exp10: capacity analysis of the 10-task regime.

Question: at 10 tasks DeLP keeps forgetting controlled (BWT ~ -3.4) yet final
accuracy plateaus ~54% while SI reaches 63%. Is this a protection problem or a
representation-capacity problem?

Decomposition used throughout:
    final_avg  =  learning_term  +  retention_term
    learning_term  = mean_t a[t][t]                  (accuracy right after learning)
    retention_term = mean_{t<T} (a[T][t] - a[t][t])  (what survives afterwards)

Probes
  A  learning/retention decomposition vs width N in {256,512,1024} for DeLP and
     for the point MLP (MLP at 256 and 1024 as a capacity-matched control).
  B  single-task ceiling: train a fresh network on each permuted task alone.
     If the ceiling is high but in-sequence a[t][t] collapses, the sequence --
     not task difficulty -- is what starves the task.
  C  neuron budget: per-task fraction of neurons inside the kWTA winner set,
     and the oversubscription ratio T * fraction(mean) / 1, plus pairwise
     Jaccard overlap of per-task winner sets.

Outputs exp10_result.json. Runs on one GPU, ~5-10 min.
"""
import json, time, statistics as st
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from exp6_baselines import (SEEDS, EPOCHS, BS, NTRAIN, LR, DEV, load_mnist,
                            permute, get_perms, eval_model, finish,
                            summarize, flat_params, kwta, PlainMLP)
from exp4_branch_protection import ADNet

NTASK = 10
NFISHER, NUSAGE = 300, 500
WIDTHS = [256, 512, 1024]


def fisher_usage(model, xs, task_id, ntask):
    """Same estimator as exp6 (label-free max-logprob Fisher), plus usage vector."""
    model.eval()
    P = flat_params(model)
    ctx1 = F.one_hot(torch.tensor([task_id], device=DEV), ntask).float()
    ns, nh = model.nseg, model.nh
    o = torch.zeros(nh, device=DEV)
    with torch.no_grad():
        for i in range(0, NUSAGE, 250):
            xb = xs[i:i + 250].to(DEV)
            ctx = ctx1.expand(xb.size(0), -1)
            t = model.ff(xb)
            u = model.dend(ctx).view(-1, ns, nh)
            d, _ = u.max(dim=1)
            h = kwta(t * torch.sigmoid(d))
            o += (h > 0).float().sum(0)
    o /= NUSAGE
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    for i in range(NFISHER):
        xb = xs[i:i + 1].to(DEV)
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
    return {"theta": {k: v.detach().clone() for k, v in P.items()},
            "fisher": fisher, "usage": o.clone()}


def run_delp(seed, data, perms, ntask, nh, lam):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = ADNet(nh=nh, protected=True, nseg=ntask, ntask=ntask).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc, usage = [], {}, []
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
        anc = fisher_usage(model, xs, phase, ntask)
        anchors.append(anc)
        usage.append(anc["usage"].cpu())
        acc.update(eval_model(model, xte, yte, perms, phase, ntask, True))
        print(f"  s{seed} delp{nh} T{phase}: t{phase}={acc[(phase,phase)]:.0f}", flush=True)
    bwt, favg = finish(acc, ntask)
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "justlearn": [round(acc[(t, t)], 2) for t in range(ntask)],
            "usage": [round(float(u.mean()), 4) for u in usage],
            "usage_vecs": [u.tolist() for u in usage],
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)] for p in range(ntask)]}


def run_mlp(seed, data, perms, ntask, nh):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = PlainMLP(nh=nh).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    acc, usage = {}, []
    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                out, _ = model(xb)
                F.cross_entropy(out, yb).backward()
                opt.step()
        model.eval()
        o = torch.zeros(nh, device=DEV)
        with torch.no_grad():
            for i in range(0, NUSAGE, 250):
                xb = xs[i:i + 250].to(DEV)
                o += (kwta(torch.sigmoid(model.ff(xb))) > 0).float().sum(0)
        usage.append((o / NUSAGE).cpu())
        acc.update(eval_model(model, xte, yte, perms, phase, ntask, False))
        print(f"  s{seed} mlp{nh} T{phase}: t{phase}={acc[(phase,phase)]:.0f}", flush=True)
    bwt, favg = finish(acc, ntask)
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "justlearn": [round(acc[(t, t)], 2) for t in range(ntask)],
            "usage": [round(float(u.mean()), 4) for u in usage],
            "usage_vecs": [u.tolist() for u in usage],
            "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)] for p in range(ntask)]}


def single_task_ceiling(seeds, data, perms, ntask, nh, kind):
    """Train a fresh net on ONE permuted task, no sequence, no penalty."""
    xtr, ytr, xte, yte = data
    out = {}
    for t in range(ntask):
        vals = []
        for s in seeds:
            torch.manual_seed(1000 + s)
            if kind == "mlp":
                model = PlainMLP(nh=nh).to(DEV)
                needs_ctx = False
            else:
                model = ADNet(nh=nh, protected=False, nseg=ntask, ntask=ntask).to(DEV)
                needs_ctx = True
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            xs, ys = permute(xtr[:NTRAIN], perms[t]), ytr[:NTRAIN]
            dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
            model.train()
            for ep in range(EPOCHS):
                for xb, yb in dl:
                    xb, yb = xb.to(DEV), yb.to(DEV)
                    opt.zero_grad()
                    if needs_ctx:
                        ctx = F.one_hot(torch.full((xb.size(0),), t, device=DEV), ntask).float()
                        o_, _ = model(xb, ctx)
                    else:
                        o_, _ = model(xb)
                    F.cross_entropy(o_, yb).backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                xt = permute(xte, perms[t]).to(DEV)
                corr = 0
                for i in range(0, xt.size(0), 2000):
                    n = min(2000, xt.size(0) - i)
                    if needs_ctx:
                        ctx = F.one_hot(torch.full((n,), t, device=DEV), ntask).float()
                        o_, _ = model(xt[i:i + 2000], ctx)
                    else:
                        o_, _ = model(xt[i:i + 2000])
                    corr += (o_.argmax(1) == yte[i:i + n].to(DEV)).sum().item()
                vals.append(corr / xt.size(0) * 100)
        out[t] = round(st.mean(vals), 2)
    return out


def overlap_stats(usage_vecs, thresh=0.5):
    """Pairwise Jaccard overlap of per-task winner sets (neurons with usage>thresh)."""
    sets = [(torch.as_tensor(u, dtype=torch.float32) > thresh).float() for u in usage_vecs]
    n = len(sets)
    js = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = (sets[i] * sets[j]).sum().item()
            union = ((sets[i] + sets[j]) > 0).float().sum().item()
            if union > 0:
                js.append(inter / union)
    return round(st.mean(js), 4) if js else 0.0


def main():
    t0 = time.time()
    data = load_mnist()
    perms = get_perms(0, NTASK)      # one permutation set, all arms (paired design)
    res = {"protocol": {"ntask": NTASK, "epochs": EPOCHS, "ntrain": NTRAIN,
                        "widths": WIDTHS, "seeds": SEEDS}, "runs": {}}

    arms = ([("mlp256", lambda s: run_mlp(s, data, perms, NTASK, 256)),
             ("mlp1024", lambda s: run_mlp(s, data, perms, NTASK, 1024))]
            + [(f"delp{w}", (lambda s, w=w: run_delp(s, data, perms, NTASK, w, 10000.0)))
               for w in WIDTHS])

    for name, fn in arms:
        runs = [fn(s) for s in SEEDS]
        agg = summarize(runs)
        just = [round(st.mean([r["justlearn"][t] for r in runs]), 2) for t in range(NTASK)]
        justsd = [round(st.pstdev([r["justlearn"][t] for r in runs]), 2) for t in range(NTASK)]
        learn_term = round(st.mean(just), 2)
        usage_mean = [round(st.mean([r["usage"][t] for r in runs]), 4) for t in range(NTASK)]
        jac = [overlap_stats(r["usage_vecs"]) for r in runs]
        res["runs"][name] = {
            **agg,
            "justlearn": just, "justlearn_sd": justsd,
            "learning_term": learn_term,
            "retention_term": round(agg["bwt_mean"], 2),
            "usage_mean": usage_mean,
            "pairwise_jaccard": round(st.mean(jac), 4),
            "acc_matrix_mean": [[round(st.mean([r["acc_matrix"][p][c] for r in runs]), 2)
                                 for c in range(p + 1)] for p in range(NTASK)],
            "runs": [{k: v for k, v in r.items() if k != "usage_vecs"} for r in runs],
        }
        print(f"[{name}] BWT {agg['bwt_mean']}±{agg['bwt_std']} final {agg['final_avg_mean']}"
              f"±{agg['final_avg_std']} learn_term {learn_term} justlearn {just}", flush=True)

    # single-task ceilings (2 seeds each)
    res["ceiling"] = {
        "delp256": single_task_ceiling([0, 1], data, perms, NTASK, 256, "ad"),
        "delp1024": single_task_ceiling([0, 1], data, perms, NTASK, 1024, "ad"),
        "mlp256": single_task_ceiling([0, 1], data, perms, NTASK, 256, "mlp"),
    }
    for k, v in res["ceiling"].items():
        print(f"[ceiling {k}] {list(v.values())} mean={st.mean(v.values()):.1f}", flush=True)

    # derived: oversubscription ratio = T * mean per-task utilisation of the winner set
    for name in res["runs"]:
        u = res["runs"][name]["usage_mean"]
        res["runs"][name]["oversubscription"] = round(NTASK * st.mean(u), 3)

    res["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(res, open("exp10_result.json", "w"), indent=2, default=str)
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk not in ("runs", "acc_matrix_mean")}
                          if k == "runs" else v) for k, v in res.items()
                      if k in ("elapsed_sec", "ceiling")}, indent=2, default=str))
    for name, v in res["runs"].items():
        print(f"SUMMARY {name}: learn={v['learning_term']} retention={v['retention_term']} "
              f"final={v['final_avg_mean']} oversub={v['oversubscription']} jaccard={v['pairwise_jaccard']}")


if __name__ == "__main__":
    main()

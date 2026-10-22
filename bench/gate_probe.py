"""gate_probe.py -- how far does DeLP's learnable gate actually open?

The ladder ablation found that a plain MLP+EWC with a fixed hidden gain of
alpha ~ 4 beats `ours` by 10-13 points, and that `ours` sits at almost exactly
the same place as alpha = 0.5.  That invites a claim like "DeLP is structurally
pinned at half gain", which is NOT obviously true: the multiplier is
sigma(d) with d a learned projection, so the network could in principle drive
d large and recover alpha -> 1.

So measure it instead of asserting it.  Train `ours` exactly as the ladder does,
then report the distribution of sigma(d) on every task's context.  Three
outcomes are all informative:

  sigma(d) stays ~0.5   -> the gate never opens; the deficit is structural.
  sigma(d) -> 1         -> the gate opens and still loses; the deficit is
                           somewhere else (protection, not scale).
  sigma(d) spreads      -> per-unit gating, needs the per-unit view.
"""
import json
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from exp6_baselines import get_perms, finish, eval_model
import ladder as L


def gate_stats(model, xte, ntask, per_task=2000):
    """sigma(d) for every task's own context, on held-out inputs."""
    ns, nh = model.nseg, model.nh
    xb = xte[:per_task].to(L.DEV)
    rows = []
    with torch.no_grad():
        t = model.ff(xb)
        for c in range(ntask):
            ctx = F.one_hot(torch.tensor([c], device=L.DEV), ntask).float()
            u = model.dend(ctx).view(-1, ns, nh)
            d, seg = u.max(dim=1)                 # same reduction as forward()
            s = torch.sigmoid(d)
            rows.append({
                "task": c,
                # seg is per-unit (the arg-max segment each hidden unit uses),
                # so summarise it rather than pretending it is a scalar.
                "seg_mean": round(float(seg.float().mean()), 3),
                "d_abs_mean": round(float(d.abs().mean()), 4),
                "s_mean": round(float(s.mean()), 4),
                "s_p05": round(float(s.quantile(0.05)), 4),
                "s_p95": round(float(s.quantile(0.95)), 4),
                "frac_s_gt_0.99": round(float((s > 0.99).float().mean()), 4),
                "frac_s_lt_0.51": round(float((s < 0.51).float().mean()), 4),
            })
    return rows


def main():
    ntask = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rung = "ad_usage"                            # == the paper's `ours`

    data = L.load_mnist()
    xtr, ytr, xte, yte = data
    perms = get_perms(seed, ntask)

    torch.manual_seed(seed)
    spec = L.RUNGS[rung]
    model = L.make_model(rung, ntask)
    opt = torch.optim.Adam(model.parameters(), lr=L.LR)
    anchors, acc = [], {}

    gate_by_phase = []
    for phase in range(ntask):
        xs, ys = L.permute(xtr[:L.NTRAIN], perms[phase]), ytr[:L.NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=L.BS, shuffle=True)
        model.train()
        for _ in range(L.EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(L.DEV), yb.to(L.DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=L.DEV),
                                ntask).float()
                opt.zero_grad()
                out, _ = model(xb, ctx)
                loss = F.cross_entropy(out, yb)
                if anchors:
                    P = L.flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        for k, v in P.items():
                            pen = pen + (a["fisher"][k] * (v - a["theta"][k]) ** 2).sum()
                    loss = loss + lam * pen
                loss.backward()
                if spec["protect"]:
                    model.protect_grads(phase)
                opt.step()

        P = L.flat_params(model)
        anchors.append({"fisher": L.fisher_block(model, xs, ys, phase, ntask,
                                                 spec["usage"], spec["fisher"]),
                        "theta": {k: v.detach().clone() for k, v in P.items()}})
        acc.update(eval_model(model, xte, yte, perms, phase, ntask, True))
        model.eval()
        gate_by_phase.append(gate_stats(model, xte, ntask))
        print(f"phase {phase}: " + " ".join(
            f"t{c}={acc[(phase, c)]:.1f}" for c in range(phase + 1))
            + "  s_mean=" + " ".join(f"{r['s_mean']:.3f}" for r in gate_by_phase[-1]),
            flush=True)

    bwt, favg = finish(acc, ntask)
    # the reduction the ladder compares against: which fixed alpha would give the
    # same multiplier on average?
    s_all = [r["s_mean"] for r in gate_by_phase[-1]]
    out = {"rung": rung, "ntask": ntask, "lam": lam, "seed": seed,
           "bwt": round(bwt, 2), "final_avg": round(favg, 2),
           "s_mean_last_phase": round(sum(s_all) / len(s_all), 4),
           "gate_last_phase": gate_by_phase[-1],
           "gate_all_phases": gate_by_phase}
    print(json.dumps({k: out[k] for k in
                      ("ntask", "lam", "bwt", "final_avg", "s_mean_last_phase")},
                     indent=1))
    with open(f"gate_probe_T{ntask}_l{lam:g}_s{seed}.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()

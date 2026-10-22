"""exp6_fixed.py -- quantify the effect of the audited defects in exp6/exp12.

Design rule: ONLY the audited defects are changed. Data, permutations, seeds,
optimiser, schedule, architecture and the evaluation protocol are identical to
exp6_baselines.py, so any difference in the numbers is attributable to the fix
and not to a different experimental setting.

Audited defects addressed (line references are to the original files)

  D1  MAS importance is wrong twice over.                       exp6:220-232
      (a) the objective `(0.5*||f||^2).mean()` is ALREADY batch-averaged, and
          then `omega += grad**2 * (1/B)` divides by the batch size again
          -> Omega ~256x too small.
      (b) `omega` is never reset between tasks, but it is divided by `n_used`
          at the end of EVERY task. After task t the contribution of task 0 has
          been scaled by (1/8)**t, i.e. 7.5e-9 at t=9. So the penalty protects
          only the most recent task, and only weakly.
      Evidence from exp12's own screening: lambda 0.1 -> 10 (100x) moves BWT by
      0.26 points, while the same sweep moves SI by 8.35 and DeLP by 8.50.

  D2  SI deviates from Zenke 2017 in three places.               exp6:183-185
      (a) denominator |dtheta| + 1e-3 instead of dtheta**2 + xi**2 (xi=0.1)
      (b) accumulates |integ| instead of -integ
      (c) xi is 1e-3 instead of 0.1

  D3  PackNet exhausts its capacity by construction.            exp6:242-296
      Fixed prune_frac=0.2 of the FREE weights per task leaves 0.8**10 = 10.7%
      of the network after 10 tasks, and the magnitude threshold is computed
      GLOBALLY across layers whose weight scales differ by orders of magnitude
      (784x256 vs 10x256), so some layer can be pruned to zero.
      Fixed variant: prune per layer, and size the per-task budget so that every
      task gets a fair share of the total capacity.

  D4  EWC (Kirkpatrick 2017) is absent from the table.           exp6:343-348
      Added as `ewc`: diagonal Fisher from true-label CE gradients, accumulated
      across tasks (online EWC).

  D5  lambda is selected on the TEST set.                        exp6:336-337
      `--lam-rule test` reproduces the original (leaky) rule;
      `--lam-rule val`  picks lambda on a held-out split of the training set
      (MNIST train[10000:12000], disjoint from the 10k training subset and from
      the test set), which keeps NTRAIN identical.

Usage
    python exp6_fixed.py --lam-rule val --task-id 0 --n-tasks 4 --out res_fixed.json
"""
import argparse, json, os, statistics as st, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from exp6_baselines import (SEEDS, EPOCHS, BS, NTRAIN, NH, LR, DEV,
                            load_mnist, get_perms, permute, kwta, ADNet,
                            summarize, run_ours, PlainMLP,
                            eval_model, finish, flat_params,
                            run_si_impl, run_mas_impl, run_packnet)

XI = 0.1            # Zenke 2017 default
NGRAD_FIX = 500     # samples used for the per-sample MAS / EWC estimate

# ------------------------------------------------------------------ lambda grids
# The grids must be wide enough to bracket the optimum for BOTH variants: the
# fixed MAS has a ~256x larger Omega, so its optimum sits ~256x lower in lambda.
GRID = {
    "delp":  [100.0, 1000.0, 5000.0],
    "si":    [0.01, 0.1, 1.0, 10.0, 100.0],
    "mas":   [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
    "ewc":   [1e-3, 1e-2, 0.1, 1.0, 10.0, 1000.0],
}
MAIN_ARMS = ["ours", "si_old", "si_new", "mas_old", "mas_new",
             "pack_old", "pack_new", "ewc"]

# Filled in from the sweep before the main stage runs (see --lam-map).
# Defaults are the values exp12 reported; *_new entries are overwritten.
LAM_MAP = {"ours": 5000.0, "si_old": 100.0, "si_new": 100.0,
           "mas_old": 10.0, "mas_new": 100.0, "ewc": 1000.0,
           "pack_old": 0.0, "pack_new": 0.0}


def load_val():
    """Held-out validation split. MNIST train[10000:12000] -- disjoint from the
    10k training subset used by every arm and from the test set."""
    from torchvision import datasets
    tr = datasets.MNIST("./data", train=True, download=True)
    return (tr.data[NTRAIN:NTRAIN + 2000].float().view(-1, 784) / 255.0,
            tr.targets[NTRAIN:NTRAIN + 2000])


# ------------------------------------------------------------------ one arm
def train_arm(base, fixed, seed, data, val, perms, ntask, lam):
    """base in {si, mas, ewc, packnet}; `fixed` toggles the audited repairs."""
    xtr, ytr, xte, yte = data
    vx, vy = val
    torch.manual_seed(seed)
    model = PlainMLP().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Pd = flat_params(model)
    omega = {k: torch.zeros_like(v) for k, v in Pd.items()}
    theta_ref = {k: v.detach().clone() for k, v in Pd.items()}
    frozen = {k: torch.zeros_like(v) for k, v in Pd.items()}
    masks, acc, vacc = [], {}, {}

    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        theta_start = {k: v.detach().clone() for k, v in Pd.items()}
        integ = {k: torch.zeros_like(v) for k, v in Pd.items()}
        model.train()

        if base == "packnet":
            # ---------------- PackNet ----------------
            if fixed:
                # fair share: the remaining free capacity is split evenly over
                # the tasks that still have to be learned (Mallya 2018 spirit)
                keep = (ntask - phase) / (ntask - phase + 1.0)
                prune_frac = 1.0 - keep
            else:
                prune_frac = 0.2
            opt.zero_grad()
            for xb, yb in dl:                      # phase A: one free epoch
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                out, _ = model(xb)
                F.cross_entropy(out, yb).backward()
                for k, v in Pd.items():
                    if v.grad is not None:
                        v.grad.mul_(1 - frozen[k])
                opt.step()
            # phase B: prune
            if fixed:                              # per layer
                surv = {}
                for k, v in Pd.items():
                    free = frozen[k].flatten() == 0
                    mags = v.detach().abs().flatten()[free]
                    if mags.numel() <= 10:
                        surv[k] = (frozen[k] > 0).float()
                        continue
                    k_n = max(1, int(mags.numel() * prune_frac))
                    thr = mags.kthvalue(min(k_n, mags.numel())).values
                    surv[k] = ((v.detach().abs() > thr) | (frozen[k] > 0)).float()
            else:                                  # original: global threshold
                mags = torch.cat([v.detach().abs().flatten()[frozen[k].flatten() == 0]
                                  for k, v in Pd.items()])
                surv = {}
                if mags.numel() > 10:
                    k_n = max(1, int(mags.numel() * prune_frac))
                    thr = mags.kthvalue(min(k_n, mags.numel())).values
                    for k, v in Pd.items():
                        surv[k] = ((v.detach().abs() > thr) | (frozen[k] > 0)).float()
                else:
                    for k, v in Pd.items():
                        surv[k] = torch.ones_like(v)
            with torch.no_grad():
                for k, v in Pd.items():
                    noise = torch.randn_like(v) * 0.01
                    v.mul_(surv[k]).add_(noise * (1 - surv[k]))
            # phase C: retrain survivors
            for ep in range(EPOCHS):
                for xb, yb in dl:
                    xb, yb = xb.to(DEV), yb.to(DEV)
                    opt.zero_grad()
                    out, _ = model(xb)
                    F.cross_entropy(out, yb).backward()
                    for k, v in Pd.items():
                        if v.grad is not None:
                            v.grad.mul_(surv[k] * (1 - frozen[k]))
                    opt.step()
            with torch.no_grad():
                for k, v in Pd.items():
                    frozen[k] = torch.clamp(frozen[k] + surv[k], max=1.0)
                    v.mul_(frozen[k])
            masks.append({k: frozen[k].clone() for k in Pd})
            # eval with per-task masks
            model.eval()
            snap = {k: v.detach().clone() for k, v in Pd.items()}
            with torch.no_grad():
                for cc in range(phase + 1):
                    for k, v in Pd.items():
                        v.copy_(snap[k] * masks[cc][k])
                    for (x_, y_, store) in ((xte, yte, acc), (vx, vy, vacc)):
                        xt = permute(x_, perms[cc]).to(DEV)
                        c = 0
                        for i in range(0, xt.size(0), 2000):
                            out, _ = model(xt[i:i + 2000])
                            c += (out.argmax(1) == y_[i:i + 2000].to(DEV)).sum().item()
                        store[(phase, cc)] = c / xt.size(0) * 100
                for k, v in Pd.items():
                    v.copy_(snap[k])
        else:
            # ---------------- penalty-based arms ----------------
            for ep in range(EPOCHS):
                for xb, yb in dl:
                    xb, yb = xb.to(DEV), yb.to(DEV)
                    pre = {k: v.detach().clone() for k, v in Pd.items()}
                    opt.zero_grad()
                    out, _ = model(xb)
                    loss = F.cross_entropy(out, yb)
                    if phase > 0 and lam > 0:
                        pen = sum((omega[k] * (Pd[k] - theta_ref[k]) ** 2).sum()
                                  for k in Pd)
                        loss = loss + lam * pen
                    loss.backward()
                    opt.step()
                    if base == "si":
                        # delta is only defined AFTER the step
                        for k, v in Pd.items():
                            if v.grad is not None:
                                integ[k] += (-v.grad.detach()) * (v.detach() - pre[k])

            # ---- importance of the task just learned ----
            if base == "si":
                if fixed:
                    for k in Pd:
                        # Zenke's accumulant is integral of -grad * dtheta, which is
                        # ALREADY positive for a descent step (dtheta = -lr*grad),
                        # so it must NOT be negated again. Clipping at 0 only
                        # guards against the rare ascent step.
                        d = (Pd[k].detach() - theta_start[k]) ** 2 + XI ** 2
                        omega[k] += torch.clamp(integ[k], min=0) / d
                else:
                    for k in Pd:
                        d = (Pd[k].detach() - theta_start[k]).abs() + 1e-3
                        omega[k] += integ[k].abs() / d
            elif base == "mas":
                if fixed:
                    # E_x[ || grad_x 0.5||f(x)||^2 ||^2 ] : square per sample,
                    # then average; omega is accumulated, never divided again.
                    o_new = {k: torch.zeros_like(v) for k, v in Pd.items()}
                    n_done = 0
                    for i in range(NGRAD_FIX):
                        model.zero_grad()
                        out, _ = model(xs[i:i + 1].to(DEV))
                        (0.5 * out.pow(2).sum()).backward()
                        for k, v in Pd.items():
                            o_new[k] += v.grad.detach() ** 2
                        n_done += 1
                    for k in Pd:
                        omega[k] += o_new[k] / max(1, n_done)
                else:
                    n_used = 0
                    for i in range(0, 2000, 250):
                        out, _ = model(xs[i:i + 250].to(DEV))
                        model.zero_grad()
                        (0.5 * out.pow(2).sum(dim=1)).mean().backward()
                        for k, v in Pd.items():
                            omega[k] += (v.grad.detach() ** 2) * (1.0 / out.size(0))
                        n_used += 1
                    for k in Pd:
                        omega[k] /= max(1, n_used)
            elif base == "ewc":
                o_new = {k: torch.zeros_like(v) for k, v in Pd.items()}
                n_done = 0
                for i in range(NGRAD_FIX):
                    model.zero_grad()
                    out, _ = model(xs[i:i + 1].to(DEV))
                    F.log_softmax(out, 1)[0, int(ys[i])].backward()   # true label
                    for k, v in Pd.items():
                        o_new[k] += v.grad.detach() ** 2
                    n_done += 1
                for k in Pd:
                    omega[k] += o_new[k] / max(1, n_done)

            theta_ref = {k: v.detach().clone() for k, v in Pd.items()}
            a = eval_model(model, xte, yte, perms, phase, ntask, False)
            acc.update(a)
            va = eval_model(model, vx, vy, perms, phase, ntask, False)
            vacc.update(va)

    bwt, favg = finish(acc, ntask)
    vbwt, vfavg = finish(vacc, ntask)
    return {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "val_bwt": round(vbwt, 2), "val_final_avg": round(vfavg, 2)}


# ------------------------------------------------------------------ jobs
def build_jobs():
    jobs = []
    # --- stage 1: lambda sweep on 5 tasks, seeds {0,1}, both variants ---
    for base in ("si", "mas", "ewc"):
        for lam in GRID[base]:
            for seed in (0, 1):
                if base != "ewc":
                    jobs.append(("sweep", f"{base}_old", 5, seed, lam))
                jobs.append(("sweep", f"{base}_new", 5, seed, lam))
    for lam in GRID["delp"]:
        for seed in (0, 1):
            jobs.append(("sweep", "ours", 5, seed, lam))
    # --- stage 2: main table, both task counts, 5 seeds ---
    for nt in (5, 10):
        for arm in MAIN_ARMS:
            for seed in SEEDS:
                jobs.append(("main", arm, nt, seed, None))
    return jobs


def pick_lambda(block, base, rule):
    """rule='test' reproduces exp6's leaky selection; rule='val' uses the
    held-out split. Ties broken on final accuracy, as in the original."""
    key = "bwt_mean" if rule == "test" else "val_bwt_mean"
    acc_key = "final_avg_mean" if rule == "test" else "val_final_avg_mean"
    cands = {l: v for l, v in block.items() if isinstance(v, dict) and key in v}
    if not cands:
        return None
    return max(cands, key=lambda l: (cands[l][key], cands[l][acc_key]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam-rule", default="val", choices=["test", "val"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--stage", default="both", choices=["sweep", "main", "both"])
    ap.add_argument("--lam-map", default="", help="JSON overriding LAM_MAP")
    ap.add_argument("--out", default="exp6_fixed_result.json")
    a = ap.parse_args()

    t0 = time.time()
    data = load_mnist()
    val = load_val()
    if a.lam_map:
        LAM_MAP.update(json.loads(a.lam_map))
    jobs = [j for j in build_jobs()
            if a.stage == "both" or j[0] == a.stage]
    mine = [j for i, j in enumerate(jobs) if i % a.n_tasks == a.task_id]
    print(f"[shard {a.task_id}/{a.n_tasks}] {len(mine)}/{len(jobs)} jobs", flush=True)

    res = {"lam_rule": a.lam_rule, "sweep": {}, "main": {}, "elapsed": None}
    for stage, arm, nt, seed, lam in mine:
        base = arm.split("_")[0] if arm != "ours" else "delp"
        if lam is None:
            lam = LAM_MAP.get(arm, 0.0)
        rec = {"stage": stage, "arm": arm, "ntask": nt, "seed": seed, "lam": lam}
        try:
            if arm == "ours":
                r = run_ours(seed, data, get_perms(seed, nt), nt, lam)
            elif arm.endswith("_old"):
                # call the ORIGINAL exp6 implementation verbatim: this arm must
                # reproduce exp12 exactly, otherwise the comparison is worthless
                l = lam if lam is not None else 0.0
                if base == "si":
                    r = run_si_impl(seed, data, get_perms(seed, nt), nt, l)
                elif base == "mas":
                    r = run_mas_impl(seed, data, get_perms(seed, nt), nt, l)
                else:
                    r = run_packnet(seed, data, get_perms(seed, nt), nt)
            else:
                b = "ewc" if arm == "ewc" else base
                r = train_arm(b, True, seed, data, val, get_perms(seed, nt), nt,
                              lam if lam is not None else 0.0)
            rec.update(r)
        except Exception as exc:                     # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {rec}", flush=True)
        if stage == "sweep":
            res["sweep"].setdefault(f"{arm}@{lam}", []).append(rec)
        else:
            res["main"].setdefault(f"{arm}|{nt}", []).append(rec)
        print(f"  {stage} {arm} T{nt} s{seed} lam={lam}: "
              f"bwt={rec.get('bwt')} val_bwt={rec.get('val_bwt')}", flush=True)

    res["elapsed"] = round(time.time() - t0, 1)
    tmp = a.out + f".part{a.task_id}"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(f"WROTE {a.out} in {res['elapsed']}s", flush=True)


if __name__ == "__main__":
    main()

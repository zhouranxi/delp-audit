"""ladder.py -- the stepwise ablation: EWC -> +context -> +protection -> +usage.

Why this file exists
--------------------
`ours` (DeLP) is not one idea. It is three mechanisms stacked on top of an EWC
penalty:

  (i)   architecture   -- per-task dendritic segments modulate a shared trunk
  (ii)  protection     -- gradients into a non-current segment are zeroed
  (iii) usage weighting-- the trunk Fisher is scaled by how often each hidden
                          unit fired within the task

The published table compares `ours` only against EWC/SI/MAS/PackNet, so it
cannot answer "which of the three pays for the gain". Every reviewer asks this
and there is currently no number to give them. This script adds one rung at a
time, holding everything else fixed (data, permutations, seeds, optimiser,
schedule, epochs, batch size, evaluation protocol):

  R0   mlp_ewc      PlainMLP + multi-anchor EWC              (the baseline)
  R0b  mlp_nosig    same MLP, kwta(0.5*t) instead of kwta(sigmoid(t))
  R1   ad_nocx      ADNet parameters, context pinned to 0    (capacity, no info)
  R2   ad_ctx       per-task segments                        (+ context modulation)
  R3   ad_prot      + branch protection                      (+ protection)
  R4   ad_usage     + usage-weighted Fisher                  == `ours`
  R5   ad_usage_ce  R4 with a true-label CE Fisher           (audit of the proxy)

R0b exists because the first pass of the ladder showed R1 already matching
R2/R3/R4 within noise: the gain may come from the activation, not from any
dendritic mechanism. R0b isolates exactly that. R0b and R1 are constructed to be
initialised identically and should agree bit for bit -- if they do, the dendrite
parameters provably contribute nothing on their own. That check is the whole
point of the rung, so it is a pass/fail test, not a nicety.

R4 must reproduce `exp6_baselines.run_ours` BIT FOR BIT. That is the same
correctness unit test as "lambda=0 must equal naive" in the CL benchmark: if the
ladder's top rung does not equal the method it is supposed to decompose, the
whole decomposition is worthless. Run `--selftest` before trusting anything.

Two caveats that belong in the paper, not hidden here
-----------------------------------------------------
* R0 -> R0b -> R1 keeps the activation change separate from the parameters, and
  R0b -> R1 is a true null step (identical initialisation, identical function,
  ~1280 extra parameters that never receive a gradient). Treating R0 -> R1 as
  "add capacity" without R0b would attribute the 6-point activation effect to
  the dendrites, which the sweep says is wrong.
* The penalty never touches `dend.weight`: `flat_params` only covers ff/out.
  Dendrites are guarded by protection alone. So "protection" and "penalty" act
  on disjoint parameter sets, which is worth stating explicitly in the paper.
* lambda selection: picking by BWT alone is DEGENERATE. BWT improves
  monotonically with lambda while average accuracy collapses, so a BWT-only rule
  always walks to the top of the grid (every rung selected lambda=1e5, where
  accuracy is 69-73 against a peak of 82). The sweep therefore selects lambda on
  held-out AVERAGE ACCURACY -- the standard CL practice -- and reports the
  BWT-selected lambda alongside so the degeneracy is visible rather than hidden.
  This also explains exp6's published lambda=5000: it was the top of its grid.

Usage
    python ladder.py --selftest                      # R4 vs run_ours, ~1 min
    python ladder.py --selftest-null                 # R0b vs R1, bit-identity
    python ladder.py --stage sweep --task-id 0 --n-tasks 4 --out res_l.json
    python ladder.py --stage main  --task-id 0 --n-tasks 26 --out res_m.json \
        --lam-map '{"ad_prot": 3000}' --only ad_ctx,ad_prot
"""
import argparse, json, os, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from exp4_branch_protection import (SEEDS, EPOCHS, BS, NTRAIN, NH, LR, DEV,
                                    load_mnist, permute, kwta, ADNet)
from exp6_baselines import (get_perms, flat_params, eval_model, finish,
                            PlainMLP, run_ours, NUSAGE, NFISHER)

# --------------------------------------------------------------------- rungs
# arch:  mlp   -> PlainMLP, h = kwta(sigmoid(t))
#        nosig -> PlainMLP, h = kwta(0.5*t)   (activation control for R0b)
#        nocx  -> ADNet with ctx pinned to 0 (u == 0, d == 0, one shared path)
#        adnet -> ADNet driven by the one-hot task context
# fisher: maxlogit -> exp6's original proxy, log_softmax(out).max(1)
#         ce       -> true-label log-prob (what EWC actually prescribes)
RUNGS = {
    "mlp_ewc":     dict(arch="mlp",   protect=False, usage=False, fisher="maxlogit"),
    "mlp_nosig":   dict(arch="nosig", protect=False, usage=False, fisher="maxlogit"),
    "ad_nocx":     dict(arch="nocx",  protect=False, usage=False, fisher="maxlogit"),
    "ad_ctx":      dict(arch="adnet", protect=False, usage=False, fisher="maxlogit"),
    "ad_prot":     dict(arch="adnet", protect=True,  usage=False, fisher="maxlogit"),
    "ad_usage":    dict(arch="adnet", protect=True,  usage=True,  fisher="maxlogit"),
    "ad_usage_ce": dict(arch="adnet", protect=True,  usage=True,  fisher="ce"),
}
ORDER = ["mlp_ewc", "mlp_nosig", "ad_nocx", "ad_ctx", "ad_prot", "ad_usage",
         "ad_usage_ce"]

# Wide enough to bracket the optimum for every rung: the usage-weighted Fisher
# is ~1/o smaller on ff.w (o is a firing rate, so a factor of 5-20), which
# pushes its optimal lambda up by the same factor.
GRID = [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]

LAM_MAP = {r: 5000.0 for r in ORDER}
LAM_MAP.update({"mlp_ewc": 5000.0, "ad_nocx": 5000.0, "ad_usage": 5000.0})
TS = (5, 10, 20)


def load_val():
    """Held-out split for lambda selection: MNIST train[10000:12000].

    exp6 selected lambda on the TEST set (exp6:336-337). The ladder must not
    inherit that leak, otherwise every rung can be tuned towards the metric it
    is later judged on. This split is disjoint from the 10k training subset and
    from the test set, and leaves NTRAIN unchanged.
    """
    from torchvision import datasets
    tr = datasets.MNIST("./data", train=True, download=True)
    return (tr.data[NTRAIN:NTRAIN + 2000].float().view(-1, 784) / 255.0,
            tr.targets[NTRAIN:NTRAIN + 2000])


class ADNetCap(ADNet):
    """ADNet whose branch protection folds task ids onto the capped segment set.

    exp4's protect_grads indexes mask[ctx_label] directly, which is only correct
    while nseg == ntask.  Under a capped nseg (the fixed-budget main table) task
    ids past the cap must share the last segment instead of raising IndexError.
    bench_cl.py already does exactly this, so the two harnesses agree.

    With nseg == ntask the min() never binds, so this class is a no-op for the
    historical configuration and the bit-identity self-test still holds.
    """

    def protect_grads(self, ctx_label):
        if not self.protected or self.dend.weight.grad is None:
            return
        W = self.dend.weight.grad
        Wv = W.view(self.nseg, self.nh, -1)
        mask = torch.zeros(self.nseg, 1, 1, device=W.device)
        mask[min(int(ctx_label), self.nseg - 1)] = 1.0
        Wv.mul_(mask)


class ADNetNoCtx(ADNet):
    """ADNet with the context input pinned to zero.

    u = dend(0) = 0 (the dendrite layer has no bias), so d = 0, sigmoid(d) = 0.5
    and h = kwta(0.5 * t): every task shares one pathway and no context
    information reaches the network. Same parameters, same initialisation, same
    RNG consumption as ADNet -- only the information is removed.
    """

    def forward(self, x, ctx=None):
        return super().forward(x, torch.zeros_like(ctx))


class MLPNoSig(PlainMLP):
    """PlainMLP with kwta(0.5*t) instead of kwta(sigmoid(t)) -- and nothing else.

    This rung exists because the first run of the ladder showed that R1 (dendrites
    present, context pinned to 0) already matches R2/R3/R4 to within noise, i.e.
    the accuracy gain of `ours` may come from the ACTIVATION and not from any
    dendritic mechanism at all. PlainMLP computes h = kwta(sigmoid(ff(x)));
    ADNet computes h = kwta(t * sigmoid(d)) which, with d == 0, is kwta(0.5*t).
    So R0 -> R1 bundles "extra parameters" with "remove the sigmoid", and the two
    must be separated or the whole ladder is uninterpretable.

    The `_pad` layer is never used. It exists only to draw the same number of
    random values in the same order as ADNet's dendrite layer, so that R0b and R1
    are initialised identically and should agree BIT FOR BIT. If they do, the
    dendrite parameters provably contribute nothing on their own.
    """

    def __init__(self, nh=NH, ntask=5):
        # The layer order must match ADNet.__init__ exactly (ff, dend-equivalent,
        # out): nn.Linear draws its initial weights from the global RNG, so any
        # reordering changes `out`'s initialisation and destroys the bit-identity
        # this rung is supposed to demonstrate.
        torch.nn.Module.__init__(self)
        self.ff = torch.nn.Linear(784, nh)
        self._pad = torch.nn.Linear(ntask, nh * ntask, bias=False)
        self.out = torch.nn.Linear(nh, 10)

    def forward(self, x, ctx=None):
        return self.out(kwta(0.5 * self.ff(x))), None


# nseg = 0 keeps the historical behaviour (one dendrite segment per task, so the
# dendrite block is 256*T*T and grows quadratically with the sequence length).
# A positive value is an upper BOUND, not a pin: ns = min(bound, ntask), i.e.
# never more than one segment per task, capped. At T = 5 a cap of 8 is inert,
# so those runs stay bit-identical to the published ones and can be used as the
# self-check that the knob does nothing when it should not.
NSEG = 0


def make_model(rung, ntask):
    spec = RUNGS[rung]
    ns = min(NSEG, ntask) if NSEG > 0 else ntask
    if spec["arch"] == "mlp":
        return PlainMLP().to(DEV)
    if spec["arch"] == "nosig":
        return MLPNoSig(ntask=ntask).to(DEV)
    if spec["arch"] == "nocx":
        return ADNetNoCtx(protected=False, nseg=ns, ntask=ntask).to(DEV)
    return ADNetCap(protected=spec["protect"], nseg=ns, ntask=ntask).to(DEV)


# ------------------------------------------------------------------ fisher
def fisher_block(model, xs, ys, task_id, ntask, usage, mode):
    """A faithful re-implementation of exp6_baselines.fisher_usage, with the
    two switches the ladder needs.

    usage=False  drops the per-unit scaling of ff.w / ff.b / out.w by the
                 firing rate (mechanism iii).
    mode='ce'    replaces the max-logit proxy with the true-label log-prob.

    Everything else -- the 300 per-sample backwards, the 500-sample usage
    statistic, the /NFISHER normalisation, the fact that only the four
    flat_params keys are touched -- is unchanged.
    """
    model.eval()
    P = flat_params(model)
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    ctx1 = F.one_hot(torch.tensor([task_id], device=DEV), ntask).float()

    if usage:
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

    for i in range(NFISHER):
        xb = xs[i:i + 1].to(DEV)
        out, _ = model(xb, ctx1)
        model.zero_grad()
        if mode == "ce":
            F.log_softmax(out, 1)[0, ys[i]].backward()
        else:
            # exp6's proxy: the log-prob of the response the model already
            # prefers. Note this is the ORIGINAL behaviour, kept so that
            # ad_usage reproduces run_ours exactly.
            F.log_softmax(out, 1).max(1).values.sum().backward()
        for k, v in P.items():
            fisher[k] += v.grad.detach() ** 2
    for k in fisher:
        fisher[k] /= NFISHER
    if usage:
        # Per-unit importance scaling: every parameter that indexes hidden unit
        # j has its Fisher multiplied by o_j (how often that unit fired within
        # this task). Three tensors, not one -- exp6:91-93 in fisher_usage.
        fisher["ff.w"] = fisher["ff.w"] * o.unsqueeze(1)
        fisher["ff.b"] = fisher["ff.b"] * o
        fisher["out.w"] = fisher["out.w"] * o.unsqueeze(0)
    return fisher


# ------------------------------------------------------------------- one rung
def run_rung(rung, seed, data, perms, ntask, lam, val=None):
    spec = RUNGS[rung]
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = make_model(rung, ntask)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc, vacc = [], {}, {}
    needs_ctx = spec["arch"] != "mlp"

    for phase in range(ntask):
        xs, ys = permute(xtr[:NTRAIN], perms[phase]), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                ctx = F.one_hot(torch.full((xb.size(0),), phase, device=DEV),
                                ntask).float()
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
                if spec["protect"]:
                    model.protect_grads(phase)
                opt.step()

        P = flat_params(model)
        anchors.append({"fisher": fisher_block(model, xs, ys, phase, ntask,
                                               spec["usage"], spec["fisher"]),
                        "theta": {k: v.detach().clone() for k, v in P.items()}})
        acc.update(eval_model(model, xte, yte, perms, phase, ntask, needs_ctx))
        if val is not None:
            vacc.update(eval_model(model, val[0], val[1], perms, phase, ntask,
                                  needs_ctx))
        print(f"  s{seed} {rung} T{phase}: "
              + " ".join(f"t{c}={acc[(phase, c)]:.0f}" for c in range(phase + 1)),
              flush=True)

    bwt, favg = finish(acc, ntask)
    # Budget accounting, so the table can state what each rung costs instead of
    # leaving the reader to guess whether a gap is a mechanism or extra weights.
    out = {"seed": seed, "bwt": round(bwt, 2), "final_avg": round(favg, 2),
           "n_param": sum(p.numel() for p in model.parameters()),
           "n_dend": int(model.dend.weight.numel()) if hasattr(model, "dend") else 0,
           "nseg": int(getattr(model, "nseg", 0)),
           "acc_matrix": [[round(acc[(p, c)], 2) for c in range(p + 1)]
                          for p in range(ntask)]}
    if val is not None:
        vbwt, vfavg = finish(vacc, ntask)
        out.update({"val_bwt": round(vbwt, 2), "val_final_avg": round(vfavg, 2)})
    return out


# --------------------------------------------------------------------- jobs
def build_jobs(only=None, sweep_nt=5, grid=None):
    rungs = [r for r in ORDER if not only or r in only]
    grid = grid or GRID
    jobs = []
    # The sweep carries the lambda curves the paper plots, so it gets the full
    # 5 seeds: a curve without error bars cannot support "the peak is flat".
    for rung in rungs:
        for lam in grid:
            for seed in SEEDS:
                jobs.append(("sweep", rung, sweep_nt, seed, lam))
    for nt in TS:
        for rung in rungs:
            for seed in SEEDS:
                jobs.append(("main", rung, nt, seed, None))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main", choices=["sweep", "main"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--lam-map", default="")
    ap.add_argument("--only", default="", help="comma-separated rung names")
    ap.add_argument("--nt", type=int, default=5,
                    help="task count used by the sweep stage")
    ap.add_argument("--lams", default="",
                    help="comma list overriding the lambda grid, e.g. 10,30")
    ap.add_argument("--nseg", type=int, default=0,
                    help="0 = nseg follows ntask (historical); >0 = upper bound, "
                         "ns = min(nseg, ntask)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-null", dest="selftest_null", action="store_true")
    ap.add_argument("--out", default="ladder_result.json")
    a = ap.parse_args()

    global NSEG
    NSEG = a.nseg

    t0 = time.time()
    data = load_mnist()

    if a.selftest:
        # R4 must equal the published method, bit for bit.
        perms = get_perms(0, 5)
        r4 = run_rung("ad_usage", 0, data, perms, 5, 5000.0)
        ref = run_ours(0, data, perms, 5, 5000.0)
        same = r4["acc_matrix"] == ref["acc_matrix"]
        print(json.dumps({"identical": same,
                          "ladder": r4["acc_matrix"],
                          "run_ours": ref["acc_matrix"],
                          "bwt_ladder": r4["bwt"], "bwt_run_ours": ref["bwt"]},
                         indent=1))
        raise SystemExit(0 if same else 1)

    if a.selftest_null:
        # R0b vs R1: identical init, identical function, only an unused tensor
        # of parameters differs. They must agree bit for bit, which is what
        # makes "dendrite capacity alone contributes nothing" a fact rather
        # than an inference from noisy means.
        perms = get_perms(0, 5)
        r0b = run_rung("mlp_nosig", 0, data, perms, 5, 1000.0)
        r1 = run_rung("ad_nocx", 0, data, perms, 5, 1000.0)
        same = r0b["acc_matrix"] == r1["acc_matrix"]
        print(json.dumps({"identical": same, "bwt_r0b": r0b["bwt"],
                          "bwt_r1": r1["bwt"],
                          "acc_matrix": r0b["acc_matrix"]}, indent=1))
        raise SystemExit(0 if same else 1)

    if a.lam_map:
        LAM_MAP.update(json.loads(a.lam_map))
    only = set(x.strip() for x in a.only.split(",") if x.strip())
    grid = [float(x) for x in a.lams.split(",") if x.strip()] if a.lams else None
    jobs = [j for j in build_jobs(only, a.nt, grid) if j[0] == a.stage]
    mine = [j for i, j in enumerate(jobs) if i % a.n_tasks == a.task_id]
    print(f"[shard {a.task_id}/{a.n_tasks}] {len(mine)}/{len(jobs)} jobs", flush=True)

    res = {"stage": a.stage, "sweep": {}, "main": {}, "elapsed": None}
    # only the sweep needs the held-out split: the main stage reuses the lambda
    # the sweep selected, so scoring val there would just burn time.
    val = load_val() if a.stage == "sweep" else None
    for stage, rung, nt, seed, lam in mine:
        lam_used = LAM_MAP.get(rung, 0.0) if lam is None else lam
        rec = {"stage": stage, "arm": rung, "ntask": nt, "seed": seed, "lam": lam_used}
        try:
            rec.update(run_rung(rung, seed, data, get_perms(seed, nt), nt,
                                lam_used, val=val))
        except Exception as exc:                      # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {rec}", flush=True)
        if stage == "sweep":
            res["sweep"].setdefault(f"{rung}@{lam_used}", []).append(rec)
        else:
            res["main"].setdefault(f"{rung}|{nt}", []).append(rec)
        print(f"  {stage} {rung} T{nt} s{seed} lam={lam_used}: "
              f"bwt={rec.get('bwt')} acc={rec.get('final_avg')}", flush=True)

    res["elapsed"] = round(time.time() - t0, 1)
    tmp = a.out + f".part{a.task_id}"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(f"WROTE {a.out} in {res['elapsed']}s", flush=True)


if __name__ == "__main__":
    main()

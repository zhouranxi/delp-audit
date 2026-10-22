"""sig_probe.py -- why does removing the sigmoid buy 5 accuracy points?

The ladder showed that R0b (kwta(0.5*t)) beats R0 (kwta(sigmoid(t))) by +5.4
points at T=5 and +2.5 at T=10, and that every dendritic mechanism adds at most
+0.8. That is a measured fact, but a bare fact is not an explanation. The
hypothesis here is concrete and testable:

    kWTA keeps the k largest activations. sigmoid saturates, so most units pile
    up near 1.0 and "the top 64" becomes close to an arbitrary subset. Without
    the squash, the top-64 is a genuine top-64 by evidence, and the magnitude
    that survives still carries information.

If that is the mechanism, then a gentler sigmoid must recover the accuracy
monotonically:  kwta(sigmoid(g*t))  with g -> 0  should approach kwta(0.5*t),
and with g -> large it should collapse towards the saturated regime. And the
measured saturation fraction has to track the accuracy curve.

So this script sweeps the gain g and, at the end of every task, RECORDS the
fraction of hidden units with sigmoid(g*t) > 0.99 on that task's data -- the
mechanism variable itself, not a proxy for it.

Design notes
  * identical data, permutations, seeds, optimiser, epochs, batch, kWTA k and
    multi-anchor EWC penalty as ladder.py -- this is the same instrument, one
    parameter richer.
  * g = 0 means "no sigmoid": h = kwta(0.5*t), which reproduces ladder's
    mlp_nosig exactly (checked with --selftest).
"""
import argparse, json, os, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from exp4_branch_protection import (SEEDS, EPOCHS, BS, NTRAIN, NH, LR, DEV,
                                    load_mnist, permute, kwta)
from exp6_baselines import get_perms, flat_params, eval_model, finish, NUSAGE, NFISHER

GAINS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]      # 0.0 == no sigmoid
LAMS = [300.0, 1000.0]
TS = (5, 10)


class GainMLP(torch.nn.Module):
    """h = kwta(alpha*t) when alpha is given, else
       h = kwta(0.5*t) for g == 0 and h = kwta(sigmoid(g*t)) for g > 0.

    The alpha mode is the control the ladder needs. R0 (PlainMLP) computes
    kwta(t) and R0b computes kwta(0.5*t): identical initialisation, identical
    penalty, and the null-control rung proved the two differ by nothing but
    that factor of 0.5. If sweeping alpha over a plain MLP+EWC reproduces the
    whole R0 -> R4 range, then "the dendritic/gating architecture" is a scalar
    the baseline was never allowed to tune.
    """

    def __init__(self, gain, nh=NH, ntask=5, alpha=None):
        super().__init__()
        self.gain = float(gain) if gain is not None else None
        self.alpha = float(alpha) if alpha is not None else None
        # Layer order and shapes must match ladder.MLPNoSig (ff, unused pad,
        # out) so that g=0 reproduces it bit for bit -- otherwise the selftest
        # compares two differently initialised networks and is meaningless.
        self.ff = torch.nn.Linear(784, nh)
        self._pad = torch.nn.Linear(ntask, nh * ntask, bias=False)
        self.out = torch.nn.Linear(nh, 10)

    def hidden(self, x):
        t = self.ff(x)
        if self.alpha is not None:
            return kwta(self.alpha * t), None
        if self.gain == 0.0:
            return kwta(0.5 * t), None
        s = torch.sigmoid(self.gain * t)
        return kwta(s), s

    def forward(self, x, ctx=None):
        h, _ = self.hidden(x)
        return self.out(h), None


def fisher_block(model, xs, ntask_dummy):
    """True-label CE Fisher over the four flat_params keys (same estimator the
    ladder uses for R0/R0b, so the comparison is apples to apples)."""
    model.eval()
    P = flat_params(model)
    fisher = {k: torch.zeros_like(v) for k, v in P.items()}
    for i in range(NFISHER):
        xb = xs[i:i + 1].to(DEV)
        out, _ = model(xb)
        model.zero_grad()
        F.log_softmax(out, 1).max(1).values.sum().backward()
        for k, v in P.items():
            fisher[k] += v.grad.detach() ** 2
    for k in fisher:
        fisher[k] /= NFISHER
    return fisher


def run(gain, seed, data, perms, ntask, lam, alpha=None):
    xtr, ytr, xte, yte = data
    torch.manual_seed(seed)
    model = GainMLP(gain, ntask=ntask, alpha=alpha).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    anchors, acc = [], {}
    sat = []                       # saturation fraction after each task

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
                if anchors:
                    P = flat_params(model)
                    pen = 0.0
                    for a in anchors:
                        for k, v in P.items():
                            pen = pen + (a["fisher"][k] * (v - a["theta"][k]) ** 2).sum()
                    loss = loss + lam * pen
                loss.backward()
                opt.step()

        # mechanism variable: how saturated is the hidden layer on this task?
        if gain is not None and gain > 0:
            with torch.no_grad():
                xb = xs[:2000].to(DEV)
                _, s = model.hidden(xb)
                sat.append(round(float((s > 0.99).float().mean()), 4))

        P = flat_params(model)
        anchors.append({"fisher": fisher_block(model, xs, None),
                        "theta": {k: v.detach().clone() for k, v in P.items()}})
        acc.update(eval_model(model, xte, yte, perms, phase, ntask, False))
        tag = f"a{alpha:g}" if alpha is not None else f"g{gain:g}"
        print(f"  s{seed} {tag} T{phase}: "
              + " ".join(f"t{c}={acc[(phase, c)]:.0f}" for c in range(phase + 1))
              + (f"  sat={sat[-1]:.3f}" if sat else ""), flush=True)

    bwt, favg = finish(acc, ntask)
    return {"seed": seed, "gain": gain, "alpha": alpha, "lam": lam,
            "bwt": round(bwt, 2), "final_avg": round(favg, 2),
            "sat_mean": round(sum(sat) / len(sat), 4) if sat else 0.0,
            "sat_final": sat[-1] if sat else 0.0}


def build_jobs(gains=None, lams=None, alphas=None):
    jobs = []
    if alphas:
        for nt in TS:
            for al in alphas:
                for lam in (lams or LAMS):
                    for seed in SEEDS:
                        jobs.append((nt, None, lam, seed, al))
        return jobs
    for nt in TS:
        for g in (gains or GAINS):
            for lam in (lams or LAMS):
                for seed in SEEDS:
                    jobs.append((nt, g, lam, seed, None))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--gains", default="", help="comma list, e.g. 2,4")
    ap.add_argument("--lams", default="", help="comma list, e.g. 30,100")
    ap.add_argument("--nts", default="", help="comma list, e.g. 5,10")
    ap.add_argument("--alphas", default="",
                    help="hidden-scale sweep instead of a gain sweep, e.g. 0.25,0.5,1,2")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-scale", action="store_true")
    ap.add_argument("--out", default="sig_result.json")
    a = ap.parse_args()

    t0 = time.time()
    data = load_mnist()

    if a.selftest_scale:
        # The anchor is alpha = 0.5 <-> R0b (kwta(0.5t)): same layer order, so
        # the two must agree bit for bit. alpha = 1 is NOT expected to equal R0,
        # because R0 (PlainMLP) is kwta(sigmoid(t)), not kwta(t) -- reported for
        # the record only.
        from ladder import run_rung
        perms = get_perms(0, 5)
        r0b = run(None, 0, data, perms, 5, 1000.0, alpha=0.5)
        l0b = run_rung("mlp_nosig", 0, data, perms, 5, 1000.0)
        r1 = run(None, 0, data, perms, 5, 1000.0, alpha=1.0)
        l0 = run_rung("mlp_ewc", 0, data, perms, 5, 1000.0)
        ok = abs(r0b["final_avg"] - l0b["final_avg"]) < 1e-9
        print(json.dumps({"anchored": ok, "a0.5": r0b, "R0b": l0b,
                          "a1.0": r1, "R0": l0}, indent=1))
        raise SystemExit(0 if ok else 1)

    if a.selftest:
        # g=0 must reproduce ladder.py mlp_nosig at the same seed/lambda, or the
        # two instruments are not measuring the same thing.
        from ladder import run_rung
        perms = get_perms(0, 5)
        got = run(0.0, 0, data, perms, 5, 1000.0)
        ref = run_rung("mlp_nosig", 0, data, perms, 5, 1000.0)
        same = abs(got["bwt"] - ref["bwt"]) < 1e-9 and \
            abs(got["final_avg"] - ref["final_avg"]) < 1e-9
        print(json.dumps({"identical": same, "probe": got, "ladder": ref}, indent=1))
        raise SystemExit(0 if same else 1)

    global TS
    gains = [float(x) for x in a.gains.split(",") if x.strip()] if a.gains else None
    lams = [float(x) for x in a.lams.split(",") if x.strip()] if a.lams else None
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()] if a.alphas else None
    if a.nts:
        TS = tuple(int(x) for x in a.nts.split(",") if x.strip())
    jobs = build_jobs(gains, lams, alphas)
    mine = [j for i, j in enumerate(jobs) if i % a.n_tasks == a.task_id]
    print(f"[shard {a.task_id}/{a.n_tasks}] {len(mine)}/{len(jobs)} jobs", flush=True)

    res = {"runs": {}, "elapsed": None}
    for nt, g, lam, seed, al in mine:
        rec = {"ntask": nt, "seed": seed}
        try:
            rec.update(run(g, seed, data, get_perms(seed, nt), nt, lam, alpha=al))
        except Exception as exc:                       # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {rec}", flush=True)
        key = (f"a{al:g}@{lam}|{nt}" if al is not None else f"g{g}@{lam}|{nt}")
        res["runs"].setdefault(key, []).append(rec)
        print(f"  nt{nt} {key} s{seed}: bwt={rec.get('bwt')} "
              f"acc={rec.get('final_avg')}", flush=True)

    res["elapsed"] = round(time.time() - t0, 1)
    tmp = a.out + f".part{a.task_id}"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, a.out)
    print(f"WROTE {a.out} in {res['elapsed']}s", flush=True)


if __name__ == "__main__":
    main()

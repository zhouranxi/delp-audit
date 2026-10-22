"""d2_xgap.py -- train at one gap, test at every gap.

Why this is the decisive test
-----------------------------
Every cell in results_d2_wm trains and tests at the SAME gap: cfg["gap"] feeds
both make_data calls, so a model only ever has to survive the delay it was
optimised for. That protocol cannot separate two very different explanations
of a "horizon":

  H1  a real memory mechanism holds the cue and decays gradually with delay;
  H2  the network fits the delay it was shown, and has no notion of "longer".

Both look identical in the same-gap protocol, and the write-gate arms in
results_d2_wm are ambiguous in exactly this way:

  point_wmh   gap 2/10/40 -> 100.0 on all five seeds
              gap 80/160/320 -> 31-33 (chance)
              i.e. a CLIFF between 40 and 80, not a decay curve.
  dual_wmh    gap 40 -> [100.0, 95.8, 64.3, 32.1, 31.0]  (bimodal; the mean
              64.6 that section 2 of d2wm.md reports is not a summary of
              anything)

H1 predicts graceful degradation when the test gap moves off the training gap.
H2 predicts a collapse as soon as the test gap differs from the training gap.
So: train once at --train-gap, then evaluate the SAME weights at every gap in
--test-gaps. Nothing else changes -- same seed, same training set, same
readout window (RATE_WINDOW = 15), same epoch budget.

usage
    python d2_xgap.py --arms point,point_slow,point_wmh,dual_wmh \
                      --train-gap 40 --test-gaps 2,10,40,80,160,320 \
                      --seeds 0,1,2,3,4
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

import d2_comp_snn as M

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HONEST_WINDOW = 15


def cfg_for(couple, tau_a, nh=96):
    return dict(C=3, M=3, nh=nh, ntrain=4000, ntest=1000, epochs=20, bs=128,
                lr=0.004, gamma=1.0, tau_s=0.85, tau_a=tau_a, kappa=0.8,
                alif_rho=0.9, alif_beta=0.5, eval_every=20, data_seed=1,
                couple=couple, ncue=3)


def train_one(arm, couple, tau, train_gap, seed, nh):
    """Train on train_gap only, then score the same weights at every test gap."""
    torch.manual_seed(seed)
    cfg = cfg_for(couple, tau, nh=nh)
    (xtr, ytr, _), _, _, NIN, nout = M.make_data(
        C=cfg["C"], M=cfg["M"], gap=train_gap, ncue=cfg["ncue"],
        ntrain=cfg["ntrain"], ntest=cfg["ntest"], seed=cfg["data_seed"])
    model = M.CompSNN(NIN, cfg["nh"], nout, arm=arm, gamma=cfg["gamma"],
                      tau_s=cfg["tau_s"], tau_a=cfg["tau_a"], kappa=cfg["kappa"],
                      alif_rho=cfg["alif_rho"], alif_beta=cfg["alif_beta"],
                      couple=couple).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    xtr, ytr = xtr.to(DEV), ytr.to(DEV)
    n = xtr.size(0)
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, cfg["bs"]):
            idx = perm[i:i + cfg["bs"]]
            opt.zero_grad()
            out, _ = model(xtr[idx])
            F.cross_entropy(out, ytr[idx]).backward()
            opt.step()
    return model, cfg, nparam


def eval_at(model, cfg, gap, nocue=False):
    (_, _, _), (xte, yte, cte), _, _, _ = M.make_data(
        C=cfg["C"], M=cfg["M"], gap=gap, ncue=cfg["ncue"],
        ntrain=cfg["ntrain"], ntest=cfg["ntest"], seed=cfg["data_seed"])
    if nocue:
        # Remove the cue from the cue window only; everything else (template,
        # background, timing) is untouched. If accuracy survives this, the
        # model is solving the task from the template alone and the "memory"
        # was never being read -- the direct check that 100.0 is not an
        # artefact of an easy readout.
        xte = xte.clone()
        xte[:, :cfg["ncue"], :cfg["C"]] = 0.0
    probe_t = cfg["ncue"] + gap - 1
    # make_data returns CPU tensors; evaluate() only moves the labels, so the
    # inputs have to be moved here (same reason run() does it explicitly).
    xte = xte.to(DEV)
    acc, _spk, ctxdec = M.evaluate(model, xte, yte, cte, cfg, probe_t)
    return round(acc, 2), round(ctxdec, 2)


@torch.no_grad()
def gate_stats(model, cfg, gap, nbat=64):
    """Mean gate value inside the cue window and inside the silent gap."""
    (xtr, _, _), _, _, _, _ = M.make_data(
        C=cfg["C"], M=cfg["M"], gap=gap, ncue=cfg["ncue"],
        ntrain=cfg["ntrain"], ntest=cfg["ntest"], seed=cfg["data_seed"])
    xb = xtr[:nbat].to(DEV)
    model.eval()
    s = torch.zeros(xb.size(0), cfg["nh"], device=DEV)
    cue, sil = [], []
    nc = cfg["ncue"]
    for t in range(xb.size(1)):
        g = torch.sigmoid(model.wgate(torch.cat([xb[:, t], s], 1)))
        (cue if t < nc else sil).append(float(g.mean()))
        s = (1 - g) * s + g * torch.tanh(model.wcand(xb[:, t]))
    return (round(sum(cue) / max(1, len(cue)), 4),
            round(sum(sil) / max(1, len(sil)), 4))


def eval_ablated(model, cfg, gap, mode):
    """Score the SAME weights with the gate overridden.

    This is the control that decides whether the gated state is doing the work
    or whether the extra parameters (wgate/wcand are ~27k on top of ~19k) have
    simply bought capacity.  mode 'frozen' holds g at the trained mean, which
    keeps every parameter and the whole forward path intact; mode 'open' pins
    g = 1, which makes s a pure function of the current input (no memory).
    """
    (_, _, _), (xte, yte, cte), _, _, _ = M.make_data(
        C=cfg["C"], M=cfg["M"], gap=gap, ncue=cfg["ncue"],
        ntrain=cfg["ntrain"], ntest=cfg["ntest"], seed=cfg["data_seed"])
    xte = xte.to(DEV)

    # mean trained gate over the whole trial, measured on the training inputs
    (xtr, _, _), _, _, _, _ = M.make_data(
        C=cfg["C"], M=cfg["M"], gap=gap, ncue=cfg["ncue"],
        ntrain=cfg["ntrain"], ntest=cfg["ntest"], seed=cfg["data_seed"])
    xb = xtr[:64].to(DEV)
    model.eval()
    s = torch.zeros(xb.size(0), cfg["nh"], device=DEV)
    tot = 0.0
    with torch.no_grad():
        for t in range(xb.size(1)):
            g = torch.sigmoid(model.wgate(torch.cat([xb[:, t], s], 1)))
            tot += float(g.mean())
            s = (1 - g) * s + g * torch.tanh(model.wcand(xb[:, t]))
    const = tot / xb.size(1)

    def _write(s_, xt_, Ic_):
        g_ = (torch.full_like(s_, const) if mode == "frozen"
              else torch.ones_like(s_))
        return (1 - g_) * s_ + g_ * torch.tanh(model.wcand(xt_))

    keep = model._write
    model._write = _write
    try:
        probe_t = cfg["ncue"] + gap - 1
        acc, _spk, ctxdec = M.evaluate(model, xte, yte, cte, cfg, probe_t)
    finally:
        model._write = keep
    return round(acc, 2), round(ctxdec, 2), round(const, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-gap", type=int, default=40)
    ap.add_argument("--test-gaps", default="2,10,40,80,160,320")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--nh", type=int, default=96)
    ap.add_argument("--ablate", action="store_true",
                    help="also score frozen-gate / open-gate controls for _wm arms")
    ap.add_argument("--only-arms", dest="only_arms", default="",
                    help="comma-separated arm names to keep, e.g. point_wmh")
    ap.add_argument("--taus", default="0.97,0.999")
    ap.add_argument("--outdir", default="results_d2_xgap")
    a = ap.parse_args()

    taus = [float(x) for x in a.taus.split(",") if x.strip()]
    gaps = [int(x) for x in a.test_gaps.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]

    # (arm, couple, tau) -- tau only matters for the leaky arms; the wm arms
    # ignore it (their state is pure write-gate, no leak), but it is carried
    # so the config dict stays uniform.
    specs = [("point", "-", 0.0)]
    specs += [("point_slow", "-", t) for t in taus]
    specs += [("point_wm", "-", 0.97), ("point_wmh", "-", 0.97)]
    specs += [("dual_wm", "thr", 0.97), ("dual_wmh", "thr", 0.97)]
    if a.only_arms:
        keep = {x.strip() for x in a.only_arms.split(",") if x.strip()}
        specs = [s for s in specs if s[0] in keep]
        if not specs:
            raise SystemExit(f"--only-arms {a.only_arms} matched nothing")

    M.RATE_WINDOW = HONEST_WINDOW
    os.makedirs(a.outdir, exist_ok=True)
    print(f"[d2_xgap] train_gap={a.train_gap} window={HONEST_WINDOW} "
          f"specs={len(specs)} seeds={len(seeds)}", flush=True)

    for arm, couple, tau in specs:
        for seed in seeds:
            tag = f"X_{arm}_{couple}_ta{tau:g}_tr{a.train_gap}_s{seed}".replace("-", "x")
            out = os.path.join(a.outdir, tag + ".json")
            if os.path.exists(out) and os.path.getsize(out) > 0:
                continue
            t0 = time.time()
            rec = {"arm": arm, "couple": couple, "tau_a": tau, "seed": seed,
                   "train_gap": a.train_gap, "rate_window": HONEST_WINDOW}
            try:
                model, cfg, nparam = train_one(arm, couple, tau, a.train_gap,
                                               seed, a.nh)
                rec["nparam"] = nparam
                rec["curve"] = {}
                for g in gaps:
                    acc, ctx = eval_at(model, cfg, g)
                    rec["curve"][str(g)] = {"acc": acc, "ctx_probe": ctx}
                if a.ablate and ("_wm" in arm):
                    rec["frozen"], rec["open"] = {}, {}
                    for g in gaps:
                        fa, fc, const = eval_ablated(model, cfg, g, "frozen")
                        oa, oc, _ = eval_ablated(model, cfg, g, "open")
                        rec["frozen"][str(g)] = {"acc": fa, "ctx_probe": fc}
                        rec["open"][str(g)] = {"acc": oa, "ctx_probe": oc}
                        rec["gate_const"] = const
                    rec["gate_at_train"] = gate_stats(model, cfg, a.train_gap)
                    # cue ablation on the same weights: does the answer still
                    # come out right when the context was never shown?
                    rec["nocue"] = {}
                    for g in gaps:
                        na, nc_ = eval_at(model, cfg, g, nocue=True)
                        rec["nocue"][str(g)] = {"acc": na, "ctx_probe": nc_}
                on = rec["curve"].get(str(a.train_gap), {}).get("acc")
                off = [f"{g}:{rec['curve'][str(g)]['acc']:.1f}"
                       for g in gaps if g != a.train_gap]
                print(f"  {tag}: on-gap {on} | off-gap {' '.join(off)} "
                      f"| {time.time()-t0:.0f} s", flush=True)
            except Exception as exc:                      # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  FAIL {tag} {rec['error']}", flush=True)
            tmp = out + ".part"
            json.dump(rec, open(tmp, "w"), indent=1)
            os.replace(tmp, out)
    print("DONE d2_xgap", flush=True)


if __name__ == "__main__":
    main()

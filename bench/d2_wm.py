"""d2_wm.py -- what sets the ~40-step context horizon: the leak, or the task?

Where this comes from
---------------------
Once the readout leak is removed (2.4d), every input-integrating arm holds the
cue for about 40 silent steps and then falls to chance, and 0.97^40 ~ 0.30 -- the
number tracks tau_a almost exactly. An ALIF, slow but NOT input-integrating,
collapses by gap=10. So the narrow claim is "a slow INPUT-DRIVEN state carries
context, and slowness alone does not".

Two experiments follow, and they answer different questions.

(1) tau_a dose-response (the law). If the horizon is set by the leak, then
    h(tau_a) ~ c / (1 - tau_a) with c the same constant for every architecture.
    A pure hyperparameter sweep, no new mechanism, no learnability problem, and
    it converts "40 steps" from a limitation into a PREDICTION. This is the
    primary experiment.

(2) write-gated state (the mechanism). Replace s <- tau_a s + Ic with
        g = sigmoid(Wg [xt, s] + bg);  s <- (1-g) s + g tanh(Wc xt)
    so the state can be HELD (g -> 0) rather than decayed. If a learned gate can
    hold the cue past 1/(1-tau_a), the ceiling was the dynamics.
    ⚠ Deviation from the pre-registration, recorded rather than hidden: the first
    smoke run used a neutral gate init (bg = 0) and the gate stayed wide open
    (mean g = 0.61 in the cue window, 0.63 in the gap) with accuracy at chance
    (34.4 at gap=40, where the leaky control gets ~81). A neutral init gives the
    optimiser no reason to close the gate, so `_wm` (neutral) and `_wmh` (hold-
    biased, bg = -2, g0 ~ 0.12) are run as SEPARATE arms and reported separately.
    If only `_wmh` works, the honest statement is "holding needs an inductive bias
    towards holding", not "gating solves it".

Readout is pinned to the last 15 steps (RATE_WINDOW=15) -- the honest instrument
from 2.4d. The whole-trial average is never used here.

usage
    python d2_wm.py --arms point_slow,dual --taus 0.90,0.97,0.99,0.995,0.999 \
                    --gaps 2,10,40,80,160,320 --seeds 0,1,2,3,4
    python d2_wm.py --arms point_wm,point_wmh,dual_wm,dual_wmh --taus 0.97
"""
import argparse
import json
import os
import statistics as st
import time

import torch
import torch.nn.functional as F

import d2_comp_snn as M

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HONEST_WINDOW = 15          # last 15 steps of the trial only
HORIZON_ACC = 45.0          # no-memory floor is 1/C = 33.3: the template is
                            # visible in the readout window, only the context
                            # has to be carried across the gap

ARMS = [("point", "-"), ("point_slow", "-"), ("point_wm", "-"),
        ("point_wmh", "-"), ("dual", "add"), ("dual", "thr"),
        ("dual_wm", "thr"), ("dual_wmh", "thr"), ("alif", "-")]


def cfg_for(couple, tau_a=0.97, nh=96, ntrain=4000, ntest=1000):
    return dict(C=3, M=3, nh=nh, ntrain=ntrain, ntest=ntest, epochs=20, bs=128,
                lr=0.004, gamma=1.0, tau_s=0.85, tau_a=tau_a, kappa=0.8,
                alif_rho=0.9, alif_beta=0.5, eval_every=20, data_seed=1,
                couple=couple, ncue=3)


def run_arm(arm, couple, gap, seed, tau_a, want_gate, nh=96):
    """Train one cell exactly as M.run does, then read the gate off that model.

    Re-implemented here (not `M.run`) because the gate diagnostic needs the
    trained model, and training a second copy to read one scalar would double the
    cost of every gated cell. The loop below is a line-for-line copy of M.run --
    same seeding order, same permutation draw, same optimiser -- so these
    accuracies are comparable with the rest of the D2 tables. Verified: the leaky
    arm reproduces the pre-change reference bit for bit (point_slow gap=80 seed 0:
    acc 43.5, ctx_probe 41.5, identical to results_ro_sweep).
    """
    torch.manual_seed(seed)
    cfg = cfg_for(couple, tau_a, nh=nh)
    (xtr, ytr, _), (xte, yte, cte), T, NIN, nout = M.make_data(
        C=cfg["C"], M=cfg["M"], gap=gap, ncue=cfg["ncue"], ntrain=cfg["ntrain"],
        ntest=cfg["ntest"], seed=cfg["data_seed"])
    probe_t = cfg["ncue"] + gap - 1
    model = M.CompSNN(NIN, cfg["nh"], nout, arm=arm, gamma=cfg["gamma"],
                      tau_s=cfg["tau_s"], tau_a=cfg["tau_a"], kappa=cfg["kappa"],
                      alif_rho=cfg["alif_rho"], alif_beta=cfg["alif_beta"],
                      couple=couple).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    xtr, ytr = xtr.to(DEV), ytr.to(DEV)
    xte, yte = xte.to(DEV), yte.to(DEV)
    n = xtr.size(0)
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, cfg["bs"]):
            idx = perm[i:i + cfg["bs"]]
            opt.zero_grad()
            out, _ = model(xtr[idx])
            F.cross_entropy(out, ytr[idx]).backward()
            opt.step()
    acc, spikes, ctxdec = M.evaluate(model, xte, yte, cte, cfg, probe_t)
    rec = dict(acc=round(acc, 2), ctx_probe=round(ctxdec, 2),
               spikes=round(spikes, 1), T=T, nparam=nparam, tau_a=tau_a)

    if want_gate:
        model.eval()
        with torch.no_grad():
            xb = xtr[:256]
            s = torch.zeros(xb.size(0), cfg["nh"], device=DEV)
            gs, smag = [], []
            for t in range(T):
                g = torch.sigmoid(model.wgate(torch.cat([xb[:, t], s], 1)))
                gs.append(float(g.mean()))
                s = (1 - g) * s + g * torch.tanh(model.wcand(xb[:, t]))
                smag.append(float(s.abs().mean()))
            smag_end = smag[probe_t]
        nc = cfg["ncue"]
        rec.update(gate_cue=round(st.mean(gs[:nc]), 4),
                   gate_gap=round(st.mean(gs[nc:nc + max(1, gap)]), 4),
                   gate_tail=round(st.mean(gs[-10:]), 4),
                   state_at_probe=round(smag_end, 5))

        # --- the two controls that decide what the gate is worth ---------------
        # A gated arm carries ~2.4x the parameters of the leaky one (wgate and
        # wcand are full input->hidden layers), so a higher accuracy is NOT
        # evidence of gating: it can be capacity, or just the extra input->state
        # pathway. Freezing the gate at its own mean keeps the pathway and the
        # parameters but removes the input-dependence (i.e. removes "hold when
        # nothing is happening"); freezing it OPEN removes memory entirely.
        # If acc_gatefreeze collapses to chance, the learned gate is doing the
        # work. If acc_gateone stays high, the task is not cue-dependent and the
        # whole comparison is void.
        def _eval_const(c):
            """Re-evaluate with the gate frozen at the scalar c (all units, all
            steps). Keeps wgate unused and wcand active, so the extra capacity
            and the input->state pathway survive; only the input-dependence of
            the gate is removed."""
            def _write(s_, xt_, Ic_):
                g_ = torch.full_like(s_, c)
                return (1 - g_) * s_ + g_ * torch.tanh(model.wcand(xt_))
            keep = model._write
            model._write = _write
            try:
                a_, _, p_ = M.evaluate(model, xte, yte, cte, cfg, probe_t)
            finally:
                model._write = keep
            return round(a_, 2), round(p_, 2)

        a_fr, p_fr = _eval_const(st.mean(gs))
        a_one, p_one = _eval_const(1.0)
        rec.update(acc_gatefreeze=a_fr, probe_gatefreeze=p_fr,
                   acc_gateone=a_one, probe_gateone=p_one)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", default="2,10,40,80,160,320")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--taus", default="0.97", help="tau_a values to sweep")
    ap.add_argument("--nh", type=int, default=96,
                    help="hidden width; raise it for the capacity control")
    ap.add_argument("--arms", default="")
    ap.add_argument("--outdir", default="results_d2_wm")
    a = ap.parse_args()

    gaps = [int(x) for x in a.gaps.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    taus = [float(x) for x in a.taus.split(",") if x.strip()]
    want = [x for x in a.arms.split(",") if x.strip()]
    arms = [p for p in ARMS if not want or p[0] in want]
    os.makedirs(a.outdir, exist_ok=True)

    M.RATE_WINDOW = HONEST_WINDOW
    todo = [(arm, cp, tau, gap, s) for arm, cp in arms for tau in taus
            for gap in gaps for s in seeds]
    print(f"[d2_wm] window={HONEST_WINDOW} cells={len(todo)} "
          f"arms={[x[0] for x in arms]} taus={taus} gaps={gaps} seeds={seeds}",
          flush=True)

    for arm, couple, tau, gap, seed in todo:
        tag = (f"W_{arm}_{couple}_ta{tau:g}_n{a.nh}_g{gap}_s{seed}"
               .replace("-", "x"))
        out = os.path.join(a.outdir, tag + ".json")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        t0 = time.time()
        rec = {"arm": arm, "couple": couple, "gap": gap, "seed": seed,
               "tau_a": tau, "rate_window": HONEST_WINDOW}
        try:
            rec.update(run_arm(arm, couple, gap, seed, tau,
                               want_gate=arm.endswith("wm") or
                               arm.endswith("wmh"), nh=a.nh))
        except Exception as exc:                          # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAIL {tag} {rec['error']}", flush=True)
        rec["elapsed"] = round(time.time() - t0, 1)
        tmp = out + ".part"
        json.dump(rec, open(tmp, "w"), indent=1)
        os.replace(tmp, out)
        print(f"  {tag}: acc={rec.get('acc')} probe={rec.get('ctx_probe')} "
              f"gcue={rec.get('gate_cue')} ggap={rec.get('gate_gap')} "
              f"| {rec['elapsed']} s", flush=True)
    print("DONE d2_wm", flush=True)


if __name__ == "__main__":
    main()

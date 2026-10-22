"""d2_gate_dist.py -- the mean gate value hides the mechanism.

analyze_abl.py reports point_wmh's gate as ~0.43 inside the cue window and
~0.46 inside the silent gap: essentially the SAME value. A single global gate
with those numbers would keep writing during the silence (s <- 0.54 s + 0.46
tanh(Wc*0)), i.e. it would erase the cue, and yet the arm scores 100.0 at
every delay. Freezing the gate at that same mean, keeping all parameters,
collapses it to 33. So the mean cannot be the thing that matters.

The obvious resolution: g is a PER-UNIT gate, and the units SPECIALISE. Some
close to ~0 during the silence (those are the memory slots that hold) and
others stay open (those keep writing). A mean of 0.43 is then the mixture of
two populations, and replacing the distribution by its mean destroys both
roles at once -- which is exactly what the frozen ablation does.

This script measures the distribution rather than the mean:

    per unit u, the time-mean of g_u over the last step of the gap, pooled
    over trials, and then:
      frac_closed  = P(g_u < 0.10)   units that are write-protected
      frac_open    = P(g_u > 0.90)   units that keep writing
      frac_mid     = the rest

If frac_closed ~ 0 and frac_open ~ 0, the specialisation story is wrong and
the 100.0 needs another explanation -- so this is a falsifiable check, not a
decorative histogram.

usage
    python d2_gate_dist.py --seeds 0,1,2 --train-gap 40
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


def cfg_for(couple, tau_a, nh=96):
    return dict(C=3, M=3, nh=nh, ntrain=4000, ntest=1000, epochs=20, bs=128,
                lr=0.004, gamma=1.0, tau_s=0.85, tau_a=tau_a, kappa=0.8,
                alif_rho=0.9, alif_beta=0.5, eval_every=20, data_seed=1,
                couple=couple, ncue=3)


def quantiles(v, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    s = sorted(v)
    n = len(s)
    return [round(s[min(n - 1, max(0, int(q * (n - 1))))], 4) for q in qs]


def one(arm, couple, tau, gap, seed, nh=96, nrep=128):
    torch.manual_seed(seed)
    cfg = cfg_for(couple, tau, nh=nh)
    (xtr, ytr, _), _, _, NIN, nout = M.make_data(
        C=cfg["C"], M=cfg["M"], gap=gap, ncue=cfg["ncue"],
        ntrain=cfg["ntrain"], ntest=cfg["ntest"], seed=cfg["data_seed"])
    model = M.CompSNN(NIN, cfg["nh"], nout, arm=arm, gamma=cfg["gamma"],
                      tau_s=cfg["tau_s"], tau_a=cfg["tau_a"], kappa=cfg["kappa"],
                      alif_rho=cfg["alif_rho"], alif_beta=cfg["alif_beta"],
                      couple=couple).to(DEV)
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

    model.eval()
    with torch.no_grad():
        xb = xtr[:nrep]
        s = torch.zeros(xb.size(0), cfg["nh"], device=DEV)
        gcue, gsil = [], None
        trace = []
        for t in range(xb.size(1)):
            g = torch.sigmoid(model.wgate(torch.cat([xb[:, t], s], 1)))
            if t < cfg["ncue"]:
                gcue.append(g)
            if t == cfg["ncue"] + gap - 1:      # last silent step
                gsil = g
            # gate profile through the silence: WHEN it closes matters as much
            # as whether it closed. A gate that opens for 20 steps and only then
            # shuts has already let the state decay, so the endpoint value says
            # nothing on its own.
            if cfg["ncue"] <= t <= cfg["ncue"] + gap - 1:
                rel = t - cfg["ncue"]
                if rel in (0, 1, 2, 4, 9, 19, 39, 79, 159, gap - 1):
                    trace.append((rel, round(float(g.mean()), 4),
                                  round(sum(float(x) < 0.10 for x in g.mean(0)) /
                                        cfg["nh"], 3)))
            s = (1 - g) * s + g * torch.tanh(model.wcand(xb[:, t]))
        # per-unit time-mean during the cue, and the pooled values at the end
        gc = torch.cat(gcue).mean(0).cpu().tolist()
        gs = (gsil if gsil is not None else g).mean(0).cpu().tolist()
    seen, dedup = set(), []
    for rel, m, f in trace:
        if rel in seen:
            continue
        seen.add(rel)
        dedup.append((rel, m, f))
    return dict(
        arm=arm, seed=seed, gap=gap, nh=cfg["nh"],
        cue_quantiles=quantiles(gc), gap_quantiles=quantiles(gs),
        frac_closed=round(sum(x < 0.10 for x in gs) / len(gs), 3),
        frac_open=round(sum(x > 0.90 for x in gs) / len(gs), 3),
        frac_mid=round(sum(0.10 <= x <= 0.90 for x in gs) / len(gs), 3),
        mean_gap=round(st.mean(gs), 4), mean_cue=round(st.mean(gc), 4),
        trace=dedup,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="point_wmh,point_wm,dual_wmh")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--train-gap", type=int, default=40)
    ap.add_argument("--outdir", default="results_gdist")
    a = ap.parse_args()

    specs = [("point_wm", "-", 0.97), ("point_wmh", "-", 0.97),
             ("dual_wm", "thr", 0.97), ("dual_wmh", "thr", 0.97)]
    want = {x.strip() for x in a.arms.split(",") if x.strip()}
    specs = [s for s in specs if s[0] in want]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    os.makedirs(a.outdir, exist_ok=True)

    print("# 门值的分布（不是均值）\n")
    print("| arm | seed | 线索 g 均值 | 间隙 g 均值 | 间隙 g 分位 5/25/50/75/95 "
          "| g<0.1 比例 | g>0.9 比例 |")
    print("|---|---|---|---|---|---|---|")
    for arm, couple, tau in specs:
        for sd in seeds:
            t0 = time.time()
            r = one(arm, couple, tau, a.train_gap, sd)
            r["elapsed"] = round(time.time() - t0, 1)
            q = "/".join(f"{x:.2f}" for x in r["gap_quantiles"])
            print(f"| {arm} | {sd} | {r['mean_cue']:.3f} | {r['mean_gap']:.3f} | "
                  f"{q} | {r['frac_closed']:.2f} | {r['frac_open']:.2f} |", flush=True)
            prof = " ".join(f"{rel}:{m:.2f}/{f:.2f}" for rel, m, f in r["trace"])
            print(f"|   └ 门轨迹（间隙内第几步 → 均值g / 已关比例） | | | | | |")
            print(f"|   └ {prof} | | | | | |", flush=True)
            out = os.path.join(a.outdir, f"GD_{arm}_s{sd}.json")
            tmp = out + ".part"
            json.dump(r, open(tmp, "w"), indent=1)
            os.replace(tmp, out)
    print("\nDONE gdist", flush=True)


if __name__ == "__main__":
    main()

"""d2_comp_snn.py -- dual-compartment LIF vs a ladder of matched controls.

Scientific question
-------------------
exp2 reported 100.0% (dual-compartment) vs 89.6% (point LIF) on a synthetic
context-XOR task. Two confounds make that comparison uninformative for a journal:

  (1) CEILING. Two classes, 500 test samples, and the reported number is 100%,
      so the effect has no headroom and cannot scale.
  (2) CONSTRUCT CONFOUND. The dual model differs from the point model in three
      ways at once: a separate membrane (compartment), a slow time constant
      (temporal separation), and a nonlinear somatodendritic coupling. Any of
      the three could produce the gain.

This script fixes both.

  Task family: C contexts x M templates, label = ctx*M + tmpl, C*M classes.
    C=3, M=3 -> 9 classes, so accuracy cannot saturate by chance-guessing a
    binary split. A silent gap of length `gap` separates the context cue from
    the sensory template, and `gap` is swept: it is the variable that a slow
    apical trace is supposed to protect, so the *slope* of accuracy-vs-gap is
    the real dependent variable.

  Control ladder (all arms share identical parameter counts and readout):
    point        fast soma, context summed into the same current
    point_slow   slow *synaptic* current added to the soma (slow trace, no
                 compartment, no nonlinearity)  -> isolates temporal separation
    point_gate   fast context, nonlinear multiplicative gate
                 -> isolates nonlinearity
    dual         slow apical membrane + kappa*sigmoid coupling   (ours)
    alif         adaptive threshold (slow somatic state)         -> the null
                 hypothesis for 'any slow state helps'
    dual_alif    both
  kappa / tau_a / tau_s are swept on `dual` to locate the mechanism.

Labels are (context, template) pairs, so answering requires holding the cue
across the gap; no arm can solve it by integrating the sensory template alone.

Usage
-----
  python d2_comp_snn.py --arm dual --gap 20 --seed 0 --out r.json
"""
import argparse, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------- task
# Readout window. 0 = average the firing rate over the WHOLE trial (historical).
# A positive value averages only the last RATE_WINDOW steps.
#
# Why this exists: the trial-averaged rate contains the cue response emitted at
# t < ncue, so a model can solve a long-gap trial from the time average alone
# without carrying anything across the silent gap. That makes accuracy at
# large gap uninformative about memory. Restricting the readout to the last
# few steps forces the decision to use only what survived the gap.
RATE_WINDOW = 0


def make_data(C=3, M=3, gap=20, ncue=3, nsens=15, nin_g=30, ntrain=5000,
              ntest=1000, p_sens=0.85, p_bg=0.02, seed=1):
    """Poisson spike trains. Context = one of C cue channels at t < ncue.
    Template = one of M disjoint input groups active for nsens steps after the
    gap. Background spikes at rate p_bg over all sensory channels fill the gap.
    Label = ctx*M + tmpl  (C*M classes)."""
    g = torch.Generator().manual_seed(seed)
    T = ncue + gap + nsens
    NIN = C + M * nin_g
    nout = C * M

    def mk(n):
        ctx = torch.randint(0, C, (n,), generator=g)
        tmpl = torch.randint(0, M, (n,), generator=g)
        x = (torch.rand(n, T, NIN, generator=g) < p_bg).float()
        # silent window: no background during the cue (clean encoding)
        x[:, :ncue, :] = 0.0
        for c in range(C):
            m = ctx == c
            x[m, :ncue, c] = 1.0
        for tid in range(M):
            m = tmpl == tid
            lo = C + tid * nin_g
            hi = lo + nin_g
            n_m = int(m.sum())
            x[m, ncue + gap:, lo:hi] = (torch.rand(n_m, nsens, nin_g,
                                                   generator=g) < p_sens).float()
        y = (ctx * M + tmpl).long()
        return x, y, ctx
    return mk(ntrain), mk(ntest), T, NIN, nout


# --------------------------------------------------------------------- model
class CompSNN(nn.Module):
    """One code path for every arm; arms differ only in dynamics/constants.

    couple mode is a first-class factor, because it decides whether the
    apical compartment is *functionally distinct* from a slow synaptic current:
      add : Vs += kappa*sigmoid(Va)        <- exp2's formulation. Equals a slow
            additive current, so it is NOT distinguishable from `point_slow`
            except by the saturating nonlinearity. Kept as a control.
      mul : Vs += I*(1 + kappa*sigmoid(Va)) <- multiplicative gain on the
            feedforward drive: context scales the response to the stimulus.
      thr : thr_eff = thr - kappa*sigmoid(Va) <- pre-depolarisation changes
            excitability. This is the faithful reading of 'plateau potential
            pre-depolarises the soma' and cannot be mimicked by an input current.
    """

    def __init__(self, nin, nh, nout, arm="dual", gamma=1.0,
                 tau_s=0.85, tau_a=0.97, kappa=0.8, alif_rho=0.9, alif_beta=0.5,
                 gain=4.0, couple="thr"):
        super().__init__()
        self.arm, self.gamma, self.couple = arm, gamma, couple
        self.win = nn.Linear(nin, nh)
        self.wctx = nn.Linear(nin, nh)
        self.rd = nn.Linear(nh, nout)
        self.nh = nh
        self.tau_s, self.tau_a = tau_s, tau_a
        # `_wm` arms replace the leaky integrator with a WRITE-GATED memory:
        #     g  = sigmoid(Wg [x_t, s] + bg)
        #     s  = (1 - g) s + g tanh(Wc x_t)
        # Everything else is held fixed, so point_wm vs point_slow isolates the
        # write rule and nothing else. Why it matters: pure leak makes the
        # horizon a function of tau_a alone (0.97^40 ~= 0.30, and the measured
        # horizon is ~40 steps), so no architecture can beat 40 without either a
        # longer tau_a or a way to STOP writing. A trained gate can hold a value
        # indefinitely by closing during the gap, which is what a biological
        # persistent state does and what the leaky version cannot do.
        #
        # The extra modules are created ONLY for the _wm arms, after win/wctx/rd.
        # nn.Linear draws from the global RNG, so creating them unconditionally
        # would shift every other arm's initial weights and silently change every
        # historical number in this file. Conditional creation keeps the old arms
        # bit-identical.
        self.wm = arm in ("point_wm", "dual_wm", "point_wmh", "dual_wmh")
        self.wm_hold = arm.endswith("wmh")
        self.kappa = 0.0 if arm in ("point", "point_slow", "point_wm",
                                    "point_wmh") else kappa
        self.gate = arm in ("point_gate", "dual", "dual_alif")
        self.slow = arm in ("point_slow", "dual", "dual_alif", "point_wm",
                            "dual_wm", "point_wmh", "dual_wmh")
        self.comp = arm in ("dual", "dual_alif", "dual_wm", "dual_wmh")
        self.alif = arm in ("alif", "dual_alif")
        if self.wm:
            self.wgate = nn.Linear(nin + nh, nh)
            self.wcand = nn.Linear(nin, nh)
            if self.wm_hold:
                # Hold-biased init: g0 = sigmoid(-2) ~ 0.12, i.e. the state
                # persists by default and the gate has to LEARN to open for the
                # cue. A neutral init (bg = 0) gave a gate that stayed wide open
                # (measured: 0.61 in the cue window, 0.63 in the gap) and an
                # accuracy at chance, because the loss offers no gradient that
                # specifically asks for "close the gate". Biasing the default is
                # standard for gated RNNs and is a DIFFERENT arm, not a fix.
                self.wgate.bias.data.fill_(-2.0)
        self.rho, self.beta, self.gain = alif_rho, alif_beta, gain
        self.thr = 1.0

    def _write(self, s, xt, Ic):
        """One step of the slow state.

        Default: leaky integration, s <- tau_a * s + Ic. That is the historical
        dynamics and it fixes the horizon at ~1/(1-tau_a) steps regardless of
        what the network wants.

        `_wm` arms: gated write, s <- (1-g) s + g tanh(Wc xt) with
        g = sigmoid(Wg [xt, s]). Closing the gate (g -> 0) HOLDS the state
        instead of decaying it, so a context written during the cue can survive
        an arbitrarily long gap. This is the only change; tau_a is not used.
        """
        if not self.wm:
            return self.tau_a * s + Ic
        g = torch.sigmoid(self.wgate(torch.cat([xt, s], 1)))
        return (1 - g) * s + g * torch.tanh(self.wcand(xt))

    def forward(self, x, probe=False):
        B, T, _ = x.shape
        Vs = torch.zeros(B, self.nh, device=x.device)
        Va = torch.zeros(B, self.nh, device=x.device)   # apical / slow state
        Bt = torch.zeros(B, self.nh, device=x.device)   # adaptation variable
        rate = torch.zeros(B, self.nh, device=x.device)
        W = int(RATE_WINDOW)
        ratew = torch.zeros(B, self.nh, device=x.device)
        nspk = torch.zeros(B, device=x.device)
        vhist = [] if probe else None
        for t in range(T):
            I = self.win(x[:, t])
            Ic = self.wctx(x[:, t])
            thr = torch.full_like(Vs, self.thr)
            if self.comp:                       # slow apical membrane
                Va = self._write(Va, x[:, t], Ic)
                u = torch.sigmoid(self.gain * (Va - 0.5))
                if self.couple == "add":
                    drive, Igain = self.kappa * u, 1.0
                elif self.couple == "mul":
                    drive, Igain = torch.zeros_like(I), 1.0 + self.kappa * u
                else:                           # thr
                    drive, Igain = torch.zeros_like(I), 1.0
                    thr = self.thr - self.kappa * u
            elif self.slow:                     # slow current, single membrane
                Va = self._write(Va, x[:, t], Ic)
                drive, Igain = (self.kappa * Va if self.kappa > 0 else Va), 1.0
            else:                               # fast context current
                Va = Ic
                drive, Igain = Ic, 1.0
            if self.gate and not self.comp:
                I = I * torch.sigmoid(self.gain * drive)
                drive = torch.zeros_like(I)
            Vs = self.tau_s * Vs + I * Igain + drive
            if self.alif:
                Vs = Vs - self.beta * Bt
            s = torch.sigmoid(self.gamma * (Vs - thr))
            Vs = Vs * (1 - s)
            if self.alif:
                Bt = self.rho * Bt + (1 - self.rho) * s
            rate = rate + s
            if W and t >= T - W:
                ratew = ratew + s
            nspk = nspk + s.sum(1)
            if probe:
                vhist.append(Va.clone())
        rate = (ratew / W) if W else (rate / T)
        out = self.rd(rate)
        aux = dict(spikes=float(nspk.mean()), vhist=vhist,
                   slow_state=Va) if probe else None
        return out, aux


# --------------------------------------------------------------------- train
def run(arm, gap, seed, cfg):
    torch.manual_seed(seed)
    (xtr, ytr, _), (xte, yte, cte), T, NIN, nout = make_data(
        C=cfg["C"], M=cfg["M"], gap=gap, ncue=cfg["ncue"], ntrain=cfg["ntrain"],
        ntest=cfg["ntest"], seed=cfg["data_seed"])
    probe_t = cfg["ncue"] + gap - 1        # last silent step: has the trace survived?
    model = CompSNN(NIN, cfg["nh"], nout, arm=arm, gamma=cfg["gamma"],
                    tau_s=cfg["tau_s"], tau_a=cfg["tau_a"], kappa=cfg["kappa"],
                    alif_rho=cfg["alif_rho"], alif_beta=cfg["alif_beta"],
                    couple=cfg["couple"]).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    xtr, ytr = xtr.to(DEV), ytr.to(DEV)
    xte, yte = xte.to(DEV), yte.to(DEV)
    n = xtr.size(0)
    hist = []
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, cfg["bs"]):
            idx = perm[i:i + cfg["bs"]]
            opt.zero_grad()
            out, _ = model(xtr[idx])
            F.cross_entropy(out, ytr[idx]).backward()
            opt.step()
        if (ep + 1) % cfg["eval_every"] == 0 or ep == cfg["epochs"] - 1:
            acc, spikes, ctxdec = evaluate(model, xte, yte, cte, cfg, probe_t)
            hist.append(dict(ep=ep + 1, acc=round(acc, 2), spikes=round(spikes, 1),
                             ctx_probe=round(ctxdec, 2)))
            print(f"  [{arm} gap{gap} s{seed}] ep{ep+1} acc={acc:.2f} "
                  f"spk={spikes:.1f} ctxprobe={ctxdec:.2f}", flush=True)
    final = hist[-1]
    return dict(arm=arm, gap=gap, seed=seed, nparam=nparam, T=T, nout=nout,
                acc=final["acc"], spikes=final["spikes"], ctx_probe=final["ctx_probe"],
                hist=hist, cfg=cfg)


@torch.no_grad()
def evaluate(model, xte, yte, cte, cfg, probe_t):
    model.eval()
    # make_data returns CPU tensors; the model lives on DEV. CPU-only smoke
    # tests cannot expose this, so the move is done here explicitly.
    yte, cte = yte.to(DEV), cte.to(DEV)
    correct, spk = 0, 0.0
    feats, labels = [], []
    nb = 0
    for i in range(0, xte.size(0), 250):
        xb = xte[i:i + 250]
        out, aux = model(xb, probe=True)
        correct += (out.argmax(1) == yte[i:i + 250]).sum().item()
        spk += aux["spikes"]
        nb += 1
        # probe the slow state at the LAST SILENT STEP, not the time average:
        # even the point neuron look like it remembered the context.
        feats.append(aux["vhist"][probe_t])
        labels.append(cte[i:i + 250])
    acc = correct / xte.size(0) * 100
    # linear probe: can the context be decoded from the mean slow state?
    f = torch.cat(feats).float()
    y = torch.cat(labels)
    p = probe_acc(f, y, cfg["C"])
    return acc, spk / nb, p


def probe_acc(f, y, nclass, epochs=200):
    """Linear probe. Called from inside @torch.no_grad(), so grad must be
    re-enabled explicitly for the probe's own parameters."""
    import torch.nn.functional as Fn
    with torch.enable_grad():
        y = y.to(f.device)
        mu, sd = f.mean(0, keepdim=True), f.std(0, keepdim=True) + 1e-6
        f = (f - mu) / sd
        n = f.size(0)
        ntr = int(0.8 * n)
        lin = nn.Linear(f.size(1), nclass).to(f.device)
        o = torch.optim.Adam(lin.parameters(), lr=1e-2)
        for _ in range(epochs):
            o.zero_grad()
            Fn.cross_entropy(lin(f[:ntr]), y[:ntr]).backward()
            o.step()
        with torch.no_grad():
            return (lin(f[ntr:]).argmax(1) == y[ntr:]).float().mean().item() * 100


DEF = dict(C=3, M=3, nh=128, ntrain=5000, ntest=1000, epochs=30, bs=128, lr=4e-3,
           gamma=1.0, tau_s=0.85, tau_a=0.97, kappa=0.8, alif_rho=0.9,
           alif_beta=0.5, eval_every=5, data_seed=1, couple="thr", ncue=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dual")
    ap.add_argument("--gap", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=0.8)
    ap.add_argument("--tau_a", type=float, default=0.97)
    ap.add_argument("--tau_s", type=float, default=0.85)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--nh", type=int, default=128)
    ap.add_argument("--ntrain", type=int, default=5000)
    ap.add_argument("--ntest", type=int, default=1000)
    ap.add_argument("--couple", default="thr", choices=["add","mul","thr"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = dict(DEF)
    cfg.update(kappa=a.kappa, tau_a=a.tau_a, tau_s=a.tau_s, epochs=a.epochs,
               nh=a.nh, couple=a.couple, ntrain=a.ntrain, ntest=a.ntest)
    t0 = time.time()
    res = run(a.arm, a.gap, a.seed, cfg)
    res["elapsed"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("WROTE", a.out, res["acc"], res["ctx_probe"], res["elapsed"], "s")


if __name__ == "__main__":
    main()

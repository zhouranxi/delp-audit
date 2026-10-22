"""bench_cl.py -- unified, information-budget-matched continual-learning benchmark.

Purpose
-------
Turn the ad-hoc comparisons of exp4-exp12 into a *measurement instrument*:

  * every protection method is implemented in its canonical published form
    (EWC-online / EWC-multi-anchor / SI / MAS / LwF / PackNet / DeLP);
  * lambda is swept over a fixed grid and NEVER selected on the test set: each
    run is one point of a frontier, selection rules are applied in the analysis;
  * the accounting identity  a_bar_T = ell + (T-1)/T * BWT  is logged per run
    together with the quantities hypothesised to drive it:
        A_t   protected mass      = lambda * sum_i Omega_i        (stability axis)
        O_t   protection overlap  = cos(Omega_old, F_cur)          (plasticity cost)
        rho_t plasticity ratio    = |g_pen| / (|g_CE| + |g_pen|)   (mechanism probe)
  * task count T in {5, 10, 20} on NESTED permutation sets, so T is a controlled
    factor and all arms are paired on the same seeds.

One JSON per run -> results/<tag>.json ; merge with bench_analyze.py.
"""
import argparse, json, math, os, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = dict(nh=256, ntrain=10000, epochs=2, bs=256, lr=1e-3,
           k_ratio=4, nfish=300, nusage=500)
XI = 0.1


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_mnist(root="./data"):
    from torchvision import datasets
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    return (tr.data.float().view(-1, 784) / 255.0, tr.targets,
            te.data.float().view(-1, 784) / 255.0, te.targets)


def get_perms(seed, nmax=20):
    g = torch.Generator().manual_seed(100 + seed)
    return [torch.randperm(784, generator=g) for _ in range(nmax)]


def permute(x, p):
    return x[:, p]


def kwta(h, k):
    idx = h.topk(k, dim=1).indices
    m = torch.zeros_like(h)
    m.scatter_(1, idx, 1.0)
    return h * m


class MLP(nn.Module):
    def __init__(self, nh=256, k=64, nhead=1):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.out = nn.Linear(nh, 10) if nhead == 1 else nn.ModuleList(
            [nn.Linear(nh, 10) for _ in range(nhead)])
        self.k, self.multi = k, (nhead > 1)
    def head(self, i):
        return self.out[i] if self.multi else self.out
    def forward(self, x, ctx=None, head=0):
        h = kwta(torch.sigmoid(self.ff(x)), self.k)
        return self.head(head)(h), None


# Hidden-layer gain. The dendrite gate is t * sigmoid(d), and sigmoid lands near
# 0.5 at init, so the dendrite arm starts with a hidden vector roughly half the
# scale of a plain MLP's. kwta() is scale-equivariant (kwta(a*h) == a*kwta(h),
# same mask), so this one scalar is a free re-parameterisation rather than extra
# capacity. exp6's published lambda grid was tuned without it, which makes the
# dendritic arm a systematically attenuated baseline -- HGAIN lets the
# alpha-calibrated arm test whether D1's family grouping is a mechanism finding
# or a scale artefact. 1.0 keeps every earlier result bit-for-bit reproducible.
HGAIN = 1.0


class ADNet(nn.Module):
    def __init__(self, nh=256, k=64, nseg=5, nctx=5):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.dend = nn.Linear(nctx, nh * nseg, bias=False)
        self.out = nn.Linear(nh, 10)
        self.nseg, self.nh, self.k = nseg, nh, k
    def forward(self, x, ctx, head=0):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh)
        d, seg = u.max(dim=1)
        h = t if HGAIN == 1.0 else t * HGAIN
        return self.out(kwta(h * torch.sigmoid(d), self.k)), seg
    def protect_grads(self, task_id):
        if self.dend.weight.grad is None:
            return
        W = self.dend.weight.grad.view(self.nseg, self.nh, -1)
        m = torch.zeros(self.nseg, 1, 1, device=W.device)
        m[min(task_id, self.nseg - 1)] = 1.0
        W.mul_(m)


CTX_ARMS = ("delp", "ad_only")


def P(model):
    d = {"ff.w": model.ff.weight, "ff.b": model.ff.bias}
    if getattr(model, "multi", False):
        for i, h in enumerate(model.out):
            d[f"out{i}.w"] = h.weight
            d[f"out{i}.b"] = h.bias
    else:
        d["out.w"] = model.out.weight
        d["out.b"] = model.out.bias
    return d


def build(arm, ntask, cfg, heads=1):
    k = cfg["nh"] // cfg["k_ratio"]
    if arm in CTX_ARMS:
        return ADNet(cfg["nh"], k, nseg=ntask, nctx=ntask)
    return MLP(cfg["nh"], k, nhead=heads)


def fwd(model, arm, xb, ctx=None, head=0):
    return model(xb, ctx) if arm in CTX_ARMS else model(xb, head=head)


# ------------------------------------------------------------------ importance
def fisher_true_label(model, arm, xs, ctx, nfish, usage=False):
    model.eval()
    Pd = P(model)
    Fd = {k: torch.zeros_like(v) for k, v in Pd.items()}
    n = min(nfish, xs.size(0))
    for i in range(n):
        out, _ = fwd(model, arm, xs[i:i + 1], ctx)
        model.zero_grad()
        F.log_softmax(out, 1).max(1).values.sum().backward()
        for k, v in Pd.items():
            Fd[k] += v.grad.detach() ** 2
    for k in Fd:
        Fd[k] /= n
    o = None
    if usage:
        o = torch.zeros(model.nh, device=xs.device)
        with torch.no_grad():
            for i in range(0, min(500, xs.size(0)), 250):
                xb = xs[i:i + 250]
                c = ctx.expand(xb.size(0), -1)
                t = model.ff(xb)
                u = model.dend(c).view(-1, model.nseg, model.nh)
                d, _ = u.max(dim=1)
                h = kwta(t * torch.sigmoid(d), model.k)
                o += (h > 0).float().sum(0)
        o /= min(500, xs.size(0))
    return Fd, o


def importance_mas(model, arm, xs, ctx, ngrad=2000, bs=250):
    model.eval()
    Pd = P(model)
    Od = {k: torch.zeros_like(v) for k, v in Pd.items()}
    nb = 0
    for i in range(0, min(ngrad, xs.size(0)), bs):
        xb = xs[i:i + bs]
        c = ctx.expand(xb.size(0), -1)
        out, _ = fwd(model, arm, xb, c)
        model.zero_grad()
        (0.5 * out.pow(2).sum(dim=1)).mean().backward()
        for k, v in Pd.items():
            Od[k] += v.grad.detach().abs()
        nb += 1
    for k in Od:
        Od[k] /= nb
    return Od


def cos_sim(a, b):
    num = float(sum((a[k] * b[k]).sum() for k in a if k in b))
    na = math.sqrt(sum(float((a[k] ** 2).sum()) for k in a))
    nb = math.sqrt(sum(float((b[k] ** 2).sum()) for k in b))
    return num / (na * nb + 1e-12)


def mass(Om):
    return float(sum(float(v.sum()) for v in Om.values()))


# ------------------------------------------------------------------ training
def train_one(arm, model, opt, xs, ys, ctx, phase, anchors, Omega, theta_ref, lam,
              cfg, lwf_model=None, lwf_lam=0.0, si=False, probe=False):
    """Train one task. Returns (diag, si_path) where si_path is the path integral."""
    Pd = P(model)
    dl = DataLoader(TensorDataset(xs, ys), batch_size=cfg["bs"], shuffle=True,
                    drop_last=True)
    si_path = {k: torch.zeros_like(v) for k, v in Pd.items()} if si else None
    diag = {}
    # NOTE: at the first batch of a task theta == theta*, so the penalty gradient
    # is exactly zero and rho would be 0 by construction. Probe one epoch later,
    # when the current task has actually started to pull the weights away.
    probe_ep = 1 if cfg["epochs"] > 1 else 0
    first = probe
    for ep in range(cfg["epochs"]):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            c = ctx.expand(xb.size(0), -1)
            pre = {k: v.detach().clone() for k, v in Pd.items()} if si else None
            opt.zero_grad()
            out, _ = fwd(model, arm, xb, c, head=phase)
            loss = F.cross_entropy(out, yb)
            if lwf_model is not None and lwf_lam > 0:
                with torch.no_grad():
                    o_t, _ = fwd(lwf_model, arm, xb, c, head=phase)
                loss = loss + lwf_lam * F.kl_div(F.log_softmax(out, 1),
                                                 F.softmax(o_t, 1),
                                                 reduction="batchmean")
            pen = None
            if lam > 0:
                if Omega is not None:
                    pen = sum((Omega[k] * (Pd[k] - theta_ref[k]) ** 2).sum() for k in Pd)
                if anchors:
                    p2 = sum(sum((a["omega"][k] * (Pd[k] - a["theta"][k]) ** 2).sum()
                                 for k in Pd) for a in anchors)
                    pen = p2 if pen is None else pen + p2
            if pen is not None:
                loss = loss + lam * pen
            if first and ep >= probe_ep and pen is not None and torch.is_tensor(pen):
                g_pen = torch.autograd.grad(lam * pen, list(Pd.values()),
                                            retain_graph=True, allow_unused=True)
                g_tot = torch.autograd.grad(loss, list(Pd.values()), retain_graph=False,
                                            allow_unused=True)
                nce = math.sqrt(sum(float((g ** 2).sum()) for g in g_tot if g is not None))
                npe = math.sqrt(sum(float((g ** 2).sum()) for g in g_pen if g is not None))
                for p, g in zip(Pd.values(), g_tot):
                    p.grad = None if g is None else g.detach().clone()
                diag = dict(grad_ce=round(nce, 6), grad_pen=round(npe, 6),
                            rho=round(npe / (nce + npe + 1e-12), 5))
                first = False
                opt.step()
            else:
                loss.backward()
                opt.step()
            if si:
                g = {k: (v.grad.detach().clone() if v.grad is not None else None)
                     for k, v in Pd.items()}
                for k in Pd:
                    if g[k] is not None:
                        si_path[k] += (-g[k]) * (Pd[k].detach() - pre[k])
    return diag, si_path


def si_finalize(si_path, theta_start, Pd):
    out = {}
    for k in Pd:
        out[k] = si_path[k] / ((Pd[k].detach() - theta_start[k]) ** 2 + XI)
    return out


def packnet_task(model, opt, xs, ys, ctx, phase, frozen, prune, cfg):
    Pd = P(model)
    dl = DataLoader(TensorDataset(xs, ys), batch_size=cfg["bs"], shuffle=True, drop_last=True)
    for xb, yb in dl:
        xb, yb = xb.to(DEV), yb.to(DEV)
        opt.zero_grad()
        out, _ = model(xb, head=phase)
        F.cross_entropy(out, yb).backward()
        for k, v in Pd.items():
            if v.grad is not None:
                v.grad.mul_(1 - frozen[k])
        opt.step()
    # per-tensor magnitude percentile among free parameters (canonical PackNet)
    surv = {}
    for k, v in Pd.items():
        free = (frozen[k] == 0)
        nfree = int(free.sum())
        s = torch.ones_like(v)
        if nfree > 0 and prune > 0:
            mag = v.detach().abs()[free].flatten()
            nth = max(1, int(prune * mag.numel()))
            thr = mag.sort().values[nth - 1]
            keep = (v.detach().abs() > thr).float()
            s = torch.where(free, keep, torch.ones_like(v))
        surv[k] = s
    with torch.no_grad():
        for k, v in Pd.items():
            v.mul_(surv[k]).add_(torch.randn_like(v) * 0.01 * (1 - surv[k]))
    for ep in range(cfg["epochs"]):
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            out, _ = model(xb, head=phase)
            F.cross_entropy(out, yb).backward()
            for k, v in Pd.items():
                if v.grad is not None:
                    v.grad.mul_(surv[k] * (1 - frozen[k]))
            opt.step()
    with torch.no_grad():
        for k, v in Pd.items():
            frozen[k] = torch.clamp(frozen[k] + surv[k], max=1.0)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def eval_all(model, arm, data, perms, ntask, phase):
    _, _, xte, yte = data
    model.eval()
    acc = {}
    for c in range(phase + 1):
        xt = permute(xte, perms[c])
        correct, n = 0, 0
        for i in range(0, xt.size(0), 2000):
            xb = xt[i:i + 2000].to(DEV)
            if arm in CTX_ARMS:
                ctx = F.one_hot(torch.full((xb.size(0),), c, device=DEV), ntask).float()
                o, _ = model(xb, ctx)
            else:
                o, _ = model(xb, head=c)
            correct += (o.argmax(1) == yte[i:i + 2000].to(DEV)).sum().item()
            n += xb.size(0)
        acc[(phase, c)] = correct / n * 100
    return acc


def finish(acc, ntask):
    bwt = sum(acc[(ntask - 1, c)] - acc[(c, c)] for c in range(ntask - 1)) / max(1, ntask - 1)
    favg = sum(acc[(ntask - 1, c)] for c in range(ntask)) / ntask
    ell = sum(acc[(t, t)] for t in range(ntask)) / ntask
    return dict(bwt=bwt, final_avg=favg, ell=ell,
                identity_resid=abs(favg - (ell + (ntask - 1) / ntask * bwt)))


# ------------------------------------------------------------------ driver
def run(arm, ntask, lam, seed, prune=0.5, cfg=None, probe=True):
    cfg = cfg or CFG
    set_seed(seed)
    data = load_mnist()
    xtr, ytr, xte, yte = data
    perms = get_perms(seed)[:ntask]
    heads = ntask if arm == "packnet" else 1
    model = build(arm, ntask, cfg, heads=heads).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    anchors, Omega, theta_ref = [], None, None
    frozen = None
    acc = {}
    log = dict(A=[], O=[], rho=[], ell=[], mass=[], freed=[], task_time=[], P=[])
    t_all = time.time()

    for phase in range(ntask):
        t0 = time.time()
        xs = permute(xtr[:cfg["ntrain"]], perms[phase]).to(DEV)
        ys = ytr[:cfg["ntrain"]].to(DEV)
        ctx = F.one_hot(torch.full((1,), phase, device=DEV), ntask).float()
        Pd = P(model)

        if lam > 0 and phase > 0 and arm in ("ewc_online", "ewc_multi", "delp", "si", "mas"):
            Fcur, _ = fisher_true_label(model, arm, xs, ctx, cfg["nfish"])
            Om_old = Omega if Omega is not None else (anchors[-1]["omega"] if anchors else None)
            log["O"].append(round(cos_sim(Om_old, Fcur), 5) if Om_old is not None else None)
        else:
            log["O"].append(None)

        if arm == "packnet":
            if frozen is None:
                frozen = {k: torch.zeros_like(v) for k, v in Pd.items()}
            free = float(sum(int((frozen[k] == 0).sum()) for k in Pd)) / \
                   float(sum(v.numel() for v in Pd.values()))
            log["freed"].append(round(free, 5))
            packnet_task(model, opt, xs, ys, ctx, phase, frozen, prune, cfg)
            diag = {}
        else:
            log["freed"].append(None)
            lwf_model = copy.deepcopy(model) if (arm == "lwf" and phase > 0) else None
            if lwf_model is not None:
                for p in lwf_model.parameters():
                    p.requires_grad_(False)
            theta_start = {k: v.detach().clone() for k, v in Pd.items()} if arm == "si" else None
            diag, si_path = train_one(arm, model, opt, xs, ys, ctx, phase, anchors, Omega,
                                      theta_ref, lam, cfg, lwf_model=lwf_model,
                                      lwf_lam=(lam if arm == "lwf" else 0.0),
                                      si=(arm == "si"), probe=(probe and lam > 0))
            if arm == "si":
                Om = si_finalize(si_path, theta_start, Pd)
                Omega = Om if Omega is None else {k: Omega[k] + Om[k] for k in Omega}
                theta_ref = {k: v.detach().clone() for k, v in Pd.items()}
        log["rho"].append(diag.get("rho", 0.0))

        # ---- protection FORCE at the end of the task: the penalty value itself.
        # Unlike A = lam*sum(Omega), this is in loss units and is therefore
        # dimensionally comparable across families (SI's Omega, MAS's Omega and
        # EWC's Fisher do not share units; the penalty they produce does).
        pv = 0.0
        if lam > 0:
            if Omega is not None and theta_ref is not None:
                pv = float(lam * sum(float((Omega[k] * (Pd[k] - theta_ref[k]) ** 2).sum())
                                     for k in Pd))
            if anchors:
                pv += float(lam * sum(float((a["omega"][k] * (Pd[k] - a["theta"][k]) ** 2).sum())
                                      for a in anchors for k in Pd))
        log["P"].append(round(pv, 8))

        # importance / anchor update
        if arm in ("ewc_online", "delp"):
            Fnew, usage = fisher_true_label(model, arm, xs, ctx, cfg["nfish"],
                                            usage=(arm == "delp"))
            if arm == "delp":
                Fnew["ff.w"] = Fnew["ff.w"] * usage.unsqueeze(1)
                Fnew["ff.b"] = Fnew["ff.b"] * usage
                Fnew["out.w"] = Fnew["out.w"] * usage.unsqueeze(0)
            Omega = ({k: v.clone() for k, v in Fnew.items()} if Omega is None
                     else {k: Omega[k] + Fnew[k] for k in Omega})
            theta_ref = {k: v.detach().clone() for k, v in Pd.items()}
        elif arm == "ewc_multi":
            Fnew, _ = fisher_true_label(model, arm, xs, ctx, cfg["nfish"])
            anchors.append({"theta": {k: v.detach().clone() for k, v in Pd.items()},
                            "omega": Fnew})
        elif arm == "mas":
            Om = importance_mas(model, arm, xs, ctx)
            Omega = Om if Omega is None else {k: Omega[k] + Om[k] for k in Omega}
            theta_ref = {k: v.detach().clone() for k, v in Pd.items()}

        if arm in ("ewc_online", "delp", "si", "mas") and Omega is not None:
            m_ = mass(Omega)
            log["mass"].append(round(m_, 6)); log["A"].append(round(lam * m_, 6))
        elif arm == "ewc_multi" and anchors:
            m_ = sum(mass(a["omega"]) for a in anchors)
            log["mass"].append(round(m_, 6)); log["A"].append(round(lam * m_, 6))
        else:
            log["mass"].append(0.0); log["A"].append(0.0)

        acc.update(eval_all(model, arm, data, perms, ntask, phase))
        log["ell"].append(round(acc[(phase, phase)], 2))
        log["task_time"].append(round(time.time() - t0, 2))
        print(f"  s{seed} {arm} T{ntask} lam{lam} t{phase}: "
              + " ".join(f"t{c}={acc[(phase, c)]:.1f}" for c in range(phase + 1))
              + f" | rho={log['rho'][-1]} A={log['A'][-1]:.3g} P={log['P'][-1]:.2g}",
              flush=True)

    out = finish(acc, ntask)
    return dict(arm=arm, ntask=ntask, lam=lam, seed=seed, prune=prune, nparam=nparam,
                bwt=round(out["bwt"], 4), final_avg=round(out["final_avg"], 4),
                ell=round(out["ell"], 4), identity_resid=round(out["identity_resid"], 8),
                acc_matrix=[[round(acc[(p, c)], 2) for c in range(p + 1)] for p in range(ntask)],
                log=log, elapsed=round(time.time() - t_all, 1), cfg=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ntask", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prune", type=float, default=0.5)
    ap.add_argument("--hgain", type=float, default=1.0,
                    help="hidden-layer gain for the dendritic arms; 1.0 = original")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    global HGAIN
    HGAIN = float(a.hgain)
    res = run(a.arm, a.ntask, a.lam, a.seed, a.prune, probe=not a.no_probe)
    res["hgain"] = HGAIN
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # atomic: a killed job must never leave a truncated .json behind, otherwise the
    # resume logic would treat it as done.
    tmp = a.out + ".part"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, a.out)
    print("WROTE", a.out, "| bwt", res["bwt"], "acc", res["final_avg"], "ell", res["ell"],
          "| resid", res["identity_resid"], "|", res["elapsed"], "s")


if __name__ == "__main__":
    main()

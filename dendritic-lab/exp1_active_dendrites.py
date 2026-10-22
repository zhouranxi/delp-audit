"""Pilot 1 (Direction D): Active Dendrites vs point-neuron baseline.
Multi-context sequential MNIST (identity / rot90 / hflip). Metric: forgetting (BWT).
CPU-only. 2026-10-17
"""
import json, time, itertools
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
DEV = "cpu"
K_WTA = 32
EPOCHS, BS, NTRAIN = 2, 256, 10000

def load_mnist():
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
    te = datasets.MNIST("./data", train=False, download=True, transform=tf)
    xtr = tr.data.float().view(-1, 784) / 255.0
    ytr = tr.targets
    xte = te.data.float().view(-1, 784) / 255.0
    yte = te.targets
    return xtr, ytr, xte, yte

def apply_ctx(x, c):
    v = x.view(-1, 1, 28, 28)
    if c == 1: v = torch.rot90(v, 1, [-2, -1])
    elif c == 2: v = torch.flip(v, [-1])
    return v.reshape(-1, 784)

class ActiveDendriteNet(nn.Module):
    def __init__(self, nh=128, nseg=10, nctx=3, nout=10):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.dend = nn.Linear(nctx, nh * nseg)
        self.out = nn.Linear(nh, nout)
        self.nseg, self.nh = nseg, nh
    def forward(self, x, ctx):
        t = self.ff(x)
        u = self.dend(ctx).view(-1, self.nseg, self.nh).max(dim=1).values
        h = t * torch.sigmoid(u)
        h = kwta(h)
        return self.out(h)

class PointNet(nn.Module):
    def __init__(self, nh=128, nctx=3, nout=10):
        super().__init__()
        self.ff = nn.Linear(784, nh)
        self.out = nn.Linear(nh, nout)
    def forward(self, x, ctx):
        h = kwta(torch.sigmoid(self.ff(x)))
        return self.out(h)

def kwta(h, k=K_WTA):
    k = min(k, h.shape[1])
    thr = h.topk(k, dim=1).values[:, -1:]
    return h * (h >= thr).float()

def run(model, xtr, ytr, xte, yte, tag):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    acc = {}   # (phase_done, ctx) -> acc
    for phase, c in enumerate([0, 1, 2]):
        xs, ys = apply_ctx(xtr[:NTRAIN], c), ytr[:NTRAIN]
        dl = DataLoader(TensorDataset(xs, ys), batch_size=BS, shuffle=True)
        model.train()
        for ep in range(EPOCHS):
            for xb, yb in dl:
                opt.zero_grad()
                ctx = F.one_hot(torch.full((xb.size(0),), c), 3).float()
                loss = F.cross_entropy(model(xb, ctx), yb)
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            for cc in [0, 1, 2]:
                xt = apply_ctx(xte, cc)
                correct = 0
                for i in range(0, xt.size(0), 1000):
                    n = min(1000, xt.size(0) - i)
                    ctx = F.one_hot(torch.full((n,), cc), 3).float()
                    pred = model(xt[i:i+1000], ctx).argmax(1)
                    correct += (pred == yte[i:i+1000]).sum().item()
                acc[(phase, cc)] = correct / xt.size(0) * 100
        print(f"[{tag}] after phase {phase} (ctx {c}): " +
              " ".join(f"ctx{cc}={acc[(phase,cc)]:.1f}%" for cc in range(phase+1)), flush=True)
    # BWT: final minus right-after-training
    bwt = sum(acc[(2, c)] - acc[(c, c)] for c in [0, 1]) / 2
    final_avg = sum(acc[(2, c)] for c in [0, 1, 2]) / 3
    return {"final_acc": [round(acc[(2,c)],2) for c in range(3)],
            "final_avg": round(final_avg,2), "bwt_forgetting": round(bwt,2)}

def main():
    t0 = time.time()
    try:
        xtr, ytr, xte, yte = load_mnist()
        src = "MNIST"
    except Exception as e:
        print("MNIST download failed, fallback synthetic:", e)
        g = torch.Generator().manual_seed(0)
        nc = 10; d = 784
        centers = torch.randn(nc, d, generator=g) * 0.3
        def mk(n):
            y = torch.randint(0, nc, (n,), generator=g)
            x = centers[y] + torch.randn(n, d, generator=g) * 0.8
            return x, y
        xtr, ytr = mk(20000); xte, yte = mk(4000)
        src = "synthetic"
    res = {"data": src, "epochs_per_task": EPOCHS, "ntrain_per_task": NTRAIN}
    res["point_baseline"] = run(PointNet(), xtr, ytr, xte, yte, "point")
    res["active_dendrites"] = run(ActiveDendriteNet(), xtr, ytr, xte, yte, "dend")
    res["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open("exp1_result.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

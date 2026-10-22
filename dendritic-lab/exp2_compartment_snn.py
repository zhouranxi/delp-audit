"""Pilot 2 (Direction A): dual-compartment LIF vs single-compartment baseline.
Synthetic temporal context task: context delivered as early distal/apical spikes (t=0..2),
sensory templates arrive t=5..19; label depends on (template, context) pair.
Point neuron: context mixed into soma input with the same fast membrane as sensory.
Dual: context enters apical compartment with SLOW membrane (distal dendrite property),
pre-depolarizing soma (plateau-like). Surrogate gradient training. CPU-only.
"""
import json, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
DEV = "cpu"
T, NIN, NH, NOUT = 40, 100, 128, 2
NTRAIN, NTEST, EPOCHS, BS = 2000, 500, 25, 128
GAMMA = 1.0  # surrogate slope

def sigmoid_sur(x):
    return torch.sigmoid(GAMMA * x)

def make_data():
    g = torch.Generator().manual_seed(1)
    def mk(n):
        ctx = torch.randint(0, 2, (n,), generator=g)
        tmpl_id = torch.randint(0, 2, (n,), generator=g)
        x = torch.zeros(n, T, NIN)
        # context: t=0..2, dedicated channel — neuron 98 for ctx0, 99 for ctx1 (strong)
        for c in range(2):
            m = ctx == c
            x[m, 0:3, 98 + c] = 1.0
        # sensory: t=5..19, dense group coding — template A fires neurons 0..47, B fires 48..95
        for tid in range(2):
            m = tmpl_id == tid
            lo, hi = (0, 48) if tid == 0 else (48, 96)
            x[m, 5:, lo:hi] = (torch.rand(int(m.sum()), T - 5, hi - lo, generator=g) < 0.9).float()
        y = (ctx ^ tmpl_id).long()   # XOR-like: label = context XOR template
        return x, y, ctx, tmpl_id
    return mk(NTRAIN), mk(NTEST)

class CompSNN(nn.Module):
    def __init__(self, dual=True):
        super().__init__()
        self.dual = dual
        self.win = nn.Linear(NIN, NH)
        self.wctx = nn.Linear(NIN, NH)
        self.rd = nn.Linear(NH, NOUT)
        self.tau_s, self.tau_a = 0.85, 0.97   # soma fast, apical slow
        self.thr, self.kappa = 1.0, 0.8
    def forward(self, x):
        B = x.size(0)
        Vs = torch.zeros(B, NH); Va = torch.zeros(B, NH)
        mem = []
        for t in range(T):
            I_s = self.win(x[:, t])                    # sensory + context if not dual
            if self.dual:
                I_a = self.wctx(x[:, t])
                Va = self.tau_a * Va + I_a
                Vs = self.tau_s * Vs + I_s + self.kappa * torch.sigmoid(Va * 4)
            else:
                Vs = self.tau_s * Vs + I_s + self.win2(x[:, t])
            s = sigmoid_sur(Vs - self.thr)
            Vs = Vs * (1 - s)                          # soft reset
            mem.append(s)
        return self.rd(torch.stack(mem, 1).sum(1) / T)
    def pre(self):
        if not hasattr(self, "win2"):
            self.win2 = nn.Linear(NIN, NH)
        return self

def train(model, data, tag):
    (xtr, ytr, _, _), (xte, yte, _, _) = data
    if not model.dual: model.pre()
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    for ep in range(EPOCHS):
        perm = torch.randperm(NTRAIN)
        tot = 0.0
        for i in range(0, NTRAIN, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * idx.numel()
        with torch.no_grad():
            acc = sum((model(xte[j:j+256]).argmax(1) == yte[j:j+256]).sum().item()
                      for j in range(0, NTEST, 256)) / NTEST * 100
        if (ep+1) % 3 == 0 or ep == 0:
            print(f"[{tag}] ep{ep+1} loss={tot/NTRAIN:.4f} test_acc={acc:.1f}%", flush=True)
    return round(acc, 2)

def main():
    t0 = time.time()
    data = make_data()
    res = {"task": "temporal context XOR (ctx pulses t0-2 slow apical vs mixed fast input)",
           "T": T, "epochs": EPOCHS, "ntrain": NTRAIN}
    torch.manual_seed(0)
    res["point_neuron_acc"] = train(CompSNN(dual=False), data, "point")
    torch.manual_seed(0)
    res["dual_compartment_acc"] = train(CompSNN(dual=True), data, "dual")
    res["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    json.dump(res, open("exp2_result.json", "w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

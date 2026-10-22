"""Pilot 3 (Direction B): dendritic error channel (apical feedback alignment) vs BP.
Hidden neurons = two compartments: basal z_b = W1 x (feedforward), apical z_a = V^T delta
(V fixed random -> biologically plausible error channel, no weight transport).
Local plasticity: dW1 = (sigmoid'(z_b) * z_a)^T x ; matches feedback-alignment gradient.
Compare convergence/accuracy vs backprop on MNIST (or synthetic fallback). CPU-only.
"""
import json, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
NH, NTRAIN, NTEST, EPOCHS, BS, LR = 100, 2000, 2000, 30, 128, 0.05

def load_data():
    try:
        from torchvision import datasets, transforms
        tf = transforms.ToTensor()
        tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
        te = datasets.MNIST("./data", train=False, download=True, transform=tf)
        xtr = tr.data.float().view(-1, 784)[:NTRAIN] / 255.0
        ytr = tr.targets[:NTRAIN]
        xte = te.data.float().view(-1, 784)[:NTEST] / 255.0
        yte = te.targets[:NTEST]
        return xtr, ytr, xte, yte, "MNIST"
    except Exception as e:
        print("MNIST failed, synthetic fallback:", e)
        g = torch.Generator().manual_seed(0)
        centers = torch.randn(10, 784, generator=g) * 0.3
        def mk(n):
            y = torch.randint(0, 10, (n,), generator=g)
            return centers[y] + torch.randn(n, 784, generator=g) * 0.8, y
        xtr, ytr = mk(NTRAIN); xte, yte = mk(NTEST)
        return xtr, ytr, xte, yte, "synthetic"

class Net:
    def __init__(self, mode):
        g = torch.Generator().manual_seed(0)
        self.mode = mode
        self.W1 = torch.randn(784, NH, generator=g) * 0.05
        self.W2 = torch.randn(NH, 10, generator=g) * 0.05
        self.V = torch.randn(10, NH, generator=g) * 0.1   # fixed random feedback (apical synapses)
    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1
        self.h = torch.sigmoid(self.z1)
        return self.h @ self.W2
    def backward(self, y):
        t = F.one_hot(y, 10).float()
        delta = torch.softmax(self.out, -1) - t
        dW2 = self.h.T @ delta
        if self.mode == "bp":
            da = (delta @ self.W2.T) * self.h * (1 - self.h)
        else:  # dendritic: apical channel with fixed random V
            za = delta @ self.V                     # (B, NH) apical potential
            da = za * self.h * (1 - self.h)
        dW1 = self.x.T @ da
        self.W2 -= LR / y.size(0) * dW2
        self.W1 -= LR / y.size(0) * dW1

def run(mode, data, tag):
    xtr, ytr, xte, yte, _ = data
    net = Net(mode)
    curve = []
    for ep in range(EPOCHS):
        perm = torch.randperm(NTRAIN)
        for i in range(0, NTRAIN, BS):
            idx = perm[i:i+BS]
            net.out = net.forward(xtr[idx])
            net.backward(ytr[idx])
        with torch.no_grad():
            acc = 0
            for j in range(0, NTEST, 500):
                acc += ((xte[j:j+500] @ net.W1).sigmoid() @ net.W2).argmax(1).eq(yte[j:j+500]).sum().item()
            acc = acc / NTEST * 100
        curve.append(round(acc, 2))
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"[{tag}] ep{ep+1} test_acc={acc:.2f}%", flush=True)
    return {"final_acc": curve[-1], "curve_every10": curve[::10]}

def main():
    t0 = time.time()
    data = load_data()
    res = {"data": data[4], "ntrain": NTRAIN, "epochs": EPOCHS,
           "note": "dendritic = apical random feedback channel (FA); biologically plausible, no weight transport"}
    torch.manual_seed(0); res["bp"] = run("bp", data, "bp")
    torch.manual_seed(0); res["dendritic_apical"] = run("dend", data, "dend")
    res["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    json.dump(res, open("exp3_result.json", "w"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

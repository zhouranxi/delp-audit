"""诊断：模型在静态规则（n_rules=1，无切换）下能否学会？

A: 复刻当前设置（lr 3e-4, 300 ep）——验证是否复现 ~0.52
B: lr 1e-3, 1200 ep——看是优化问题还是预算问题
每 25 episodes 打印滚动正确率。
"""
import argparse
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.rule_switch import RuleSwitchEnv
from models.baselines import PointNet
from train import run_episode


def run(lr, episodes, seed, n_rules, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = RuleSwitchEnv(seed=seed, n_rules=n_rules)
    model = PointNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    accs = []
    for ep in range(episodes):
        block_accs, _ = run_episode(model, env, opt, device=device)
        accs.append(np.mean(block_accs))
        if (ep + 1) % 50 == 0:
            recent = np.mean(accs[-50:])
            print(f"  ep {ep+1:>5}  rolling50_acc={recent:.3f}", flush=True)
    return accs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print("== A: 静态规则, lr=3e-4, 300ep（复刻当前设置）==")
    run(3e-4, 300, seed=0, n_rules=1, device=args.device)
    print("== B: 静态规则, lr=1e-3, 1200ep ==")
    run(1e-3, 1200, seed=0, n_rules=1, device=args.device)

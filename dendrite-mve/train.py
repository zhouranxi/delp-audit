"""MVE 训练入口（REINFORCE + 价值基线 + 信念辅助损失）。

用法：python train.py --model dc_net --seed 0
输出：results/<model>_seed<seed>.json（逐块正确率 + 各切换点适应 trial 数）
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.rule_switch import RuleSwitchEnv
from models.dc_net import DCNet
from models.baselines import PointNet, RandomMLP, CheapLearner


def build_model(name, seed):
    if name == "dc_net":
        return DCNet()
    if name == "dc_net_no_apical":
        return DCNet(ablate_apical=True)
    if name == "dc_net_no_topdown":
        return DCNet(ablate_topdown=True)
    if name == "dc_net_swap":
        return DCNet(swap_compartments=True)
    if name == "point_net":
        return PointNet()
    if name == "random_mlp":
        return RandomMLP(seed=seed)
    if name == "cheap_learner":
        return CheapLearner(seed=seed)
    raise ValueError(name)


def is_torch_model(model):
    return not isinstance(model, CheapLearner)


def adapt_trials(block_accs, window=10, thresh=0.8):
    """切换后达到 80% 正确率所需 trial 数（滑动窗 10）。返回 None 表示本块未达标。"""
    for i in range(len(block_accs) - window + 1):
        if np.mean(block_accs[i:i + window]) >= thresh:
            return i
    return None


def run_episode(model, env, optimizer=None, device="cpu"):
    is_torch = not isinstance(model, CheapLearner)
    obs = env.reset_with_first()
    h = (torch.zeros(1, 64, device=device) if is_torch else None)
    feedback = (torch.zeros(1, 3, device=device) if is_torch else 0.0)
    log_probs, rewards, values, entropies = [], [], [], []
    beliefs = []
    correct_flags, labels_guess = [], []
    prev_correct = 0.0
    prev_rpe = 0.0
    done = False
    while not done:
        x = torch.tensor(obs, device=device).unsqueeze(0)
        if is_torch:
            logits, v, belief_logits, h = model(x, feedback, h)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(v)
            action_int = int(action.item())
        else:
            action_int = model.act(obs, feedback)
        rule_before_step = int(env.rule_idx)
        next_obs, reward, label, done, switched = env.step(action_int)
        correct_flags.append(int(action_int == label))
        if is_torch:
            rewards.append(torch.tensor(reward, device=device))
            fb = torch.zeros(1, 3, device=device)
            fb[0, 0] = reward
            fb[0, 1] = prev_correct
            fb[0, 2] = prev_rpe
            feedback = fb
            prev_correct = float(action_int == label)
            prev_rpe = float(reward - v.detach().item())
            if belief_logits is not None:
                beliefs.append(belief_logits)
                # 监督标签取 step 之前的规则（生成当前刺激的规则），避免切换步错位
                labels_guess.append(rule_before_step)
        else:
            model.update(obs, reward, action_int)
        obs = next_obs
    # 指标：逐 40-trial 块正确率 + 各切换点适应 trial 数
    n_blocks = len(correct_flags) // env.block_len
    block_accs = [float(np.mean(correct_flags[b * env.block_len:(b + 1) * env.block_len]))
                  for b in range(n_blocks)]
    adapt = []
    for s in env.switch_points:
        adapt.append(adapt_trials(correct_flags[s:s + env.block_len]))
    if is_torch and optimizer is not None:
        R = torch.zeros(1, device=device)
        policy_loss, value_loss = 0.0, 0.0
        for lp, r, v in zip(reversed(log_probs), reversed(rewards), reversed(values)):
            R = r + 0.9 * R
            adv = R - v
            policy_loss = policy_loss - lp * adv.detach()
            value_loss = value_loss + adv.pow(2)
        loss = policy_loss + 0.5 * value_loss
        loss = loss - 0.01 * torch.stack(entropies).mean()
        if beliefs and len(labels_guess) == len(beliefs):
            aux = F.cross_entropy(torch.cat(beliefs, dim=0),
                                  torch.tensor(labels_guess, device=device))
            loss = loss + 0.3 * aux
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return block_accs, adapt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-rules", type=int, default=4)
    ap.add_argument("--block-len", type=int, default=40)
    ap.add_argument("--episode-len", type=int, default=400)
    args = ap.parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = RuleSwitchEnv(seed=args.seed, n_rules=args.n_rules,
                        block_len=args.block_len, episode_len=args.episode_len)
    model = build_model(args.model, args.seed)
    if is_torch_model(model):
        model.to(device)
    is_torch = not isinstance(model, CheapLearner)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr) if is_torch else None
    all_block, all_adapt = [], []
    curve = []
    for ep in range(args.episodes):
        block_accs, adapt = run_episode(model, env, opt, device=device)
        all_block.append(block_accs)
        all_adapt.append(adapt)
        curve.append(float(np.mean(block_accs)))
        if (ep + 1) % 100 == 0:
            rolling = float(np.mean(curve[-100:]))
            print(f"ep {ep+1}/{args.episodes} rolling100_acc={rolling:.3f}", flush=True)
    os.makedirs("results", exist_ok=True)
    out = {"model": args.model, "seed": args.seed, "lr": args.lr,
           "n_rules": args.n_rules, "block_len": args.block_len,
           "episode_len": args.episode_len,
           "block_accs_mean_last20": float(np.mean([np.mean(b[-20:]) for b in all_block])),
           "adapt_trials_median": (None if all(
               a is None for e in all_adapt for a in e)
               else float(np.median([a for e in all_adapt for a in e if a is not None]))),
           "adapt_trials_all": [list(map(lambda v: v, e)) for e in all_adapt],
           "curve_every50": [float(np.mean(curve[i:i + 50])) for i in range(0, len(curve), 50)]}
    path = f"results/{args.model}_seed{args.seed}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved {path}: final_acc={out['block_accs_mean_last20']:.3f} "
          f"adapt_median={out['adapt_trials_median']}")


if __name__ == "__main__":
    main()

"""PR-002 训练入口（带线索任务）。用法同 train.py，模型名为 *_cue。

输出额外记录 switch_cue_valid（各切换点之后新块的线索是否失准），
供 stats2.py 分条件统计。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.rule_switch_cue import RuleSwitchCueEnv
from models.cue_models import DCNetCue, PointNetCue, RandomMLPCue, CheapLearnerCue
from train import adapt_trials


def build_model(name, seed):
    if name == "dc_net_cue":
        return DCNetCue()
    if name == "point_net_cue":
        return PointNetCue()
    if name == "random_mlp_cue":
        return RandomMLPCue(seed=seed)
    if name == "cheap_learner_cue":
        return CheapLearnerCue(seed=seed)
    raise ValueError(name)


def run_episode(model, env, optimizer=None, device="cpu"):
    is_torch = not isinstance(model, CheapLearnerCue)
    obs = env.reset_with_first()
    h = (torch.zeros(1, 64, device=device) if is_torch else None)
    feedback = (torch.zeros(1, 3, device=device) if is_torch else 0.0)
    log_probs, rewards, values, entropies, beliefs = [], [], [], [], []
    correct_flags, labels_guess = [], []
    prev_correct, prev_rpe = 0.0, 0.0
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
                labels_guess.append(rule_before_step)
        else:
            model.update(obs, reward, action_int)
        obs = next_obs
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
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-rules", type=int, default=4)
    ap.add_argument("--block-len", type=int, default=80)
    ap.add_argument("--episode-len", type=int, default=480)
    ap.add_argument("--p-cue-valid", type=float, default=0.9)
    ap.add_argument("--tag", default="results")
    args = ap.parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = RuleSwitchCueEnv(seed=args.seed, n_rules=args.n_rules,
                           block_len=args.block_len, episode_len=args.episode_len,
                           p_cue_valid=args.p_cue_valid)
    model = build_model(args.model, args.seed)
    is_torch = not isinstance(model, CheapLearnerCue)
    if is_torch:
        model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr) if is_torch else None
    all_block, all_adapt, all_cue_valid = [], [], []
    curve = []
    for ep in range(args.episodes):
        block_accs, adapt = run_episode(model, env, opt, device=device)
        all_block.append(block_accs)
        all_adapt.append(adapt)
        all_cue_valid.append(env.switch_cue_valid())
        curve.append(float(np.mean(block_accs)))
        if (ep + 1) % 100 == 0:
            print(f"ep {ep+1}/{args.episodes} rolling100_acc={np.mean(curve[-100:]):.3f}",
                  flush=True)
    os.makedirs(args.tag, exist_ok=True)
    out = {"model": args.model, "seed": args.seed, "lr": args.lr,
           "n_rules": args.n_rules, "block_len": args.block_len,
           "episode_len": args.episode_len, "p_cue_valid": args.p_cue_valid,
           "block_accs_mean_last20": float(np.mean([np.mean(b[-20:]) for b in all_block])),
           "adapt_trials_median": (None if all(
               a is None for e in all_adapt for a in e)
               else float(np.median([a for e in all_adapt for a in e if a is not None]))),
           "adapt_trials_all": [list(map(lambda v: v, e)) for e in all_adapt],
           "switch_cue_valid_all": all_cue_valid,
           "curve_every50": [float(np.mean(curve[i:i + 50])) for i in range(0, len(curve), 50)]}
    path = f"{args.tag}/{args.model}_seed{args.seed}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved {path}: final_acc={out['block_accs_mean_last20']:.3f} "
          f"adapt_median={out['adapt_trials_median']}")


if __name__ == "__main__":
    main()

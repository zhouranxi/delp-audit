"""PR-002 带线索模型（信息等价、路由不同）。

dc_net_cue:  线索仅入顶端区室（隔离路由）；基树突仅收感觉输入 x。
point_net_cue / random_mlp_cue:  线索与 x 融合入共享主干（融合路由）。
所有模型 GRU 摘要器输入相同（x ⊕ feedback），价值头同构，信息量完全等价。
"""
import numpy as np
import torch
import torch.nn as nn

from .dc_net import HistorySummarizer


class DCNetCue(nn.Module):
    """区室隔离路由：cue → 顶端 only，x → 基树突 only。"""

    def __init__(self, obs_dim: int = 16, n_rules: int = 4, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.sens_dim = obs_dim
        self.n_rules = n_rules
        self.summarizer = HistorySummarizer(obs_dim)  # 摘要器只吃感觉输入（与其他模型同构）
        self.basal = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        apical_in = hidden + n_rules + 1  # GRU 摘要 + cue + RPE
        self.apical = nn.Sequential(nn.Linear(apical_in, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU())
        self.soma = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2))
        self.value = nn.Linear(hidden, 1)
        self.belief_head = nn.Linear(hidden, n_rules)

    def forward(self, x, feedback, h):
        h = self.summarizer(x[:, :self.sens_dim], feedback, h)
        v_base = self.value(h).squeeze(-1)
        xs, cue = x[:, :self.obs_dim], x[:, self.obs_dim:]
        rpe = feedback[:, 2:3] if feedback.size(-1) >= 3 else \
            torch.zeros(x.size(0), 1, device=x.device)
        basal_out = self.basal(xs)
        apical_out = self.apical(torch.cat([h, cue, rpe], dim=-1))
        logits = self.soma(torch.cat([basal_out, apical_out], dim=-1))
        return logits, v_base, self.belief_head(apical_out), h


class PointNetCue(nn.Module):
    """融合路由：cue 与 x 一起进共享主干。"""

    def __init__(self, obs_dim: int = 16, n_rules: int = 4, hidden: int = 64):
        super().__init__()
        self.sens_dim = obs_dim
        self.summarizer = HistorySummarizer(obs_dim)
        self.value = nn.Linear(hidden, 1)
        in_dim = obs_dim + n_rules + hidden
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden * 2), nn.ReLU(),
                                   nn.Linear(hidden * 2, hidden * 2), nn.ReLU(),
                                   nn.Linear(hidden * 2, 2))

    def forward(self, x, feedback, h):
        h = self.summarizer(x[:, :self.sens_dim], feedback, h)
        v_base = self.value(h).squeeze(-1)
        logits = self.trunk(torch.cat([x, h], dim=-1))
        return logits, v_base, None, h


class RandomMLPCue(nn.Module):
    """融合路由 + 参数打散（同 DCNetCue 参数预算）。"""

    def __init__(self, obs_dim: int = 16, n_rules: int = 4, hidden: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 54321)

        def shuffled(linear):
            w = linear.weight.detach().clone()
            perm = torch.randperm(w.numel(), generator=g)
            with torch.no_grad():
                linear.weight.copy_(w.flatten()[perm].view_as(w))
            return linear

        self.sens_dim = obs_dim
        self.summarizer = HistorySummarizer(obs_dim)
        self.value = nn.Linear(hidden, 1)
        in_dim = obs_dim + n_rules + hidden
        self.trunk = nn.Sequential(
            shuffled(nn.Linear(in_dim, hidden)), nn.ReLU(),
            shuffled(nn.Linear(hidden, hidden)), nn.ReLU(),
            shuffled(nn.Linear(hidden, hidden)), nn.ReLU(),
            nn.Linear(hidden, 2))
        self.aux = shuffled(nn.Linear(hidden, n_rules))

    def forward(self, x, feedback, h):
        h = self.summarizer(x[:, :self.sens_dim], feedback, h)
        v_base = self.value(h).squeeze(-1)
        z = self.trunk[:6](torch.cat([x, h], dim=-1))
        logits = self.trunk[6](z)
        return logits, v_base, self.aux(z), h


class CheapLearnerCue:
    """平凡策略闸门（忽略线索，纯感觉逻辑回归 + 奖励加权）。"""

    def __init__(self, obs_dim: int = 16, n_rules: int = 4, lr: float = 0.05, seed: int = 0):
        self.w = np.zeros(obs_dim + 1)
        self.lr = lr

    def act(self, x, feedback, h=None):
        xs = x[:16] if hasattr(x, '__len__') and len(x) > 16 else x
        z = float(self.w[:len(xs)] @ xs + self.w[-1])
        return 1 if z >= 0 else 0

    def update(self, x, reward, action):
        xs = x[:16] if len(x) > 16 else x
        y = 1.0 if reward > 0 else 0.0
        z = float(self.w[:len(xs)] @ xs + self.w[-1])
        p = 1 / (1 + np.exp(-z))
        err = (p - (1 - y))
        self.w[:len(xs)] -= self.lr * err * reward * xs
        self.w[-1] -= self.lr * err * reward

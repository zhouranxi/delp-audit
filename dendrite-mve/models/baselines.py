"""三组对照（Q3/Q4 定案）。

PointNet：同 GRU 摘要器，无区室。FLOPs 主口径对齐（区室宽度相应调窄）。
RandomMLP：与 DC-Net 同参数量、结构随机重排（无区室语义）——结构先验因果对照。
CheapLearner：无梯度、纯历史的在线逻辑回归（奖励加权）——防平凡策略闸门。
"""
import numpy as np
import torch
import torch.nn as nn

from .dc_net import HistorySummarizer


class PointNet(nn.Module):
    def __init__(self, obs_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.summarizer = HistorySummarizer(obs_dim)
        self.value = nn.Linear(hidden, 1)
        # FLOPs 对齐：DC-Net 主干 = basal(2层) + apical(2层) + soma(2层)；
        # PointNet 用 2 层较宽 MLP 匹配前向计算量（参数量差异如实报告）。
        self.trunk = nn.Sequential(nn.Linear(obs_dim + hidden, hidden * 2), nn.ReLU(),
                                   nn.Linear(hidden * 2, hidden * 2), nn.ReLU(),
                                   nn.Linear(hidden * 2, 2))

    def forward(self, x, feedback, h):
        h = self.summarizer(x, feedback, h)
        v_base = self.value(h).squeeze(-1)
        logits = self.trunk(torch.cat([x, h], dim=-1))
        return logits, v_base, None, h


class RandomMLP(nn.Module):
    """与 DC-Net 同参数预算、区室结构打散进普通 MLP。实现策略：
    用固定种子生成与 DCNet 各层同形状的权重并把其排列随机重排，
    训练方式与 DCNet 一致——收益差异只能来自结构语义而非参数量。"""

    def __init__(self, obs_dim: int = 16, hidden: int = 64, n_rules: int = 4, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 12345)
        def shuffled(linear):
            w = linear.weight.detach().clone()
            perm = torch.randperm(w.numel(), generator=g)
            with torch.no_grad():
                linear.weight.copy_(w.flatten()[perm].view_as(w))
            return linear
        self.summarizer = HistorySummarizer(obs_dim)
        self.value = nn.Linear(hidden, 1)
        self.trunk = nn.Sequential(
            shuffled(nn.Linear(obs_dim + hidden, hidden)), nn.ReLU(),
            shuffled(nn.Linear(hidden, hidden)), nn.ReLU(),
            shuffled(nn.Linear(hidden, hidden)), nn.ReLU(),
            nn.Linear(hidden, 2))
        self.aux = shuffled(nn.Linear(hidden, n_rules))

    def forward(self, x, feedback, h):
        h = self.summarizer(x, feedback, h)
        v_base = self.value(h).squeeze(-1)
        z = self.trunk[:6](torch.cat([x, h], dim=-1))
        logits = self.trunk[6](z)
        return logits, v_base, self.aux(z), h


class CheapLearner:
    """在线逻辑回归（奖励加权样本），无区室无梯度网络——
    若它与 DC-Net 差距不大，说明任务可被平凡策略解决（Q4 闸门）。"""

    def __init__(self, obs_dim: int = 16, lr: float = 0.05, seed: int = 0):
        self.w = np.zeros(obs_dim + 1)
        self.lr = lr

    def act(self, x, feedback, h=None):
        z = float(self.w[:len(x)] @ x + self.w[-1])
        return 1 if z >= 0 else 0

    def update(self, x, feedback, action):
        # 用带噪声的反馈构造伪标签样本，奖励加权更新
        y = 1.0 if feedback > 0 else 0.0
        z = float(self.w[:len(x)] @ x + self.w[-1])
        p = 1 / (1 + np.exp(-z))
        err = (p - (1 - y))  # 伪标签 = 反馈指向的标签
        self.w[:len(x)] -= self.lr * err * feedback * x
        self.w[-1] -= self.lr * err * feedback

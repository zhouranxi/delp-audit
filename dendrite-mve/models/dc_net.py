"""DC-Net：多区室模型（Q12 定案实现）。

基树突区室：接收感觉输入 x，局部非线性。
顶端区室：接收自上而下信号 [奖励预测误差(广播) ⊕ GRU 历史摘要(64 维)]。
胞体：MLP 整合两路区室输出 → 选择 logits + 价值。
辅助头：顶端区室输出 -> 当前规则 one-hot 交叉熵（lambda_aux=0.3，可扫）。
消融纪律：只动顶端通路接线，其余结构冻结。
"""
import torch
import torch.nn as nn


class HistorySummarizer(nn.Module):
    """三组网络共享同一结构（归因纪律）：GRU(64)。"""

    def __init__(self, obs_dim: int, feedback_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRUCell(obs_dim + feedback_dim, hidden)
        self.hidden = hidden

    def forward(self, x, feedback, h):
        return self.gru(torch.cat([x, feedback], dim=-1), h)


class DCNet(nn.Module):
    def __init__(self, obs_dim: int = 16, n_rules: int = 4, hidden: int = 64,
                 ablate_apical: bool = False, ablate_topdown: bool = False,
                 swap_compartments: bool = False):
        super().__init__()
        self.n_rules = n_rules
        self.ablate_apical = ablate_apical        # 消融1：去顶端区室
        self.ablate_topdown = ablate_topdown      # 消融2：顶端只收噪声
        self.swap = swap_compartments             # 消融3：打乱区室分工
        self.summarizer = HistorySummarizer(obs_dim)
        self.basal = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        apical_in = hidden + 1  # GRU 摘要 + RPE 标量
        self.apical = nn.Sequential(nn.Linear(apical_in, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU())
        self.soma = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2))          # 选择 logits
        self.value = nn.Linear(hidden, 1)                        # 价值头（自 RPE）
        self.belief_head = nn.Linear(hidden, n_rules)            # 信念辅助头
        self.lambda_aux = 0.3

    def forward(self, x, feedback, h):
        h = self.summarizer(x, feedback, h)
        # 价值基线用于构造 RPE（当前步的粗略奖励预测）
        v_base = self.value(h).squeeze(-1)
        basal_out = self.basal(x)
        if self.ablate_apical:
            apical_out = torch.zeros_like(basal_out)
        else:
            topdown = h
            if self.ablate_topdown:
                topdown = torch.randn_like(h) * 0.1  # 只收噪声
            # 前一奖励预测误差由 train.py 经 feedback[:,2] 传入
            rpe = feedback[:, 2:3] if feedback.size(-1) >= 3 \
                else torch.zeros(x.size(0), 1, device=x.device)
            apical_out = self.apical(torch.cat([topdown, rpe], dim=-1))
        if self.swap:
            rpe0 = feedback[:, 2:3] if feedback.size(-1) >= 3 \
                else torch.zeros(x.size(0), 1, device=x.device)
            basal_out, apical_out = self.apical(torch.cat([h, rpe0], dim=-1)), self.basal(x)
        soma_in = torch.cat([basal_out, apical_out], dim=-1)
        logits = self.soma(soma_in)
        belief_logits = self.belief_head(apical_out) if not self.ablate_apical else None
        return logits, v_base, belief_logits, h

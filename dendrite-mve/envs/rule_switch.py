"""规则切换分类环境（MVE 定案参数：d=16, 4 规则, 40 trial 切换, 85% 噪声奖励）。

任务：智能体逐步接收刺激 x，输出二分类选择；隐藏规则 = 随机线性边界，
每 40 trial 无预警切换（不立即重复上一规则）。奖励反馈有噪声：
选择正确时以 p=0.85 给 r=1，错误时以 p=0.15 给 r=1（对标行为实验的反馈噪声）。
"""
import numpy as np


class RuleSwitchEnv:
    def __init__(self, dim: int = 16, n_rules: int = 4, block_len: int = 40,
                 episode_len: int = 400, p_reward_correct: float = 0.85,
                 seed: int = 0):
        self.dim = dim
        self.n_rules = n_rules
        self.block_len = block_len
        self.episode_len = episode_len
        self.p_ok = p_reward_correct
        self.rng = np.random.default_rng(seed)
        # 每 seed 重新采样规则边界，避免固定规则被背下来
        self.rules = self.rng.normal(size=(n_rules, dim))
        self.rules /= np.linalg.norm(self.rules, axis=1, keepdims=True)

    def reset(self):
        self.t = 0
        # 无预警切换：随机排列保证不立即重复
        order = []
        while len(order) * self.block_len < self.episode_len + self.n_rules:
            perm = self.rng.permutation(self.n_rules)
            if len(perm) > 1 and order and perm[0] == order[-1]:
                perm[[0, 1]] = perm[[1, 0]]
            order.extend(perm.tolist())
        self.order = order
        self.rule_idx = order[0]
        self.switch_points = list(range(self.block_len, self.episode_len, self.block_len))
        return self._obs()

    def _obs(self):
        return self.rng.normal(size=self.dim).astype(np.float32)

    def _true_label(self, x):
        return 1 if float(self.rules[self.rule_idx] @ x) >= 0 else 0

    def step(self, action: int):
        """action: 0/1。返回 (下一刺激, 噪声奖励, 隐藏真值, 是否切换)。"""
        label = self._true_label(self._obs_cache)
        correct = (action == label)
        reward = 1.0 if self.rng.random() < (self.p_ok if correct else 1 - self.p_ok) else 0.0
        self.t += 1
        switched = False
        if self.t < self.episode_len and self.t in self.switch_points:
            b = self.t // self.block_len
            if b < len(self.order):
                self.rule_idx = int(self.order[b])
            switched = True
        done = self.t >= self.episode_len
        obs = self._obs()
        self._obs_cache = obs
        return obs, reward, label, done, switched

    # 让 reset 返回的第一个刺激可被 step 使用
    def reset_with_first(self):
        obs = self.reset()
        self._obs_cache = obs
        return obs

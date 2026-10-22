"""带规则提示线索的规则切换环境（PR-002）。

在 RuleSwitchEnv 基础上增加线索：每块开始时以 p_valid=0.9 给真实规则的 one-hot
线索，否则给随机错误规则线索（逐块独立）。obs = [x(16) ⊕ cue(4)]。
其余任务几何同 PR-001 A2 定案（n_rules=4, block=80, episode=480, noise=0.85）。
"""
import numpy as np

from .rule_switch import RuleSwitchEnv


class RuleSwitchCueEnv(RuleSwitchEnv):
    def __init__(self, dim: int = 16, n_rules: int = 4, block_len: int = 80,
                 episode_len: int = 480, p_reward_correct: float = 0.85,
                 p_cue_valid: float = 0.9, seed: int = 0):
        super().__init__(dim, n_rules, block_len, episode_len, p_reward_correct, seed)
        self.p_cue_valid = p_cue_valid
        self.obs_dim = dim + n_rules
        # 逐块线索有效性（含第一块），reset 时重画
        self._block_cue_valid = []
        self._cue_rule = 0

    def _draw_cues(self):
        n_blocks = int(np.ceil(self.episode_len / self.block_len))
        self._block_cue_valid = list(
            self.rng.random(n_blocks) < self.p_cue_valid)
        self._cue_rules = []
        for b, valid in enumerate(self._block_cue_valid):
            true_r = self.order[b] if b < len(self.order) else self.order[-1]
            if valid:
                self._cue_rules.append(int(true_r))
            else:
                wrong = self.rng.integers(0, self.n_rules)
                if wrong == true_r:
                    wrong = (wrong + 1) % self.n_rules
                self._cue_rules.append(int(wrong))

    def reset(self):
        self._x = super().reset()  # 16 维纯感觉 obs
        self._draw_cues()
        self._cue_rule = self._cue_rules[0]
        return self._obs_with_cue(self._x)

    def reset_with_first(self):
        obs = self.reset()
        self._obs_cache = self._x  # 父类 step 用它算 label，必须是纯感觉 obs
        return obs

    def _obs_with_cue(self, x):
        cue = np.zeros(self.n_rules, dtype=np.float32)
        cue[self._cue_rule] = 1.0
        return np.concatenate([x, cue]).astype(np.float32)

    def _block_index(self):
        return min(self.t // self.block_len, len(self._cue_rules) - 1)

    def step(self, action):
        obs_next, reward, label, done, switched = super().step(action)
        # 块切换时同步更新线索（super().step 已按 switch_points 换规则）
        self._cue_rule = self._cue_rules[self._block_index()]
        return self._obs_with_cue(obs_next), reward, label, done, switched

    # 对齐 switch_points 的「切换后新块线索是否失准」
    def switch_cue_valid(self):
        out = []
        for s in self.switch_points:
            b = min(s // self.block_len, len(self._block_cue_valid) - 1)
            out.append(bool(self._block_cue_valid[b]))
        return out

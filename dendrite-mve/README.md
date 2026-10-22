# Dendrite-MVE：树突区室分工最小可行实验

对标徐宁龙团队《Science》2026（顶端树突编码抽象规则、胞体编码感觉信息）的
计算层机制验证。目标：检验"区室功能分工作为结构先验"能否加速规则切换后的适应。

## 定案约束（拷问三轮后的共识，改动需修订 prereg）

- 主卖点：机制验证（不是刷榜）
- 第一版非脉冲连续值 ANN；SNN 降级为扩展实验
- 基线对齐口径：FLOPs 主口径 + 参数量如实报告；必含同参数量随机 MLP 对照
- 防平凡策略：85% 噪声奖励 + 历史型廉价学习器对照
- MVE 范围：单规模（d=16, 4 规则, 40 trial 切换, 400 trial/回合, 8 seeds）
- 判定阈值：DC-Net 适应 trial 数中位数较 Point-Net 缩短 >=25% 且 Mann-Whitney p<0.05
- 执行环境：RTX 4060 笔记本；人机分工 = AI 包工程、用户包科学解读
- 推进姿态：自娱自乐式（预注册纪律保留，arXiv/workshop 为可选项而非承诺）

## 目录

```
envs/rule_switch.py    任务环境（规则切换分类，噪声奖励）
models/dc_net.py       DC-Net 多区室模型（基树突/顶端区室/胞体 + 信念辅助头）
models/baselines.py    PointNet / RandomMLP / CheapLearner 三组对照
train.py               单配置训练入口（python train.py --model dc_net --seed 0）
stats.py               汇总结果 + Mann-Whitney U + bootstrap CI + Cohen's d
prereg/                预注册（开跑前写好，跑完不改；不符则如实报探索性发现）
results/               输出目录（自动创建）
```

## 纪律（人机分工的硬约束）

1. 每个实验开跑前先写 `prereg/prereg_<name>.md`（假设/预期/阈值/seeds/日期）。
2. 实验设计变更、结果解读、结论措辞由用户（Lance）决定；AI 只改代码不改结论。
3. 结果与预注册不符时，如实记为探索性发现，禁止事后改阈值。
4. 三组可学习网络共享同一 GRU 历史摘要器结构，保证区室收益归因干净。

## 快速开始

```bash
pip install -r requirements.txt
# 8 seeds 全量（约数小时，4060 可承受）
for s in 0 1 2 3 4 5 6 7; do
  for m in dc_net point_net random_mlp cheap_learner; do
    python train.py --model $m --seed $s
  done
done
python stats.py   # 输出 H1 判定
```

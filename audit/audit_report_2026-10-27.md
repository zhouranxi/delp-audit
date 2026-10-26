# DeLP 审计论文（DOI 10.5281/zenodo.23602281, arXiv v0.2）数据一致性审计

审计对象：D:/Download/DeLP_audit_v0.2_Zhou2026.pdf（7 页）
数据源：github.com/zhouranxi/delp-audit（main @ 3cc89c5d，2026-10-22）
本地副本：_audit/repo/

## 一、核验通过（可逐位复算）

| 论文主张 | 数据源 | 结果 |
|---|---|---|
| Table II 全 10 格（Fashion-MNIST） | bench/results_fashion/*.json stage=main（5 seeds） | **10/10 逐位吻合** |
| Table V 全 3 行（PR-001/002/003） | dendrite-mve/prereg/*.md 结果记录 | **全吻合**（0.648/0.643/6.5/7.0/7.1%/p=0.3346；0.649/0.654/6.5/6.0/0.296；0.647/0.632/6.5/0.0%/0.4576） |
| 增益扫描 76.24→92.40（+16.2） | bench/reports/sig_probe.md | **吻合**（+16.16；α=2 在 T=5/10/20 均胜过 DELP ✓） |
| 原表 DELP −1.94±0.42/80.37 | dendritic-lab/exp12_result.json | **吻合** |
| 原表 MAS orig −12.32/75.27、PackNet orig 0.00/43.95、27.38 | exp12_result.json | **吻合** |
| PackNet corr −12.67/74.96 | exp4_result.json/mlp | **吻合** |
| Fisher 300 / usage 500 | exp6_baselines.py:15 `NFISHER, NUSAGE = 300, 500` | **一致** |
| val split train[10000:12000] | exp6_fixed.py:80 | **一致** |
| kWTA k=N/4（N=256→k=64） | bench_cl.py:63 `nh=256, k=64` | **一致** |
| 全部 98 个论文数字中 91 个在仓库有字面来源 | 全仓扫描 | 7 个无源（见下） |

## 二、确凿问题（必须处理）

### P1【严重·自相矛盾】正文与 Table I 的 SI 数字不一致
- 正文 §IV：「DELP at −1.94/80.37, ahead of **SI (−3.39/77.56)**, MAS (−12.32/75.27)」
- Table I：「**SI (orig. impl.)** −0.15±0.38 / **63.34**±1.63」（5 任务）
- exp12_result.json 里 SI 原表确实是 −3.39 / 77.56
→ 同一张表里 SI 原实现有两个互相矛盾的数。审稿人必然抓。

### P2【严重·数据缺失】Table I 中 8/16 格在仓库数据里找不到来源
缺源格：EWC 5/10 任务、SI (ref) 5/10 任务、SI (orig) 5/10 任务、MAS (corr) 5/10 任务、PackNet (corr) 10 任务。
来源应为 bench/exp6_fixed.py 的 `results_ei/*.json`，**该目录未进仓库**（仓库 bench/ 下只有 results_fashion/ 和 out/、reports/）。
→ 论文最核心的「修正后对比表」无法从公开数据复现。

### P3【严重·口径混用】Table III 与正文的 elasticity 区间来自两份不同的 ladder 报告
- 仓库有两份：bench/out/ladder.md（945 行 CSV）与 bench/reports/ladder.md（980 行 CSV），T=20 数字不同
- Table III 的 T=20 数字（+2.13 / −0.37 / +1.02 / −0.76）匹配 **reports/** 版
- 但正文称「every rung's λ elasticity must be positive (all are, **+2.72 to +9.11**)」——**+9.11 只存在于 out/ 版**（reports 版 T=20 最大 +7.94）
→ 表格和正文引用了两个不同数据批次。

### P4【中·量化口径缺失】Table III 差值与表格精度不自洽
Table III T=20 由 reports 版精度反推：R0→R0b=+2.13 ✓、R1→R2=+1.35 ✓、R2→R3=−0.37 ✓、R3→R4=**+1.03**（论文写 +1.02）、R4 vs R0b=**+2.01**（论文写 +2.00）。
T=10：R3→R4=**+0.08**（论文写 +0.07）、R4 vs R0b=**+0.50**（论文写 +0.49）。
→ 论文用了未四舍五入的原始值算差值，却用表格里的两位小数呈现。审稿人用表内数字一减就对不上（差 0.01）。
→ 需在 Table III 加注「差值由未舍入 per-seed 均值计算」。

### P5【中·样本量不一致】Table II 的 DELP 5 任务行
仓库中 ours T=5 的 stage=main 是 5 seeds（66.94±0.87 / −8.53±1.38），与论文 66.9±0.9 / −8.5±1.4 一致 ✓
但同文件另有 stage=sweep 的 2-seed 记录（67.41）——**若分析脚本未过滤 stage 会算错**。已确认论文值取 main，故无问题，但建议在分析脚本加 stage 过滤断言。

### P6【中·D1 数字无源】MAS 修复后 5 任务 BWT「−3.60」
仓库无此值。summary.csv 里 mas@λ=1.0 的 5-seed 均值是 −5.38（逐 seed：−2.99/−8.13/−5.62/−4.80/−5.38）。
Fashion 数据里 mas_new T=5 λ=1.0 sweep 2-seed 均值 = −3.61（近似但 n=2 且是另一数据集）。
→ −3.60 的确切来源不明（很可能来自 results_ei/*.json，即 P2 同一缺口）。

### P7【轻·Table I 与 grid_check.md 的 λ 选择不同】
bench/reports/grid_check.md 显示 DELP T=5 最优 λ=2000、acc=81.49；论文 Table I 用 λ=5000、acc=80.37。
两者不矛盾（验证集选 λ vs 其它口径），但**论文未说明 Table I 用的是哪个 λ 网格与选法**，审稿人对照 grid_check.md 会困惑。
建议在 Table I caption 注明「λ 由 held-out validation accuracy 选出，具体网格见仓库 grid_check.md」。

## 三、协议一致性（论文 vs 代码）：全部通过
- N=256, kWTA k=N/4 ✓（bench_cl.py:63）
- Fisher 300 / usage 500 ✓（exp6_baselines.py:15）
- val split train[10000:12000] ✓（exp6_fixed.py:80）
- 手动种子 torch.manual_seed(seed) ✓（exp6_fixed.py:93）
- perms 种子 100+seed ✓（exp6_baselines.py:19）
- λ 网格按族覆盖、argmax 全为内点 ✓（grid_check.md）
- kWTA 尺度等变性已验证 ✓（sig_probe.md）

## 四、复现脚本
_audit/audit_table1.py, audit_table1b.py, audit_table2.py, audit_table3.py, trace_numbers.py

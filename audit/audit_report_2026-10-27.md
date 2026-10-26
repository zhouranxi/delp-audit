# DeLP 审计论文（DOI 10.5281/zenodo.23602281, arXiv v0.2）数据一致性审计 —— 最终版

审计对象：D:/Download/DeLP_audit_v0.2_Zhou2026.pdf（7 页）
数据源：github.com/zhouranxi/delp-audit + 学校服务器 10.10.52.210:/mnt/mydisk/xinguan-dendritic/
审计日期：2026-10-27

## 一、Table I 全部 16 格 —— 已找到数据源，100% 吻合

**数据源（服务器上找到并已下载上传）**：
- `dendritic-lab/results_ei_main.json`（80 分片，含 Table I 全部 16 格：八臂 × T∈{5,10} × 5 seeds，per-seed 含 acc_matrix）
- `dendritic-lab/results_ei_sweep.json`（66 分片，λ 选择用的 sweep 记录）

| 臂 | T=5 论文 | 实测 | T=10 论文 | 实测 |
|---|---|---|---|---|
| DELP(ours) | −1.94/80.37 | ✓ | −5.38/55.23 | ✓ |
| EWC | −7.08/74.28 | ✓ | −7.81/68.35 | ✓ |
| SI (ref) | −0.03/62.33 | ✓ | −1.31/40.57 | ✓ |
| SI (orig) | −0.15/63.34 | ✓ | −0.86/39.46 | ✓ |
| MAS (corr) | −3.72/75.22 | ✓ | −5.22/62.91 | ✓ |
| MAS (orig) | −12.32/75.27 | ✓ | −19.88/68.00 | ✓ |
| PackNet (corr) | −12.67/74.96 | ✓ | −19.92/67.93 | ✓ |
| PackNet (orig) | 0.00/43.95 | ✓ | 0.00/27.38 | ✓ |

**标准差口径**：论文用**总体标准差 pstdev**（n=5 时 = stdev × √(4/5)），与 `analyze_ei.py` 的 `agg()` 一致。实测比值 0.42/0.47 = 0.894 = √(4/5) ✓

## 二、其他表 —— 全部逐位吻合

| 项目 | 结果 |
|---|---|
| Table II 全 10 格（Fashion） | 10/10 逐位吻合（`bench/results_fashion/main_*.json` stage=main，5 seeds） |
| Table III 全 21 格 | 用 reports/ladder.csv per-seed 值精确复算，Δ 与 p 全部吻合（含 +1.02/+2.00/+0.07） |
| Table V 全 3 行（PR-001/002/003） | 全吻合（0.648/0.643/6.5/7.0/7.1%/p=0.3346；0.649/0.654/6.5/6.0/0.296；0.647/0.632/6.5/0.0%/p=0.4576） |
| 增益扫描 76.24→92.40 (+16.2) | 吻合；α=2 在 T=5/10/20 均胜过 DELP ✓ |
| 协议项 | N=256、kWTA k=N/4、Fisher 300/usage 500、val split 10000:12000、perms 种子 100+seed、torch.manual_seed 全部一致 |

## 三、发现并修复的问题（8 处）

| # | 严重度 | 问题 | 处理 |
|---|---|---|---|
| 1 | 高 | 正文原表 SI=−3.39/77.56 与 Table I SI(orig)=−0.15/63.34 表面矛盾 | 正文加来源说明：前者是 exp12 存档原表，后者是修正重跑 |
| 2 | 高 | 正文 elasticity 区间 +2.72~+9.11 来自 out/ 旧批次，与 Table III 用的 reports/ 批次不一致 | 改为 +2.55~+7.94（从发布批次重算） |
| 3 | 高 | **Table I caption 称「Every arm uses its own best λ selected on held-out validation」不准确** | 改为如实表述：三个重实现臂用验证集；复用原实现的臂（含 proposed 臂本身）保留原 λ，因原实现不记录验证集指标。D5 同步加说明 |
| 4 | 中 | Table I 数据未在仓库 | 已从服务器下载 results_ei_main/sweep.json 并上传 |
| 5 | 中 | Table III 差值用未舍入值算但两位小数呈现 | caption 加注 |
| 6 | 轻 | Table I 未说明 λ 选择口径 | caption 加 grid_check.md 指引（后被第 3 条的更准确表述覆盖） |
| 7 | 中 | D1 的 MAS 修复后 BWT=−3.60 仓库无源 | 已定位：results_ei_main mas_new λ=100 族；README 数据来源表补全后可通过 sweep+main 复现 |
| 8 | 轻 | README 未说明论文表格↔数据文件对应 | README 新增 Data provenance 表 |

## 四、λ 选择问题的完整技术说明（供回应审稿人）

- 论文 Table I 的 λ：ours=5000, si_old=100, si_new=100, mas_old=10, mas_new=100, ewc=1000, pack_*=0
- `exp6_fixed.py` 中 `LAM_MAP` 注释：「Defaults are the values exp12 reported; *_new entries are overwritten」
- 只有 `train_arm`（修正版实现）返回 `val_bwt`/`val_final_avg`；`run_ours`/`run_si_impl`/`run_mas_impl`（调用 exp6_baselines.py 的原实现）**不返回验证集字段**
- 全盘 grep `"lam_rule": "val"`：只命中 Fashion 批次（20 个），MNIST 主表 80 分片全为 `"test"`
- 从 sweep 数据复算：si_new/mas_new/ewc_new 的 val 规则选出 λ = 100/100/1000，与论文一致 ✓
- ours/si_old/mas_old 无 val 字段，test 规则选出 λ = 5000/100/10，与论文一致 ✓
- 结论：**数值全部正确（无数字错误），但"每臂都用验证集选 λ"的表述过强**——已改为如实说明

## 五、复现脚本（本地 _audit/，已上传 audit/）
audit_table1.py / audit_table1b.py / audit_table2.py / audit_table3.py / audit_table3_p.py / trace_numbers.py
verify_table1_full.py（Table I 全 16 格 + p 值重算）/ verify_lamsel.py（λ 选择规则核查）

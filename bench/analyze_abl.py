"""analyze_abl.py -- is the write gate doing the work, or the extra parameters?

point_wmh holds the cue for 320 silent steps at 100.0 with zero variance across
seven seeds (analyze_xgap.py). That is a strong claim, so it needs the standard
four controls, all evaluated on the SAME trained weights across all gaps:

  trained   the learned gate                       -- the headline number
  frozen    g pinned at its trained mean           -- every parameter kept, only
                                                      the input dependence of g
                                                      is removed
  open      g = 1                                  -- no memory at all
  nocue     cue channels zeroed in the test set    -- was the answer ever read
                                                      from memory?
  (capacity) point_slow widened to nh=237          -- same parameter count, no
                                                      gate; comes from the
                                                      same-gap grid

If frozen still scores 100, the gate is decorative and the win is capacity.
If nocue still scores 100, the model never used the cue and the task is broken.
Either failure kills the mechanism claim, so both are reported unconditionally.

usage
    python analyze_abl.py results_d2_abl
"""
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

GAPS = [2, 10, 40, 80, 160, 320]
CHANCE = 100.0 / 3.0


def mean(v):
    return st.mean(v) if v else float("nan")


def sd(v):
    return st.stdev(v) if len(v) > 1 else 0.0


def main(folder):
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(folder, "*.json")))]
    rows = [d for d in rows if "error" not in d]
    if not rows:
        print("no rows yet")
        return

    print("# 方向二：门控臂的四项对照（同一套权重，训练 gap=40，跨全部 gap）\n")
    print(f"读出最后 15 步，无记忆地板 {CHANCE:.1f}%。\n")

    print("## 1  门值在学习什么\n")
    print("| arm | 种子数 | 线索窗口 g | 间隙 g | 冻结用的常数 |")
    print("|---|---|---|---|---|")
    for arm in sorted({d["arm"] for d in rows}):
        sub = [d for d in rows if d["arm"] == arm and "gate_at_train" in d]
        if not sub:
            continue
        cue = mean([d["gate_at_train"][0] for d in sub])
        sil = mean([d["gate_at_train"][1] for d in sub])
        const = mean([d.get("gate_const", float("nan")) for d in sub])
        print(f"| {arm} | {len(sub)} | {cue:.3f} | {sil:.3f} | {const:.3f} |")
    print("\n读法：间隙里 g 越接近 0，状态越接近“保持”；若线索与间隙的 g 差不多，")
    print("说明门没有分化，学到的不是“写入 vs 保持”。\n")

    print("## 2  四个条件下逐 gap 的 acc\n")
    print("| arm | 条件 | " + " | ".join(f"g={g}" for g in GAPS) + " |")
    print("|---|---|" + "---|" * len(GAPS))
    for arm in sorted({d["arm"] for d in rows}):
        sub = [d for d in rows if d["arm"] == arm]
        for cond, key in (("训练后（门）", "curve"), ("门冻结在均值", "frozen"),
                          ("门全开（无记忆）", "open"), ("无线索输入", "nocue")):
            cells = []
            for g in GAPS:
                v = [d[key][str(g)]["acc"] for d in sub if key in d and str(g) in d[key]]
                cells.append(f"{mean(v):.1f}±{sd(v):.1f}" if v else "·")
            print(f"| {arm} | {cond} | " + " | ".join(cells) + " |")
        # context probe, same structure: does the cue survive in the state?
        cells = []
        for g in GAPS:
            v = [d["curve"][str(g)]["ctx_probe"] for d in sub if str(g) in d.get("curve", {})]
            cells.append(f"{mean(v):.1f}" if v else "·")
        print(f"| {arm} | 上下文线性探针 | " + " | ".join(cells) + " |")

    print("\n## 3  判读\n")
    for arm in sorted({d["arm"] for d in rows}):
        sub = [d for d in rows if d["arm"] == arm]
        tr = mean([d["curve"]["40"]["acc"] for d in sub])
        fr = mean([d["frozen"]["40"]["acc"] for d in sub if "frozen" in d])
        op = mean([d["open"]["40"]["acc"] for d in sub if "open" in d])
        nc = mean([d["nocue"]["40"]["acc"] for d in sub if "nocue" in d])
        far = mean([d["curve"]["320"]["acc"] for d in sub])
        verdict = []
        verdict.append("门是机制" if tr - max(fr, op) > 10 else
                       "**门不是机制**（冻结/全开没掉下来）")
        if nc == nc and nc > CHANCE + 10:
            verdict.append("**无线索仍然答对 ⇒ 任务或读出有问题**")
        elif nc == nc:
            verdict.append("答案确实来自记忆")
        if far > CHANCE + 10:
            verdict.append(f"g=320 仍保持（{far:.1f}）")
        else:
            verdict.append(f"g=320 已归零（{far:.1f}）")
        print(f"* **{arm}**：训练 {tr:.1f} / 冻结 {fr:.1f} / 全开 {op:.1f} / 无线索 {nc:.1f}"
              f" → " + "；".join(verdict))
    print()
    print("**通过条件**（全部满足才能把这条写成主结果）：")
    print("1. 训练后 ≫ 冻结 ≥ 全开 ≈ 33.3（门必须做工）；")
    print("2. 无线索 ≈ 33.3（否则答案不是从记忆读出来的）；")
    print("3. g=320 仍明显高于 33.3 且跨种子方差小；")
    print("4. 参数对齐的泄漏臂（point_slow nh=237, 46,698 参数）达不到同样的跨 gap 保持。")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_d2_abl")

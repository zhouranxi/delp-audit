import json, glob, os, statistics as st
from collections import defaultdict

SWEEP = r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/_audit/results_ei_sweep"
MAIN = r"C:/Users/Lance/WorkBuddy/2026-10-25-10-08-27/_audit/results_ei_main"

# 论文 Table I 实际用的 lambda
PAPER_LAM = {
    "ours": 5000.0, "si_old": 100.0, "si_new": 100.0, "mas_old": 10.0,
    "mas_new": 100.0, "ewc": 1000.0, "pack_old": 0.0, "pack_new": 0.0,
}
# sweep 里的臂名 -> main 里的臂名
ARM_ALIAS = {"ewc_new": "ewc", "ewc_old": "ewc", "si_new": "si_new", "si_old": "si_old",
              "mas_new": "mas_new", "mas_old": "mas_old", "ours": "ours"}

sweep = defaultdict(dict)
for f in glob.glob(SWEEP + "/*.json"):
    try: d = json.load(open(f, encoding="utf-8"))
    except Exception: continue
    for k, v in (d.get("sweep") or {}).items():
        arm, lam = k.split("@")
        sweep[arm][float(lam)] = v

print("=== 验证：论文 Table I 用的 λ 是否等于「按验证集 acc 选出」的 λ ===")
print(f"{'sweep臂':10} {'val选出的λ':>12} {'论文用的λ':>10}  判定")
ok_all = True
for arm in sorted(sweep):
    lams = sorted(sweep[arm])
    stats = {}
    for lam in lams:
        rs = sweep[arm][lam]
        va = [r.get("val_final_avg") for r in rs if r.get("val_final_avg") is not None]
        vb = [r.get("val_bwt") for r in rs if r.get("val_bwt") is not None]
        ta = [r.get("final_avg") for r in rs if r.get("final_avg") is not None]
        tb = [r.get("bwt") for r in rs if r.get("bwt") is not None]
        stats[lam] = (st.mean(va) if va else None, st.mean(vb) if vb else None,
                      st.mean(ta) if ta else None, st.mean(tb) if tb else None)
    # val 规则：max(val_bwt, tie-break val_acc) —— 与 pick_lambda 的 val 分支一致
    cands = {l: s for l, s in stats.items() if s[0] is not None and s[1] is not None}
    if cands:
        val_pick = max(cands, key=lambda l: (cands[l][1], cands[l][0]))
    else:
        val_pick = None
    # test 规则：max(test_bwt, tie test_acc)
    cands2 = {l: s for l, s in stats.items() if s[2] is not None and s[3] is not None}
    test_pick = max(cands2, key=lambda l: (cands2[l][3], cands2[l][2])) if cands2 else None
    main_arm = ARM_ALIAS.get(arm)
    paper_lam = PAPER_LAM.get(main_arm)
    m1 = (val_pick == paper_lam)
    m2 = (test_pick == paper_lam)
    ok_all = ok_all and m1
    print(f"{arm:10} {str(val_pick):>12} {str(paper_lam):>10}  "
          f"{'val一致 OK' if m1 else 'val不一致'}{'（test规则也一致）' if m2 else ''}"
          f"   [test选出={test_pick}]")

print()
print("全部 val 规则一致:", ok_all)

# main 分片的 lam_rule 标签
print("\n=== main 分片的 lam_rule 标签 ===")
rules = defaultdict(int)
for f in glob.glob(MAIN + "/*.json"):
    try: d = json.load(open(f, encoding="utf-8"))
    except Exception: continue
    rules[d.get("lam_rule")] += 1
print(" ", dict(rules))

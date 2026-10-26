import os, json, glob, statistics as st, math

BASE = "/mnt/mydisk/xinguan-dendritic/dendritic-lab/results_ei_main"

# 论文 Table I: arm -> {T: (BWT, BWT_sd, acc, acc_sd)}
PAPER = {
 "ours":     {5: (-1.94, 0.42, 80.37, 0.67), 10: (-5.38, 0.44, 55.23, 1.41)},
 "ewc":      {5: (-7.08, 1.19, 74.28, 0.86), 10: (-7.81, 1.28, 68.35, 0.95)},
 "si_new":   {5: (-0.03, 0.35, 62.33, 1.41), 10: (-1.31, 0.29, 40.57, 1.51)},
 "si_old":   {5: (-0.15, 0.38, 63.34, 1.63), 10: (-0.86, 0.13, 39.46, 1.46)},
 "mas_new":  {5: (-3.72, 0.82, 75.22, 0.55), 10: (-5.22, 0.66, 62.91, 1.02)},
 "mas_old":  {5: (-12.32, 2.53, 75.27, 1.99), 10: (-19.88, 1.27, 68.00, 1.23)},
 "pack_new": {5: (-12.67, 2.60, 74.96, 2.01), 10: (-19.92, 0.68, 67.93, 0.83)},
 "pack_old": {5: (0.00, 0.00, 43.95, 0.82), 10: (0.00, 0.00, 27.38, 0.63)},
}
# 论文表格里的臂名与代码臂名对照
PAPER_NAME = {"ours":"DELP (ours)","ewc":"EWC (online, multi-anchor)","si_new":"SI (reference impl.)",
              "si_old":"SI (orig. impl.)","mas_new":"MAS (corrected)","mas_old":"MAS (orig. impl.)",
              "pack_new":"PackNet (corrected)","pack_old":"PackNet (orig. impl.)"}

recs = []
for f in sorted(glob.glob(BASE + "/*.json")):
    try: d = json.load(open(f, encoding="utf-8"))
    except Exception: continue
    if isinstance(d, dict) and "bwt" in d and "final_avg" in d:
        r = dict(d); r["_f"] = os.path.basename(f); recs.append(r)
    elif isinstance(d, dict):
        for k, v in (d.get("main") or {}).items():
            for r in v:
                if "bwt" in r:
                    r = dict(r); r["_f"] = os.path.basename(f); r.setdefault("arm", k.split("|")[0]); recs.append(r)

cells = {}
for r in recs:
    cells.setdefault((r.get("arm"), r.get("ntask"), r.get("lam")), []).append(r)

def pval(t, df):
    x = abs(t); df = max(df, 1)
    def betacf(a,b,x):
        MAXIT,EPS,FPMIN = 300,3.0e-16,1.0e-300
        qab,qap,qam = a+b,a+1.0,a-1.0
        c=1.0; d=1.0-qab*x/qap
        if abs(d)<FPMIN: d=FPMIN
        d=1.0/d; h=d
        for m in range(1,MAXIT+1):
            m2=2*m
            aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1.0+aa*d
            if abs(d)<FPMIN: d=FPMIN
            c=1.0+aa/c
            if abs(c)<FPMIN: c=FPMIN
            d=1.0/d; h*=d*c
            aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1.0+aa*d
            if abs(d)<FPMIN: d=FPMIN
            c=1.0+aa/c
            if abs(c)<FPMIN: c=FPMIN
            d=1.0/d; de=d*c; h*=de
            if abs(de-1.0)<EPS: break
        return h
    def betai(a,b,x):
        if x<=0: return 0.0
        if x>=1: return 1.0
        bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1.0-x))
        if x<(a+1.0)/(a+b+2.0): return bt*betacf(a,b,x)/a
        return 1.0-bt*betacf(b,a,1.0-x)/b
    return betai(df/2.0,0.5,df/(df+x*x))

print("=== Table I 全 16 格精确核对（含标准差口径）===")
print(f"{'臂':22} {'T':>2} {'论文 BWT':>9} {'实测BWT':>9} {'论文sd':>7} {'实测sd':>7} "
      f"{'论文acc':>8} {'实测acc':>8} {'论文sd':>7} {'实测sd':>7}  文件")
allok = True
srcs = set()
for arm, d in PAPER.items():
    for T, (pb, pbsd, pa, pasd) in d.items():
        cand = [(k, v) for k, v in cells.items() if k[0]==arm and k[1]==T and len(v)>=5]
        if not cand:
            print(f"{PAPER_NAME[arm]:22} {T:>2}  <<< 无数据"); allok=False; continue
        k, rs = max(cand, key=lambda kv: len(kv[1]))
        b = [r["bwt"] for r in rs]; a = [r.get("final_avg") for r in rs]
        mb, sb = st.mean(b), st.stdev(b)
        ma, sa = st.mean(a), st.stdev(a)
        pb_sd = st.pstdev(b); pa_sd = st.pstdev(a)
        ok = (abs(mb-pb)<0.006 and abs(ma-pa)<0.006 and abs(sb-pbsd)<0.06 and abs(sa-pasd)<0.06)
        okp = (abs(sb-pbsd)<0.06 and abs(pb_sd-pbsd)<0.06) or (abs(pa_sd-pasd)<0.06 and abs(sa-pasd)<0.06)
        allok = allok and ok
        srcs.add((k[2], rs[0]["_f"]))
        print(f"{PAPER_NAME[arm]:22} {T:>2} {pb:+9.2f} {mb:+9.2f} {pbsd:7.2f} {sb:7.2f} "
              f"{pa:8.2f} {ma:8.2f} {pasd:7.2f} {sa:7.2f}  {rs[0]['_f']}  "
              f"{'OK' if ok else ('sd口径不同' if okp else '<<<')}")

print()
print("=== 论文声称的 p 值复算（配对 t，DELP 为基准）===")
for T in (5, 10):
    base_key = next(k for k in cells if k[0]=="ours" and k[1]==T and len(cells[k])>=5)
    base = {r["seed"]: r for r in cells[base_key]}
    for arm in ("ewc","si_new","si_old","mas_new","mas_old","pack_new","pack_old"):
        cand = [(k,v) for k,v in cells.items() if k[0]==arm and k[1]==T and len(v)>=5]
        if not cand: continue
        k, rs = max(cand, key=lambda kv: len(kv[1]))
        other = {r["seed"]: r for r in rs}
        common = sorted(set(base) & set(other))
        if len(common) < 2: continue
        for field, name in (("final_avg","acc"), ("bwt","BWT")):
            dd = [base[s][field] - other[s][field] for s in common]
            m = st.mean(dd); sd = st.stdev(dd)
            t = m/(sd/math.sqrt(len(dd))) if sd>1e-12 else 0.0
            p = pval(t, len(dd)-1)
            print(f"  T={T:2} DELP vs {arm:10} {name:3}: Δ={m:+7.2f} t({len(dd)-1})={t:+7.2f} p={p:.4f}")
    print()
print("数据来源分片:", sorted(srcs))

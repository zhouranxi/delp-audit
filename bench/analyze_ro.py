"""analyze_ro.py -- the real retention curve, measured with a restricted readout.

`d2_comp_snn` classifies from the firing rate averaged over the WHOLE trial. The
cue fires at t < ncue, so its evoked spikes sit inside that average: a model can
label the trial without carrying anything across the silent gap. Restricting the
readout to the last `RATE_WINDOW` steps removes that route.

This script reports, for every arm and gap:
  * leak   = accuracy(whole-trial) - accuracy(restricted), the size of the
             illusion at that gap (only where both were run);
  * kept   = restricted accuracy, i.e. what actually survived the gap;
  * probe  = linear decodability of the context from the slow state at the end
             of the gap, which is the instrument that never had the leak.

The number that matters is `kept`, not `leak`: if `kept` sits at chance for every
gap beyond a few steps, then no arm retains context across a silent gap and the
task (not the model) was doing the work.
"""
import glob
import json
import statistics as st
import sys
from collections import defaultdict

CHANCE_NOTE = "chance = 33.3 (3 contexts x 3 templates)"


def load(pats):
    by = defaultdict(list)
    for pat in pats:
        for f in glob.glob(pat):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            # d2_readout.py writes {"rows": [...]}; accept either shape so the
            # analysis does not silently report zeros when the wrapper changes.
            recs = d.get("rows") if isinstance(d, dict) and "rows" in d else [d]
            for r in recs:
                if not isinstance(r, dict) or "error" in r or "acc" not in r:
                    continue
                by[(r["arm"], int(r["gap"]), int(r["rate_window"]))].append(r)
    return by


def main():
    pats = sys.argv[1:] or ["results_readout/*.json", "results_ro_sweep/*.json"]
    by = load(pats)
    arms = sorted({k[0] for k in by})
    gaps = sorted({k[1] for k in by})

    L = []
    L.append("# D2 with the readout leak removed\n")
    L.append("`whole` = trial-averaged readout (the historical number). "
             "`last15` = only the final 15 steps. `probe` = context decodable from "
             "the slow state at the end of the gap. " + CHANCE_NOTE + ".\n")

    L.append("| arm | gap | whole | last15 | leak | probe |")
    L.append("|---|---|---|---|---|---|")
    kept = defaultdict(dict)
    for arm in arms:
        for gap in gaps:
            w = by.get((arm, gap, 0))
            k = by.get((arm, gap, 15))
            if not k:
                continue
            kw = st.mean([r["acc"] for r in k])
            kept[arm][gap] = kw
            kp = st.mean([r["ctx_probe"] for r in k if r.get("ctx_probe") is not None])
            if w:
                ww = st.mean([r["acc"] for r in w])
                L.append(f"| {arm} | {gap} | {ww:.1f} | {kw:.1f} | "
                         f"{ww - kw:+.1f} | {kp:.0f} |")
            else:
                L.append(f"| {arm} | {gap} | - | {kw:.1f} | - | {kp:.0f} |")
    L.append("")

    L.append("## Restricted-readout retention curve\n")
    L.append("This is the only column that can support a memory claim.\n")
    L.append("| arm | " + " | ".join(f"gap {g}" for g in gaps) + " |")
    L.append("|---" * (len(gaps) + 1) + "|")
    for arm in arms:
        if not kept[arm]:
            continue
        cells = []
        for g in gaps:
            v = kept[arm].get(g)
            if v is None:
                cells.append("-")
            else:
                cells.append(f"{v:.1f}" + (" *" if v < 40 else ""))
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("`*` marks values within ~7 points of chance, i.e. no evidence of "
             "retention.\n")

    # Where does each arm cross the "no memory" line?
    L.append("## Largest gap at which each arm still beats chance\n")
    L.append("| arm | largest gap with acc >= 45 | restricted acc there |")
    L.append("|---|---|---|")
    for arm in arms:
        ok = [(g, v) for g, v in sorted(kept[arm].items()) if v >= 45]
        if ok:
            g, v = ok[-1]
            L.append(f"| {arm} | {g} | {v:.1f} |")
        elif kept[arm]:
            L.append(f"| {arm} | none | - |")
    L.append("")

    txt = "\n".join(L)
    open("d2_readout.md", "w").write(txt)
    print(txt)
    print("wrote d2_readout.md")


if __name__ == "__main__":
    main()

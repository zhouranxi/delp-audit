"""Grid fairness and protocol diagnostics.

A sweep is a measurement only when the argmax sits strictly INSIDE the grid. When
it lands on an edge, the "best" number is a truncation artefact; comparing two
families whose optima are differently truncated compares two search budgets,
not two methods. That is the defect which started the DeLP audit (a baseline
whose penalty was insensitive to a 100-fold change of its only hyperparameter).
"""
from collections import defaultdict

from .stats import mean


def _gain(r):
    g = r.get("gain", r.get("hgain", 1.0))
    return float(g if g else 1.0)


def admissible(records):
    """Drop errored / incomplete per-seed records, mirroring bench/grid_check.load."""
    out = []
    for r in records:
        if "error" in r or "lam" not in r or "final_avg" not in r:
            continue
        out.append(r)
    return out


def cell_means(records, metric="final_avg"):
    """(arm, ntask, gain, lam) -> mean of metric over seeds."""
    acc = defaultdict(list)
    for r in admissible(records):
        try:
            v = float(r[metric])
            key = (r.get("arm", "?"), int(r.get("ntask", 0)), _gain(r), float(r["lam"]))
        except (KeyError, TypeError, ValueError):
            continue
        acc[key].append(v)
    return {k: mean(v) for k, v in acc.items()}


def argmax_edge(points, min_points=3):
    """Given [(axis_value, metric)], decide whether the optimum is grid-pinned."""
    pts = sorted((float(x), float(y)) for x, y in points)
    n = len(pts)
    res = {"n": n, "evaluable": n >= min_points and pts[0][0] != pts[-1][0],
           "best": None, "edge": False, "side": None, "lo": None, "hi": None,
           "neighbour_gap": None}
    if not res["evaluable"]:
        return res
    res["lo"], res["hi"] = pts[0][0], pts[-1][0]
    i = max(range(n), key=lambda j: pts[j][1])
    best = pts[i]
    res["best"] = best[0]
    if best[0] == res["lo"]:
        res["edge"], res["side"] = True, "floor"
    elif best[0] == res["hi"]:
        res["edge"], res["side"] = True, "ceiling"
    if i > 0 and i < n - 1:
        res["neighbour_gap"] = best[1] - max(pts[i - 1][1], pts[i + 1][1])
    return res


def boundary_audit(records, metric="final_avg", min_points=3):
    """Check the lambda axis and the gain axis separately for every family/T."""
    means = cell_means(records, metric)
    lam_axes = defaultdict(list)
    gain_axes = defaultdict(list)
    for (arm, nt, gain, lam), m in means.items():
        lam_axes[(arm, nt, gain)].append((lam, m))
        gain_axes[(arm, nt, lam)].append((gain, m))

    checked = {"lam": 0, "gain": 0}
    pinned = []
    for axis, axes in (("lam", lam_axes), ("gain", gain_axes)):
        for key, pts in axes.items():
            r = argmax_edge(pts, min_points)
            if not r["evaluable"]:
                continue
            checked[axis] += 1
            if r["edge"]:
                pinned.append({"axis": axis, "key": list(key), "side": r["side"],
                               "best": r["best"], "range": [r["lo"], r["hi"]]})
    return {"n_checked": sum(checked.values()), "checked": checked,
            "n_pinned": len(pinned), "pinned": pinned,
            "n_cells": len(means)}


def lambda_insensitive_groups(records, metric="final_avg", tol=1e-9):
    """Groups where the metric is identical across >=2 distinct lambda values.

    If the manifest declares that lambda is applied to the loss, a flat curve is
    evidence of an implementation defect, not evidence of robustness.
    """
    means = cell_means(records, metric)
    by_group = defaultdict(list)
    for (arm, nt, gain, lam), m in means.items():
        by_group[(arm, nt, gain)].append((lam, m))
    flat = []
    for key, pts in by_group.items():
        lams = {p[0] for p in pts}
        vals = [p[1] for p in pts]
        if len(lams) >= 2 and max(vals) - min(vals) <= tol:
            flat.append({"arm": key[0], "ntask": key[1], "gain": key[2],
                         "n_lambda": len(lams)})
    return flat


def joint_optimum_boundary(records, metric=None, min_points=3):
    """Check the JOINT (gain, lambda) optimum against the GLOBAL grid boundary.

    A 2-D grid legitimately has many 1-D slices whose optimum sits on that
    slice's edge: the global best simply lives on a different slice. Treating
    every slice-pinned optimum as a truncation artefact produces false alarms
    (the real 8295-cell grid reports 113/420 slice-pinned but 0/21 joint-pinned).
    Only the joint optimum decides whether the reported best is truncated.
    This is the definition used by analyze_ladgain.py section 1b.

    Selection uses held-out `val_final_avg` when present, else `final_avg`.
    """
    admissible_records = admissible(records)
    if metric is None:
        metric = ("val_final_avg" if any("val_final_avg" in r
                                         for r in admissible_records)
                  else "final_avg")
    means = cell_means(admissible_records, metric)
    gains = sorted({g for (_, _, g, _) in means})
    lams = sorted({l for (_, _, _, l) in means})
    ge, le = len(gains) >= min_points, len(lams) >= min_points
    groups = defaultdict(list)
    for (arm, nt, g, l), m in means.items():
        groups[(arm, nt)].append((g, l, m))

    pinned, ng, nl = [], 0, 0
    for key, pts in sorted(groups.items(), key=lambda kv: str(kv[0])):
        best = max(pts, key=lambda p: p[2])
        pg = bool(ge and (best[0] == gains[0] or best[0] == gains[-1]))
        pl = bool(le and (best[1] == lams[0] or best[1] == lams[-1]))
        ng += pg
        nl += pl
        if pg or pl:
            pinned.append({"key": list(key), "gain": best[0], "lam": best[1],
                           "gain_pinned": pg, "lam_pinned": pl,
                           "metric": best[2]})
    return {"n_joint": len(groups), "n_gain_pinned": ng, "n_lam_pinned": nl,
            "pinned": pinned, "metric_used": metric,
            "gain_evaluable": ge, "lam_evaluable": le,
            "gain_range": [gains[0], gains[-1]] if gains else [],
            "lambda_range": [lams[0], lams[-1]] if lams else []}


def protocol_consistency(records):
    """All arms must see the same (ntask, lam, gain) cells and the same seeds."""
    cells = defaultdict(set)
    seeds = defaultdict(lambda: defaultdict(set))
    for r in admissible(records):
        arm = r.get("arm", "?")
        key = (int(r.get("ntask", 0)), _gain(r), float(r["lam"]))
        cells[arm].add(key)
        s = r.get("seed")
        if s is not None:
            seeds[arm][key].add(s)

    issues = []
    if not cells:
        return {"uniform": False, "issues": ["no admissible records"], "arms": [],
                "n_cells": {}, "seed_counts": {}}

    reference = None
    for arm, cs in cells.items():
        if reference is None:
            reference = (arm, cs)
        elif cs != reference[1]:
            missing = sorted(reference[1] - cs)[:4]
            extra = sorted(cs - reference[1])[:4]
            issues.append("%s covers %d cells vs %s's %d (missing=%s extra=%s)"
                          % (arm, len(cs), reference[0], len(reference[1]), missing, extra))
    for arm, per_cell in seeds.items():
        counts = {len(v) for v in per_cell.values()}
        if len(counts) > 1:
            issues.append("%s has unequal seeds per cell: %s" % (arm, sorted(counts)))
        reference_count = None
        for k, v in sorted(per_cell.items(), key=lambda kv: str(kv[0])):
            if reference_count is None:
                reference_count = len(v)
            elif len(v) != reference_count:
                issues.append("%s cell %s has %d seeds vs %d elsewhere"
                              % (arm, list(k), len(v), reference_count))
                break
    return {"uniform": not issues, "issues": issues,
            "arms": sorted(cells), "n_cells": {a: len(c) for a, c in cells.items()},
            "seed_counts": {a: sorted({len(v) for v in p.values()})
                            for a, p in seeds.items()}}

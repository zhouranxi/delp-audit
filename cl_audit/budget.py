"""Anchor-budget SLO checks (companion to, not a replacement for, the six gates).

`bench/anchor_budget.py` studied what happens when the DeLP anchor stack is
truncated to M anchors. Its pilot result was reported as "+6.5 ell for -1.6
BWT" -- a trade-off, not a free win. These checks make that discipline
mechanical:

  B1  bounded policies never exceed their declared budget M
  B2  the unbounded reference actually grows with the number of tasks
  B3  any accuracy/learning-term gain is reported together with its BWT cost
  B4  errored runs are disclosed rather than silently dropped

Standard library only. No torch, no GPU, no lab server.
"""
import json
import pathlib
from collections import defaultdict

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
EXCLUDED = {"manifest.json", "budget-report.md", "budget-report.json",
            "report.md", "report.json", "ci-summary.md"}


def load_budget_records(folder):
    """Read anchor-budget result JSONs from a directory.

    Accepts one-record-per-file, a JSON list, or a dict with a ``results`` list
    (the shape written by bench/anchor_budget.py).
    """
    p = pathlib.Path(folder)
    if not p.is_dir():
        raise FileNotFoundError("not a directory: %s" % p)
    out = []
    for f in sorted(p.glob("*.json")):
        if f.name in EXCLUDED:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(d, list):
            out.extend(x for x in d if isinstance(x, dict))
        elif isinstance(d, dict) and isinstance(d.get("results"), list):
            out.extend(x for x in d["results"] if isinstance(x, dict))
        elif isinstance(d, dict):
            out.append(d)
    return out


def _num(r, *keys):
    for k in keys:
        v = r.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def check_budget(folder, max_anchors=None):
    """Return a JSON-serialisable SLO report for an anchor-budget result set."""
    records = load_budget_records(folder)
    p = pathlib.Path(folder)
    manifest = {}
    if (p / "manifest.json").is_file():
        try:
            manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        except (ValueError, OSError):
            manifest = {}

    checks = []

    def add(cid, title, status, detail, evidence=None):
        checks.append({"id": cid, "title": title, "status": status,
                       "detail": detail, "evidence": evidence or {}})

    if not records:
        add("B0", "Budget records present", SKIP,
            "No anchor-budget JSON records found.")
        return _finish(folder, records, checks)

    clean = [r for r in records if "error" not in r]
    errored = [r for r in records if "error" in r]

    # B1 -- bounded policies must respect M (and the optional SLO ceiling).
    violations, inspected = [], 0
    for i, r in enumerate(clean):
        policy = str(r.get("policy", ""))
        if policy in ("", "unbounded"):
            continue
        n = _num(r, "n_anchors_final", "anchors_final", "n_anchors")
        declared_m = _num(r, "M", "budget", "max_anchors")
        limits = [x for x in (declared_m, max_anchors) if x is not None]
        if n is None or not limits:
            continue
        inspected += 1
        limit = min(limits)
        if n > limit:
            violations.append({"index": i, "policy": policy, "M": declared_m,
                               "n_anchors_final": n, "limit": limit,
                               "ntask": r.get("ntask"), "seed": r.get("seed")})
    if not inspected:
        add("B1", "Bounded policies respect their anchor budget", SKIP,
            "No bounded record carries both an anchor count and a budget M.")
    elif violations:
        add("B1", "Bounded policies respect their anchor budget", FAIL,
            "%d/%d bounded runs exceed their anchor budget. The claimed "
            "bounded-memory property is not implemented."
            % (len(violations), inspected),
            {"violations": violations[:8], "n_violations": len(violations)})
    else:
        add("B1", "Bounded policies respect their anchor budget", PASS,
            "%d/%d bounded runs stay within M%s." %
            (inspected, inspected,
             " and the SLO ceiling %g" % max_anchors if max_anchors else ""),
            {"n_inspected": inspected})

    # B2 -- the unbounded reference must actually grow with task count.
    by_t = defaultdict(list)
    for r in clean:
        if str(r.get("policy", "")) != "unbounded":
            continue
        n = _num(r, "n_anchors_final", "anchors_final", "n_anchors")
        t = _num(r, "ntask", "n_tasks", "T")
        if n is not None and t is not None:
            by_t[t].append(n)
    if len(by_t) < 2:
        add("B2", "Unbounded reference grows with task count", SKIP,
            "Need >=2 task horizons with unbounded-run anchor counts.")
    else:
        means = {t: _mean(v) for t, v in by_t.items()}
        lo, hi = min(means), max(means)
        if means[hi] <= means[lo]:
            add("B2", "Unbounded reference grows with task count", WARN,
                "Unbounded anchor count does not increase from T=%g to T=%g "
                "(%.2f -> %.2f); the reference arm may not have trained all tasks."
                % (lo, hi, means[lo], means[hi]), {"by_task": means})
        else:
            add("B2", "Unbounded reference grows with task count", PASS,
                "Unbounded anchors grow %.2f -> %.2f from T=%g to T=%g, "
                "as expected for the reference arm."
                % (means[lo], means[hi], lo, hi), {"by_task": means})

    # B3 -- a reported learning-term gain must come with its BWT cost.
    groups = defaultdict(lambda: defaultdict(list))
    keys = ("policy", "M", "ntask", "lam", "seed")
    for r in clean:
        key = tuple(str(r.get(k, "?")) for k in keys[2:])
        groups[key][str(r.get("policy", "?"))].append(r)
    tradeoffs, undeclared = [], []
    declared = bool(manifest.get("tradeoff_declared"))
    for key, by_policy in groups.items():
        ref = by_policy.get("unbounded") or []
        if not ref:
            continue
        for policy, runs in by_policy.items():
            if policy == "unbounded":
                continue
            for metric in ("ell", "final_avg"):
                a = _mean([_num(x, metric) for x in runs
                           if _num(x, metric) is not None])
                b = _mean([_num(x, metric) for x in ref
                           if _num(x, metric) is not None])
                if a is None or b is None:
                    continue
                da = a - b
                bw_a = _mean([_num(x, "bwt") for x in runs
                              if _num(x, "bwt") is not None])
                bw_b = _mean([_num(x, "bwt") for x in ref
                              if _num(x, "bwt") is not None])
                if da > 0 and bw_a is not None and bw_b is not None \
                        and bw_a < bw_b:
                    tradeoffs.append({"key": list(key), "policy": policy,
                                      "metric": metric, "delta": round(da, 3),
                                      "delta_bwt": round(bw_a - bw_b, 3)})
                    if not declared:
                        undeclared.append(policy)
    if not tradeoffs:
        add("B3", "Gain/cost pair is disclosed", SKIP,
            "No bounded-vs-unbounded pair shows a positive accuracy/learning "
            "gain together with a BWT cost.")
    elif undeclared:
        add("B3", "Gain/cost pair is disclosed", WARN,
            "%d bounded arm(s) improve accuracy or the learning term while "
            "worsening BWT. Report both numbers (e.g. '+6.5 ell for -1.6 BWT'); "
            "do not present the gain alone. Set manifest.tradeoff_declared once "
            "the trade-off is stated explicitly."
            % len(set(undeclared)), {"tradeoffs": tradeoffs[:8],
                                     "n_tradeoffs": len(tradeoffs)})
    else:
        add("B3", "Gain/cost pair is disclosed", PASS,
            "%d gain/cost pair(s) found and the manifest declares the trade-off."
            % len(tradeoffs), {"tradeoffs": tradeoffs[:8],
                               "n_tradeoffs": len(tradeoffs)})

    # B4 -- errored runs.
    if errored:
        add("B4", "Failed runs are disclosed", WARN,
            "%d/%d runs carry an error field; dropping them silently changes "
            "the seed count." % (len(errored), len(records)),
            {"errors": [str(r.get("error"))[:120] for r in errored[:5]]})
    else:
        add("B4", "Failed runs are disclosed", PASS,
            "%d/%d runs completed without an error field."
            % (len(clean), len(records)))

    return _finish(folder, records, checks)


def _finish(folder, records, checks):
    summary = {s: sum(1 for c in checks if c["status"] == s)
               for s in (PASS, WARN, FAIL, SKIP)}
    return {"tool": "cl-audit budget", "folder": str(folder),
            "n_records": len(records), "checks": checks, "summary": summary,
            "ok": summary[FAIL] == 0}


def exit_code(report, strict=False):
    if report["summary"][FAIL]:
        return 2
    if strict and report["summary"][WARN]:
        return 1
    return 0


def render_budget(report):
    s = report["summary"]
    lines = [
        "# cl-audit budget — anchor memory SLO",
        "",
        "Records: **%d**  |  PASS %d / WARN %d / FAIL %d / SKIP %d"
        % (report["n_records"], s["PASS"], s["WARN"], s["FAIL"], s["SKIP"]),
        "",
        "| Check | Status | Finding |",
        "|---|---|---|",
    ]
    for c in report["checks"]:
        detail = str(c["detail"]).replace("|", "\\|")
        lines.append("| %s %s | **%s** | %s |"
                     % (c["id"], c["title"], c["status"], detail))
    lines += ["", "---", "",
              "Companion SLO check; it does not alter the six publication gates.", ""]
    return "\n".join(lines)

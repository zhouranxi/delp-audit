"""The six gates.

G1 declaration-implementation consistency : declared lambda/seeds/self-check vs records
G2 tuning fairness                        : no argmax pinned to a grid edge
G3 baseline protocol parity               : same cells and seeds for every arm
G4 statistical validity                   : declared significance vs paired tests
G5 forgetting direction                   : accuracy gains must not worsen BWT
G6 mechanism attribution                  : separate architectural/gain effects

Status semantics: FAIL blocks a claim, WARN forces an explicit caveat, SKIP means
the input did not carry enough evidence to judge (and is reported as such, never
silently as a pass).
"""
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from .grid import (_gain, boundary_audit, joint_optimum_boundary,
                   lambda_insensitive_groups, protocol_consistency)
from .stats import t_paired

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Result:
    gate: str
    title: str
    status: str
    detail: str
    evidence: dict = field(default_factory=dict)


def _g1(manifest, records):
    ev, problems = {}, []
    if not records:
        return Result("G1", "Declaration-implementation consistency", SKIP,
                      "No per-seed records; declaration cannot be verified.", ev)

    if manifest.get("lambda_applied"):
        flat = lambda_insensitive_groups(records)
        ev["lambda_flat_groups"] = flat
        if flat:
            problems.append("%d (arm, T, gain) groups show an identical final_avg "
                            "across >=2 distinct lambda values while the manifest "
                            "declares lambda is applied to the loss"
                            % len(flat))

    if "n_seeds" in manifest:
        observed = len({r.get("seed") for r in records if r.get("seed") is not None})
        ev["seeds"] = {"declared": manifest["n_seeds"], "observed": observed}
        if observed and observed != manifest["n_seeds"]:
            problems.append("declared %s seeds but observed %d"
                            % (manifest["n_seeds"], observed))

    sc = manifest.get("selfcheck")
    if sc:
        ev["selfcheck"] = sc
        if sc.get("passed") != sc.get("pairs") or not sc.get("pairs"):
            problems.append("implementation-equivalence self-check did not pass")

    checked = [k for k in ("lambda_applied", "n_seeds", "selfcheck") if k in manifest]
    if not checked:
        return Result("G1", "Declaration-implementation consistency", SKIP,
                      "Manifest declares nothing checkable.", ev)
    if problems:
        return Result("G1", "Declaration-implementation consistency", FAIL,
                      "; ".join(problems), ev)
    return Result("G1", "Declaration-implementation consistency", PASS,
                  "Checked: %s." % ", ".join(checked), ev)


def _g2(records, manifest):
    """Judge the JOINT (gain, lambda) optimum, not every 1-D slice.

    In a 2-D grid most slices legitimately have their optimum on the slice edge
    (the global best lives on another slice). Only a joint optimum pinned to the
    global grid boundary is a truncation artefact. See grid.joint_optimum_boundary.
    """
    joint = joint_optimum_boundary(records)
    slices = boundary_audit(records)
    declared = manifest.get("boundary")
    title = "Tuning fairness (joint optimum not grid-pinned)"
    ev = {
        "joint": {"n_joint": joint["n_joint"],
                  "n_gain_pinned": joint["n_gain_pinned"],
                  "n_lam_pinned": joint["n_lam_pinned"],
                  "metric_used": joint["metric_used"],
                  "gain_range": joint["gain_range"],
                  "lambda_range": joint["lambda_range"],
                  "pinned": joint["pinned"][:8]},
        "slices": {"n_checked": slices["n_checked"],
                   "n_pinned": slices["n_pinned"],
                   "note": ("1-D slice optima; edge-pinned slices are expected "
                            "in a 2-D grid and do not by themselves truncate "
                            "the joint optimum")},
    }
    if declared:
        ev["declared_boundary"] = declared
    slice_note = ("%d/%d one-dimensional slices are edge-pinned (expected in a "
                  "2-D grid; the joint optimum decides truncation)."
                  % (slices["n_pinned"], slices["n_checked"]))

    if joint["n_joint"]:
        if declared:
            exp = (int(declared.get("gain_pinned", 0)),
                   int(declared.get("lambda_pinned", 0)),
                   int(declared.get("total", joint["n_joint"])))
            got = (joint["n_gain_pinned"], joint["n_lam_pinned"],
                   joint["n_joint"])
            if exp != got:
                return Result("G2", title, FAIL,
                              "Manifest boundary audit %s disagrees with the "
                              "recomputation from records %s." % (exp, got), ev)
        bad = joint["n_gain_pinned"] + joint["n_lam_pinned"]
        if bad:
            return Result("G2", title, FAIL,
                          "%d/%d JOINT optima are pinned to the global grid edge; "
                          "widen the grid. %s" % (bad, joint["n_joint"], slice_note), ev)
        return Result("G2", title, PASS,
                      "0/%d joint (gain, lambda) optima pinned; manifest "
                      "cross-check %s. %s"
                      % (joint["n_joint"], "OK" if declared else "unavailable",
                         slice_note), ev)

    if declared:
        pinned = (int(declared.get("gain_pinned", 0))
                  + int(declared.get("lambda_pinned", 0)))
        if pinned:
            return Result("G2", title, FAIL,
                          "Declared boundary audit reports %d pinned optima." % pinned, ev)
        return Result("G2", title, PASS,
                      "Declared boundary audit: 0/%s joint optima pinned "
                      "(no raw records to recompute against)."
                      % declared.get("total", "?"), ev)
    return Result("G2", title, SKIP,
                  "No grid records and no declared boundary audit.", ev)


def _g3(records):
    pc = protocol_consistency(records)
    ev = {"arms": pc["arms"], "n_cells": pc["n_cells"],
          "seed_counts": pc["seed_counts"], "issues": pc["issues"][:8]}
    if not pc["arms"]:
        return Result("G3", "Baseline protocol parity", SKIP,
                      "No admissible records.", ev)
    if pc["issues"]:
        return Result("G3", "Baseline protocol parity", FAIL,
                      "; ".join(pc["issues"][:3]), ev)
    return Result("G3", "Baseline protocol parity", PASS,
                  "%d arms share identical (T, gain, lambda) cells and seed counts."
                  % len(pc["arms"]), ev)


def _paired_from_records(manifest, records):
    a, b = manifest["primary_comparison"]
    pairs = defaultdict(dict)
    for r in records:
        if "error" in r or "final_avg" not in r or "lam" not in r:
            continue
        key = (int(r.get("ntask", 0)), r.get("seed"), float(r["lam"]), _gain(r))
        pairs[key][r.get("arm")] = float(r["final_avg"])
    by_t = defaultdict(list)
    for (nt, seed, lam, gain), d in pairs.items():
        if a in d and b in d:
            by_t[nt].append((d[a], d[b]))
    out = []
    for nt, ps in sorted(by_t.items()):
        md, t, p = t_paired([x for x, _ in ps], [y for _, y in ps])
        out.append({"T": nt, "dacc": round(md, 4), "p": round(p, 6),
                    "n_pairs": len(ps), "source": "records"})
    return out


def _g4(manifest, records):
    effects = manifest.get("effects") or []
    primary = [e for e in effects if e.get("primary")]
    if primary:
        measured = [{"T": e.get("T"), "dacc": e["dacc"], "p": e["p"],
                     "source": "effects"} for e in primary]
    elif records and manifest.get("primary_comparison"):
        measured = _paired_from_records(manifest, records)
    else:
        measured = []

    ctrl = manifest.get("ours_vs_control") or []
    ev = {"measured": measured, "ours_vs_control": ctrl}
    if not measured and not ctrl:
        return Result("G4", "Statistical validity of the headline claim", SKIP,
                      "No paired comparison or conclusion-level effects supplied.", ev)

    sig = [m for m in measured if m.get("p") is not None and m["p"] < 0.05]
    ev["n_significant"] = len(sig)
    ev["n_comparisons"] = len(measured)
    declared = manifest.get("declared_significant")

    if declared is True and not sig:
        return Result("G4", "Statistical validity of the headline claim", FAIL,
                      "Manifest declares a significant effect, but none of the %d "
                      "comparisons reaches p<0.05." % len(measured), ev)
    if ctrl and all(c["p"] >= 0.05 for c in ctrl):
        ev["control_verdict"] = "no significant net advantage over the dendrite-free control"
    if 0 < len(sig) < len(measured):
        return Result("G4", "Statistical validity of the headline claim", WARN,
                      "Only %d/%d comparisons are significant; the effect is not "
                      "stable across task horizons. Report per-horizon results."
                      % (len(sig), len(measured)), ev)
    if not sig:
        return Result("G4", "Statistical validity of the headline claim", PASS,
                      "No significant effect; the negative result is reported as such.", ev)
    return Result("G4", "Statistical validity of the headline claim", PASS,
                  "%d/%d comparisons significant (check multiplicity before claiming)."
                  % (len(sig), len(measured)), ev)


def _g5(manifest):
    effects = manifest.get("effects") or []
    primary = [e for e in effects if e.get("primary")]
    significant = [e for e in primary if e.get("p") is not None and e["p"] < 0.05]
    ev = {"significant_primary": significant}
    if not primary:
        return Result("G5", "Forgetting direction (BWT)", SKIP,
                      "No primary mechanism effects supplied.", ev)
    if not significant:
        return Result("G5", "Forgetting direction (BWT)", PASS,
                      "No significant accuracy effect, hence no anti-forgetting "
                      "claim to qualify.", ev)
    conflicts = [e for e in significant
                 if e.get("dbwt") is not None and e["dacc"] > 0 and e["dbwt"] < 0]
    ev["conflicts"] = conflicts
    if conflicts:
        return Result("G5", "Forgetting direction (BWT)", WARN,
                      "Accuracy improves but BWT gets worse (dBWT=%+.2f, p=%s). "
                      "This cannot be described as an anti-forgetting mechanism."
                      % (conflicts[0]["dbwt"], conflicts[0].get("p_bwt", "?")), ev)
    zero = [e for e in significant if e.get("dbwt") is not None and e["dbwt"] == 0]
    if zero:
        return Result("G5", "Forgetting direction (BWT)", WARN,
                      "Accuracy improves with no measured BWT improvement; do not "
                      "claim reduced forgetting.", ev)
    return Result("G5", "Forgetting direction (BWT)", PASS,
                  "Accuracy and BWT move in compatible directions.", ev)


def _g6(manifest):
    effects = manifest.get("effects") or []
    if not effects:
        return Result("G6", "Mechanism attribution", SKIP,
                      "No per-step effect table supplied.", {})
    arch = [e for e in effects if e.get("kind") == "architecture"]
    mech = [e for e in effects if e.get("kind") == "mechanism"]
    big_arch = max(arch, key=lambda e: abs(e["dacc"])) if arch else None
    big_mech = max(mech, key=lambda e: abs(e["dacc"])) if mech else None
    ev = {"largest_architectural": big_arch, "largest_mechanism": big_mech}
    if big_arch and (big_mech is None or
                     abs(big_arch["dacc"]) > abs(big_mech["dacc"])):
        # The step label comes from the manifest, not from DeLP vocabulary: the
        # same gate must read correctly for a plain-EWC or any other method.
        arch_label = big_arch.get("step", "architectural change")
        return Result("G6", "Mechanism attribution", WARN,
                      "The largest effect is architectural (%s: %+.2f points), "
                      "not a protection mechanism (largest mechanism step: "
                      "%+.2f). The gain cannot be attributed to the method."
                      % (arch_label, big_arch["dacc"],
                         big_mech["dacc"] if big_mech else 0.0), ev)
    if big_mech and big_mech["p"] >= 0.05:
        return Result("G6", "Mechanism attribution", WARN,
                      "No mechanism step reaches significance.", ev)
    return Result("G6", "Mechanism attribution", PASS,
                  "Largest effect is attributable to a declared mechanism step.", ev)


def check_experiment(manifest, records, notes=None):
    """Run all six gates and return a JSON-serialisable report dict."""
    notes = list(notes or [])
    results = [
        _g1(manifest, records),
        _g2(records, manifest),
        _g3(records),
        _g4(manifest, records),
        _g5(manifest),
        _g6(manifest),
    ]
    summary = {s: sum(1 for r in results if r.status == s)
               for s in (PASS, WARN, FAIL, SKIP)}
    return {
        "tool": "cl-audit",
        "name": manifest.get("name", "(unnamed experiment)"),
        "n_records": len(records),
        "notes": notes,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }


def exit_code(report, strict=False):
    if report["summary"][FAIL]:
        return 2
    if strict and report["summary"][WARN]:
        return 1
    return 0

"""Built-in known-good / deliberately broken fixtures and the final readings.

IMPORTANT HONESTY NOTE
----------------------
`FINAL_READINGS` is a CONCLUSION-LEVEL transcription of
`bench/reports/ladgain2.md` (8295 cells, 2026-11-11). It is NOT the raw per-seed
data; those JSON files live on the lab server and are not shipped here. The
numbers are pinned so that the six gates can be regression-tested against the
actual published conclusions (in particular: usage weighting must be reported as
NOT an anti-forgetting mechanism, and the sigmoid removal must be attributed to
architecture). The synthetic per-seed directories exist only to exercise the
checker's logic -- they are not experimental results and must never be cited.
"""
import json
import math
import pathlib
import tempfile

FINAL_READINGS = {
    "source": "bench/reports/ladgain2.md (8295 cells, 2026-11-11)",
    "grid_size": {"files": 8295, "rung_T": 21, "gain_lambda_cells": 1659, "seeds": 5},
    "grid_ranges": {"gain": [0.25, 1024], "lambda": [0.1, 100000]},
    "selfcheck": {"pairs": 1185, "passed": 1185, "rule": "mlp_nosig == ad_nocx"},
    "boundary": {"gain_pinned": 0, "lambda_pinned": 0, "total": 21},
    "activation_removal": [
        {"T": 5, "dacc": 12.03, "p": 0.000, "dbwt": 2.70, "p_bwt": 0.000},
        {"T": 10, "dacc": 15.35, "p": 0.000, "dbwt": 6.96, "p_bwt": 0.000},
        {"T": 20, "dacc": 12.76, "p": 0.001, "dbwt": 2.71, "p_bwt": 0.081},
    ],
    "usage_weighting": [
        {"T": 5, "dacc": -0.03, "p": 0.761, "dbwt": -0.28, "p_bwt": 0.025},
        {"T": 10, "dacc": 0.22, "p": 0.519, "dbwt": -0.17, "p_bwt": 0.330},
        {"T": 20, "dacc": 1.42, "p": 0.041, "dbwt": -1.18, "p_bwt": 0.033},
    ],
    "ours_vs_control": [
        {"T": 5, "dacc": -0.06, "p": 0.500},
        {"T": 10, "dacc": 0.08, "p": 0.761},
        {"T": 20, "dacc": 1.33, "p": 0.187},
    ],
}


def final_manifest():
    """A conclusion-level manifest for the 8295-cell report (no per-seed files)."""
    effects = []
    for e in FINAL_READINGS["activation_removal"]:
        effects.append({"step": "activation: kwta(sigmoid(t)) -> kwta(0.5t)",
                        "kind": "architecture", "primary": False, **e})
    for e in FINAL_READINGS["usage_weighting"]:
        effects.append({"step": "usage-weighted Fisher",
                        "kind": "mechanism", "primary": True, **e})
    return {
        "name": "delp-ladder joint gain x lambda (conclusion-level, 8295 cells)",
        "claim": "anti_forgetting",
        "declared_significant": False,
        "lambda_applied": True,
        "primary_comparison": ["ad_usage", "mlp_nosig"],
        "effects": effects,
        "selfcheck": FINAL_READINGS["selfcheck"],
        "boundary": FINAL_READINGS["boundary"],
        "ours_vs_control": FINAL_READINGS["ours_vs_control"],
        "note": ("Conclusion-level summary only. Raw per-seed JSON is NOT included; "
                 "gates G1-G3 are skipped for this fixture."),
    }


LAMS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
TS = [5, 10, 20]
ARMS = ["mlp_nosig", "ad_nocx", "ad_usage"]
SEEDS = [0, 1, 2, 3, 4]


def _acc(arm, nt, lam, seed, kind):
    base = 90.0 - 6.0 * (math.log10(lam) - math.log10(30.0)) ** 2
    base -= {5: 0.0, 10: 3.0, 20: 8.0}[nt]
    base += {"mlp_nosig": 0.0, "ad_nocx": 0.0, "ad_usage": 0.4}[arm]
    wobble = 0.15 * math.sin(seed * 1.7)
    if kind == "lambda_not_applied" and arm == "ad_usage":
        return round(84.0 + wobble, 6)
    if kind == "edge_pinned":
        return round(60.0 + 12.0 * math.log10(lam) + wobble, 6)
    return round(base + wobble, 6)


def make_dir(kind="good", root=None):
    """Create a synthetic per-seed result directory.

    kind: good | adaptive_2d | ewc_generic | lambda_not_applied | edge_pinned |
          unequal_protocol | final_readings
    """
    d = pathlib.Path(tempfile.mkdtemp(prefix="claudit_", dir=root))
    if kind == "final_readings":
        (d / "manifest.json").write_text(json.dumps(final_manifest(), indent=2),
                                          encoding="utf-8")
        return d

    if kind == "ewc_generic":
        # P3 hard target: a NON-DeLP method must run all six gates without any
        # import error and without a single DeLP/MNIST assumption. Column names
        # are deliberately foreign; there is no gain axis. Adapter does the rest.
        ewc_lams = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        for method in ("sgd", "ewc"):
            for nt in TS:
                for lam in ewc_lams:
                    for seed in SEEDS:
                        acc = (88.0
                               - 4.0 * (math.log10(lam) - 2.0) ** 2
                               - {5: 0.0, 10: 3.0, 20: 7.0}[nt]
                               + {"sgd": 0.0, "ewc": 1.2}[method]
                               + 0.12 * math.sin(seed * 1.9))
                        rec = {"method": method, "n_tasks": nt,
                               "ewc_lambda": lam, "run_id": seed,
                               "avg_acc": round(acc, 6),
                               "heldout_acc": round(acc - 0.3, 6),
                               "backward_transfer": -2.0 - 0.1 * seed,
                               "dataset": "synthetic-cifar-like"}
                        name = "ewc_%s_T%d_l%g_r%d.json" % (method, nt, lam, seed)
                        (d / name).write_text(json.dumps(rec), encoding="utf-8")
        effects = [{"step": "wider hidden layer", "kind": "architecture",
                    "primary": False, "T": 10, "dacc": 0.20, "p": 0.62,
                    "dbwt": 0.05, "p_bwt": 0.80}]
        for nt, dacc, p, dbwt, pb in ((5, 0.31, 0.42, 0.10, 0.61),
                                      (10, 0.55, 0.19, -0.05, 0.77),
                                      (20, 1.10, 0.031, -0.40, 0.04)):
            effects.append({"step": "EWC quadratic penalty", "kind": "mechanism",
                            "primary": True, "T": nt, "dacc": dacc, "p": p,
                            "dbwt": dbwt, "p_bwt": pb})
        manifest = {
            "name": "plain EWC (synthetic non-DeLP format fixture)",
            "claim": "anti_forgetting",
            "adapter": "ewc_generic",
            "declared_significant": False,
            "lambda_applied": True,
            "n_seeds": len(SEEDS),
            "selfcheck": {"pairs": 150, "passed": 150,
                          "rule": "adapter field-map round trip"},
            "boundary": {"gain_pinned": 0, "lambda_pinned": 0, "total": 6},
            "primary_comparison": ["ewc", "sgd"],
            "effects": effects,
            "ours_vs_control": [{"T": 5, "dacc": 0.11, "p": 0.63},
                                {"T": 10, "dacc": 0.24, "p": 0.51},
                                {"T": 20, "dacc": 0.48, "p": 0.29}],
            "note": ("Synthetic non-DeLP FORMAT fixture, not an experimental "
                     "result. It proves the six gates work on foreign column "
                     "names with no dendritic/MNIST assumption."),
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")
        return d

    if kind == "adaptive_2d":
        # A 2-D grid whose 1-D slices are heavily edge-pinned but whose JOINT
        # (gain, lambda) optimum is interior. Regression fixture for the G2 bug
        # found on the real 8295-cell grid (113/420 slices pinned, 0/21 joint).
        gains = [1.0, 4.0, 16.0]
        lams = [1.0, 3.0, 10.0, 30.0, 100.0]
        for arm in ARMS[:2]:
            for nt in TS[:2]:
                for g in gains:
                    target = 10.0 ** (1.0 - 2.5 * (math.log10(g) - math.log10(4.0)))
                    for lam in lams:
                        for seed in SEEDS:
                            acc = (92.0
                                   - 3.0 * (math.log10(g) - math.log10(4.0)) ** 2
                                   - 5.0 * (math.log10(lam) - math.log10(target)) ** 2
                                   + 0.15 * math.sin(seed * 1.7))
                            rec = {"arm": arm, "ntask": nt, "lam": lam, "gain": g,
                                   "seed": seed, "final_avg": round(acc, 6),
                                   "val_final_avg": round(acc, 6), "bwt": -2.0}
                            name = ("%s_T%d_g%g_l%g_s%d.json"
                                    % (arm, nt, g, lam, seed))
                            (d / name).write_text(json.dumps(rec), encoding="utf-8")
        manifest = {
            "name": "adaptive 2-D grid (slice-pinned, joint-interior)",
            "claim": "anti_forgetting",
            "lambda_applied": True,
            "n_seeds": len(SEEDS),
            "boundary": {"gain_pinned": 0, "lambda_pinned": 0, "total": 4},
            "note": "Synthetic regression fixture; slice pinning is expected here.",
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                          encoding="utf-8")
        return d

    for arm in ARMS:
        for nt in TS:
            for lam in LAMS:
                if kind == "unequal_protocol" and arm == "ad_usage" and lam == LAMS[-1]:
                    continue
                for seed in SEEDS:
                    rec = {"arm": arm, "ntask": nt, "lam": lam, "gain": 1.0,
                           "seed": seed, "final_avg": _acc(arm, nt, lam, seed, kind),
                           "bwt": -2.0 - 0.1 * seed}
                    name = "%s_T%d_g1_l%g_s%d.json" % (arm, nt, lam, seed)
                    (d / name).write_text(json.dumps(rec), encoding="utf-8")

    manifest = {
        "name": "synthetic %s fixture" % kind,
        "claim": "anti_forgetting",
        "declared_significant": False,
        "lambda_applied": True,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "primary_comparison": ["ad_usage", "mlp_nosig"],
        "note": "Synthetic data for checker regression tests. Not an experimental result.",
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return d

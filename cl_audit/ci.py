"""The fixed CI fixture suite, shared by `cl-audit ci` and `cl-audit selftest`.

CI must do more than assert that the checker runs: it must prove that the
checker *catches* deliberately broken inputs and that a non-DeLP method passes
through the adapter layer with all six gates evaluated. A green suite whose
known-bad fixtures are never exercised is worthless.
"""
import tempfile

from .checks import FAIL, SKIP, WARN
from .fixtures import make_dir

# (fixture kind, gate that must FAIL, or None for a known-good fixture)
CASES = (
    ("good", None),
    ("adaptive_2d", None),
    ("ewc_generic", None),
    ("lambda_not_applied", "G1"),
    ("edge_pinned", "G2"),
    ("unequal_protocol", "G3"),
)


def run_fixture_suite(runner):
    """Run every fixture through ``runner(folder) -> report dict``.

    Returns {"ok", "rows", "problems", "final_readings", "n_broken_caught"}.
    The suite deliberately does not depend on GPU, torch, network or lab data.
    """
    problems, rows = [], []
    n_broken = n_caught = 0
    with tempfile.TemporaryDirectory() as tmp:
        for kind, expected in CASES:
            rep = runner(make_dir(kind, root=tmp))
            failed = [r["gate"] for r in rep["results"] if r["status"] == FAIL]
            s = rep["summary"]
            ok = True
            if expected is None:
                if failed:
                    problems.append("%s (known-good) FAILed: %s"
                                    % (kind, ", ".join(failed)))
                    ok = False
                if kind == "adaptive_2d":
                    g2 = next(r for r in rep["results"] if r["gate"] == "G2")
                    if not g2["evidence"]["slices"]["n_pinned"]:
                        problems.append("adaptive_2d no longer exercises 1-D "
                                        "slice pinning")
                        ok = False
                if kind == "ewc_generic":
                    skipped = [r["gate"] for r in rep["results"]
                               if r["status"] == SKIP]
                    if skipped:
                        problems.append("ewc_generic left gates unevaluated: %s"
                                        % ", ".join(skipped))
                        ok = False
            else:
                n_broken += 1
                if expected not in failed:
                    problems.append("%s was not caught by %s (got %s)"
                                    % (kind, expected, ", ".join(failed) or "no FAIL"))
                    ok = False
                else:
                    n_caught += 1
            rows.append({
                "fixture": kind,
                "expected_fail_gate": expected,
                "observed_fail_gates": failed,
                "summary": s,
                "verdict": "OK" if ok else "BROKEN",
            })

        rep = runner(make_dir("final_readings", root=tmp))
        by_gate = {r["gate"]: r for r in rep["results"]}
        finals = {g: by_gate[g]["status"] for g in ("G4", "G5", "G6")}
        if finals["G4"] != WARN:
            problems.append("final readings: G4 should WARN (effect not stable)")
        if finals["G5"] != WARN or "anti-forgetting" not in by_gate["G5"]["detail"]:
            problems.append("final readings: G5 must WARN 'not anti-forgetting'")
        if finals["G6"] != WARN or "architectural" not in by_gate["G6"]["detail"]:
            problems.append("final readings: G6 must WARN 'architectural effect'")
        if rep["summary"][FAIL]:
            problems.append("final readings should not produce a FAIL")

    return {"ok": not problems, "rows": rows, "problems": problems,
            "final_readings": finals, "n_broken": n_broken,
            "n_caught": n_caught}


def render_ci_markdown(suite):
    lines = [
        "# cl-audit CI",
        "",
        "**%s** — %d/%d deliberately broken fixtures caught; %d known-good fixtures."
        % ("OK" if suite["ok"] else "FAILED",
           suite["n_caught"], suite["n_broken"],
           sum(1 for r in suite["rows"] if r["expected_fail_gate"] is None)),
        "",
        "| Fixture | Must FAIL | Observed FAIL | PASS/WARN/FAIL/SKIP | Verdict |",
        "|---|---|---|---|---|",
    ]
    for r in suite["rows"]:
        s = r["summary"]
        lines.append("| `%s` | %s | %s | %d/%d/%d/%d | %s |"
                     % (r["fixture"], r["expected_fail_gate"] or "—",
                        ", ".join(r["observed_fail_gates"]) or "—",
                        s["PASS"], s["WARN"], s["FAIL"], s["SKIP"],
                        r["verdict"]))
    lines += [
        "",
        "Published conclusions (`final_readings`): G4=%s G5=%s G6=%s — the tool "
        "must classify the 2026-11-11 re-analysis as WARN, never as a pass."
        % (suite["final_readings"]["G4"], suite["final_readings"]["G5"],
           suite["final_readings"]["G6"]),
        "",
        "`ewc_generic` is a synthetic **plain-EWC** result table with completely "
        "different column names and no gain axis; it exercises the MethodAdapter "
        "layer. It is a format fixture, not an experimental result.",
    ]
    if suite["problems"]:
        lines += ["", "## Problems", ""] + ["- " + p for p in suite["problems"]]
    lines += ["", "---", "", "Generated by `cl-audit ci`. Offline; no GPU or "
              "lab-server data required.", ""]
    return "\n".join(lines)

"""Command line entry point for cl-audit.

  cl-audit check <dir> [--adapter NAME] [--out report.md] [--strict]
  cl-audit selftest
  cl-audit ci [--out ci-summary.md] [--strict]
  cl-audit verify <archive-dir>
  cl-audit budget <dir> [--max-anchors M] [--out report.md] [--strict]
"""
import argparse
import json
import pathlib
import sys

from . import archive, budget, ci
from .checks import FAIL, WARN, check_experiment, exit_code
from .loader import load_experiment
from .report import render_markdown


def _run_dir(folder, adapter=None):
    manifest, records, notes = load_experiment(folder, adapter=adapter)
    return check_experiment(manifest, records, notes)


def _write_pair(out_md, markdown, payload):
    out_md = pathlib.Path(out_md)
    out_json = out_md.with_suffix(".json")
    out_md.write_text(markdown, encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return out_json


def cmd_check(args):
    report = _run_dir(args.results, adapter=args.adapter)
    out_json = _write_pair(args.out or "cl-audit-report.md",
                           render_markdown(report), report)
    print(render_markdown(report))
    print("wrote %s and %s" % (args.out or "cl-audit-report.md", out_json))
    return exit_code(report, args.strict)


def cmd_selftest(args):
    """Known-good fixtures must not FAIL; broken fixtures must be caught.

    Also pins the 2026-11-11 conclusions: usage weighting must be WARNed as NOT
    an anti-forgetting mechanism, and the sigmoid removal must be attributed to
    architecture rather than to a protection mechanism.
    """
    suite = ci.run_fixture_suite(_run_dir)
    print("cl-audit selftest")
    for r in suite["rows"]:
        print("  %-20s FAIL=%s"
              % (r["fixture"], ",".join(r["observed_fail_gates"]) or "-"))
    f = suite["final_readings"]
    print("  %-20s G4=%s G5=%s G6=%s"
          % ("final_readings", f["G4"], f["G5"], f["G6"]))
    if not suite["ok"]:
        for problem in suite["problems"]:
            print("  FAIL: " + problem, file=sys.stderr)
        print("SELFTEST FAILED (%d)" % len(suite["problems"]))
        return 1
    print("SELFTEST OK: %d/%d broken fixtures caught; final readings classified "
          "correctly." % (suite["n_caught"], suite["n_broken"]))
    return 0


def cmd_ci(args):
    """CI gate: non-zero if any fixture misbehaves (a FAIL that is not expected)."""
    suite = ci.run_fixture_suite(_run_dir)
    md = ci.render_ci_markdown(suite)
    if args.out:
        pathlib.Path(args.out).write_text(md, encoding="utf-8")
        print("wrote %s" % args.out)
    print(md)
    if not suite["ok"]:
        return 2
    if args.strict and any(r["summary"][WARN] for r in suite["rows"]):
        return 1
    return 0


def cmd_verify(args):
    res = archive.verify_archive(args.archive)
    print(archive.render_verify(res))
    return 0 if res["ok"] else 2


def cmd_budget(args):
    rep = budget.check_budget(args.results, args.max_anchors)
    out_md = args.out or "cl-audit-budget.md"
    out_json = _write_pair(out_md, budget.render_budget(rep), rep)
    print(budget.render_budget(rep))
    print("wrote %s and %s" % (out_md, out_json))
    return budget.exit_code(rep, args.strict)


def main(argv=None):
    # Allow callers to pass pathlib.Path objects (the shell always passes strings).
    if argv is not None:
        argv = [str(a) for a in argv]
    ap = argparse.ArgumentParser(
        prog="cl-audit",
        description="Six-gate pre-publication audit for continual-learning claims.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="audit a result directory")
    p.add_argument("results", help="directory with manifest.json and per-seed JSON")
    p.add_argument("--out", default="", help="output markdown path (json alongside it)")
    p.add_argument("--adapter", default=None,
                   help="method adapter name (overrides manifest.adapter); "
                        "try: identity, delp_ladder, ewc_generic")
    p.add_argument("--strict", action="store_true", help="make WARN a non-zero exit")
    p.set_defaults(func=cmd_check)

    sub.add_parser("selftest", help="verify the checker against known fixtures") \
       .set_defaults(func=cmd_selftest)

    p3 = sub.add_parser("ci", help="run the fixed fixture suite and emit a PR summary")
    p3.add_argument("--out", default="", help="write a markdown CI summary here")
    p3.add_argument("--strict", action="store_true",
                    help="also fail when any fixture reports a WARN")
    p3.set_defaults(func=cmd_ci)

    p4 = sub.add_parser("verify", help="verify a shipped tar.gz + SHA256SUMS archive")
    p4.add_argument("archive", help="directory containing one .tar.gz")
    p4.set_defaults(func=cmd_verify)

    p5 = sub.add_parser("budget", help="check the anchor-memory SLO")
    p5.add_argument("results", help="directory with anchor-budget JSON results")
    p5.add_argument("--max-anchors", type=float, default=None,
                    help="SLO ceiling on surviving anchors (overrides a larger M)")
    p5.add_argument("--out", default="", help="output markdown path (json alongside it)")
    p5.add_argument("--strict", action="store_true", help="make WARN a non-zero exit")
    p5.set_defaults(func=cmd_budget)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

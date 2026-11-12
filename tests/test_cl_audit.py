"""Regression tests for the six-gate checker and its extraction contract."""
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

try:
    from ._loader import ROOT, load_script
except ImportError:  # python -m unittest discover -s tests
    from _loader import ROOT, load_script

sys.path.insert(0, str(ROOT))

from cl_audit import checks, cli, fixtures, grid, report, stats  # noqa: E402


class TestExtractionEquivalence(unittest.TestCase):
    """The extracted t-test must stay numerically identical to the bench script."""

    def test_matches_legacy_on_several_samples(self):
        legacy = load_script("bench/analyze_ladgain.py")
        cases = [([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]),
                 ([3, 5, 8, 9, 11], [1, 2, 2, 4, 5]),
                 ([70.1, 71.4, 69.8, 72.0, 70.5], [69.9, 70.2, 70.1, 71.0, 69.7]),
                 ([2, 2, 2], [1, 1, 1])]
        for a, b in cases:
            with self.subTest(a=a):
                self.assertEqual(len(legacy.t_paired(a, b)), 3)
                for x, y in zip(stats.t_paired(a, b), legacy.t_paired(a, b)):
                    self.assertAlmostEqual(x, y, places=12)


class TestGridPrimitives(unittest.TestCase):
    def test_interior_optimum_is_not_pinned(self):
        pts = [(1, 70), (3, 75), (10, 82), (30, 88), (100, 84), (300, 76)]
        r = grid.argmax_edge(pts)
        self.assertTrue(r["evaluable"])
        self.assertFalse(r["edge"])
        self.assertEqual(r["best"], 30)

    def test_ceiling_optimum_is_pinned(self):
        pts = [(1, 60), (3, 66), (10, 72), (30, 78), (100, 84), (300, 90)]
        r = grid.argmax_edge(pts)
        self.assertTrue(r["edge"])
        self.assertEqual(r["side"], "ceiling")

    def test_single_point_axis_is_not_evaluable(self):
        self.assertFalse(grid.argmax_edge([(1.0, 70.0)])["evaluable"])


class TestSixGates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _report(self, kind):
        return cli._run_dir(fixtures.make_dir(kind, root=self.tmp.name))

    def test_known_good_has_no_fail(self):
        rep = self._report("good")
        self.assertEqual(rep["summary"]["FAIL"], 0)
        gates = {r["gate"]: r["status"] for r in rep["results"]}
        self.assertEqual(gates["G1"], checks.PASS)
        self.assertEqual(gates["G2"], checks.PASS)
        self.assertEqual(gates["G3"], checks.PASS)

    def test_lambda_not_applied_is_caught_by_g1(self):
        rep = self._report("lambda_not_applied")
        g1 = next(r for r in rep["results"] if r["gate"] == "G1")
        self.assertEqual(g1["status"], checks.FAIL)
        self.assertIn("lambda", g1["detail"])

    def test_edge_pinned_is_caught_by_g2(self):
        rep = self._report("edge_pinned")
        g2 = next(r for r in rep["results"] if r["gate"] == "G2")
        self.assertEqual(g2["status"], checks.FAIL)
        self.assertGreater(g2["evidence"]["joint"]["n_lam_pinned"], 0)

    def test_adaptive_2d_slice_pinning_is_not_a_failure(self):
        # Real 8295-cell grid: 113/420 one-dimensional slices pinned, but 0/21
        # joint optima pinned. Slice pinning must NOT be reported as truncation.
        rep = self._report("adaptive_2d")
        g2 = next(r for r in rep["results"] if r["gate"] == "G2")
        self.assertGreater(g2["evidence"]["slices"]["n_pinned"], 0)
        self.assertEqual(g2["evidence"]["joint"]["n_gain_pinned"], 0)
        self.assertEqual(g2["evidence"]["joint"]["n_lam_pinned"], 0)
        self.assertEqual(g2["status"], checks.PASS)
        self.assertIn("cross-check OK", g2["detail"])

    def test_manifest_boundary_disagreement_is_caught(self):
        d = fixtures.make_dir("adaptive_2d", root=self.tmp.name)
        mp = pathlib.Path(d) / "manifest.json"
        m = json.loads(mp.read_text(encoding="utf-8"))
        m["boundary"]["gain_pinned"] = 2
        mp.write_text(json.dumps(m), encoding="utf-8")
        g2 = next(r for r in cli._run_dir(d)["results"] if r["gate"] == "G2")
        self.assertEqual(g2["status"], checks.FAIL)
        self.assertIn("disagrees", g2["detail"])

    def test_unequal_protocol_is_caught_by_g3(self):
        rep = self._report("unequal_protocol")
        g3 = next(r for r in rep["results"] if r["gate"] == "G3")
        self.assertEqual(g3["status"], checks.FAIL)

    def test_final_readings_are_classified_correctly(self):
        rep = self._report("final_readings")
        g = {r["gate"]: r for r in rep["results"]}
        self.assertEqual(rep["summary"]["FAIL"], 0)
        self.assertEqual(g["G4"]["status"], checks.WARN)
        self.assertEqual(g["G5"]["status"], checks.WARN)
        self.assertIn("anti-forgetting", g["G5"]["detail"])
        self.assertEqual(g["G6"]["status"], checks.WARN)
        self.assertIn("architectural", g["G6"]["detail"])
        # Gate 2 uses the declared boundary audit when there are no raw records.
        self.assertEqual(g["G2"]["status"], checks.PASS)

    def test_final_readings_match_the_published_report(self):
        f = fixtures.FINAL_READINGS
        self.assertEqual(f["grid_size"]["files"], 8295)
        self.assertEqual(f["boundary"], {"gain_pinned": 0, "lambda_pinned": 0, "total": 21})
        self.assertEqual(f["selfcheck"], {"pairs": 1185, "passed": 1185,
                                          "rule": "mlp_nosig == ad_nocx"})
        self.assertEqual(f["usage_weighting"][2]["dacc"], 1.42)
        self.assertEqual(f["usage_weighting"][2]["p"], 0.041)
        self.assertEqual(f["usage_weighting"][2]["dbwt"], -1.18)
        self.assertEqual([e["dacc"] for e in f["ours_vs_control"]],
                         [-0.06, 0.08, 1.33])


class TestCLI(unittest.TestCase):
    def test_selftest_passes(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["selftest"])
        self.assertEqual(rc, 0)
        self.assertIn("SELFTEST OK", buf.getvalue())

    def test_check_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = fixtures.make_dir("good", root=tmp)
            out = pathlib.Path(tmp) / "report.md"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["check", str(d), "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            self.assertTrue(out.with_suffix(".json").is_file())
            self.assertIn("| G1", out.read_text(encoding="utf-8"))
            data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(len(data["results"]), 6)

    def test_fail_gives_exit_code_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = fixtures.make_dir("edge_pinned", root=tmp)
            out = pathlib.Path(tmp) / "r.md"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cli.main(["check", str(d), "--out", str(out)])
            self.assertEqual(rc, 2)
            self.assertEqual(checks.exit_code(
                json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))), 2)

    def test_render_is_a_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = cli._run_dir(fixtures.make_dir("good", root=tmp))
        md = report.render_markdown(rep)
        self.assertIn("| Gate | Status | Finding |", md)
        self.assertIn("cl-audit", md)


if __name__ == "__main__":
    unittest.main()

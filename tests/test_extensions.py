"""Regression tests for the P2/P3 extensions: adapters, CI, archive, budget SLO.

Every new checker here also gets a deliberately broken input. A check that can
only ever report OK is not a check.
"""
import contextlib
import hashlib
import io
import json
import pathlib
import sys
import tarfile
import tempfile
import unittest

try:
    from ._loader import ROOT
except ImportError:  # python -m unittest discover -s tests
    from _loader import ROOT

sys.path.insert(0, str(ROOT))

from cl_audit import adapters, archive, budget, checks, ci, cli, fixtures  # noqa: E402


class TestMethodAdapters(unittest.TestCase):
    def test_ewc_generic_renames_foreign_columns(self):
        a = adapters.get_adapter("ewc_generic")
        r = a.normalize_record({"method": "ewc", "n_tasks": 10,
                                "ewc_lambda": 100.0, "run_id": 2,
                                "avg_acc": 81.5, "heldout_acc": 80.9,
                                "backward_transfer": -1.2})
        self.assertEqual(r["arm"], "ewc")
        self.assertEqual(r["ntask"], 10)
        self.assertEqual(r["lam"], 100.0)
        self.assertEqual(r["final_avg"], 81.5)
        self.assertEqual(r["val_final_avg"], 80.9)
        self.assertEqual(r["bwt"], -1.2)
        self.assertEqual(r["gain"], 1.0)
        self.assertNotIn("method", r)

    def test_custom_mapping_from_manifest_spec(self):
        spec = {"name": "lab_x", "field_map": {"group": "arm", "score": "final_avg",
                                               "strength": "lam"},
                "defaults": {"gain": 2.0}}
        a = adapters.get_adapter(spec)
        r = a.normalize_record({"group": "ewc", "score": 70.0, "strength": 5})
        self.assertEqual((r["arm"], r["final_avg"], r["lam"], r["gain"]),
                         ("ewc", 70.0, 5, 2.0))

    def test_unknown_adapter_fails_loudly(self):
        with self.assertRaises(KeyError):
            adapters.get_adapter("not_an_adapter")

    def test_cli_spec_overrides_manifest_spec(self):
        self.assertEqual(adapters.resolve("ewc_generic", "identity").name,
                         "identity")
        self.assertEqual(adapters.resolve("ewc_generic").name, "ewc_generic")

    def test_non_delp_fixture_evaluates_all_six_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = cli._run_dir(fixtures.make_dir("ewc_generic", root=tmp))
        self.assertEqual(len(rep["results"]), 6)
        self.assertEqual(rep["summary"][checks.SKIP], 0)
        self.assertEqual(rep["summary"][checks.FAIL], 0)
        self.assertTrue(any("adapter: ewc_generic" in n for n in rep["notes"]))
        gates = {r["gate"]: r["status"] for r in rep["results"]}
        self.assertEqual(gates["G1"], checks.PASS)
        self.assertEqual(gates["G3"], checks.PASS)
        self.assertEqual(gates["G6"], checks.PASS)

    def test_manifest_adapter_is_honoured_without_a_cli_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = pathlib.Path(fixtures.make_dir("ewc_generic", root=tmp))
            m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(m["adapter"], "ewc_generic")
            rep = cli._run_dir(d)
        self.assertEqual(rep["summary"][checks.SKIP], 0)


class TestCISuite(unittest.TestCase):
    def test_suite_is_green_and_catches_three_broken_fixtures(self):
        s = ci.run_fixture_suite(cli._run_dir)
        self.assertTrue(s["ok"], s["problems"])
        self.assertEqual((s["n_broken"], s["n_caught"]), (3, 3))
        self.assertEqual(s["final_readings"],
                         {"G4": checks.WARN, "G5": checks.WARN, "G6": checks.WARN})

    def test_a_checker_that_stops_detecting_failures_is_caught(self):
        def blind_runner(folder):
            rep = cli._run_dir(folder)
            for r in rep["results"]:
                if r["status"] == checks.FAIL:
                    r["status"] = checks.PASS
            rep["summary"][checks.FAIL] = 0
            return rep

        s = ci.run_fixture_suite(blind_runner)
        self.assertFalse(s["ok"])
        self.assertTrue(any("was not caught" in p for p in s["problems"]))

    def test_ci_markdown_lists_every_fixture(self):
        s = ci.run_fixture_suite(cli._run_dir)
        md = ci.render_ci_markdown(s)
        self.assertIn("| Fixture |", md)
        for row in s["rows"]:
            self.assertIn(row["fixture"], md)

    def test_cli_ci_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "ci.md"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cli.main(["ci", "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            self.assertIn("cl-audit CI", out.read_text(encoding="utf-8"))


class TestArchiveVerification(unittest.TestCase):
    @staticmethod
    def _mini_archive(root, corrupt_member=False, missing_entry=False):
        p = pathlib.Path(root)
        inner = p / "scratch"
        inner.mkdir()
        rec = inner / "r.json"
        payload = {"records": [{"arm": "a", "ntask": 5, "lam": 1.0,
                                "seed": 0, "final_avg": 80.0}]}
        rec.write_text(json.dumps(payload), encoding="utf-8")
        with tarfile.open(str(p / "mini.tar.gz"), "w:gz") as tf:
            tf.add(str(rec), arcname="data/r.json")
        digest = hashlib.sha256((p / "mini.tar.gz").read_bytes()).hexdigest()
        (p / "mini.tar.gz.sha256").write_text(
            "%s  mini.tar.gz\n" % digest, encoding="utf-8")
        h = hashlib.sha256(rec.read_bytes()).hexdigest()
        if corrupt_member:
            h = "0" * 64
        name = "data/missing.json" if missing_entry else "data/r.json"
        (p / "SHA256SUMS.txt").write_text("%s  %s\n" % (h, name),
                                          encoding="utf-8")
        (p / "manifest.json").write_text(
            json.dumps({"effects": [], "boundary": {}, "selfcheck": {}}),
            encoding="utf-8")
        return p

    def test_shipped_8295_cell_archive_verifies_fully(self):
        res = archive.verify_archive(ROOT / "results" / "ladgain8295")
        self.assertTrue(res["ok"], res["problems"])
        self.assertEqual(res["n_sums"], 8295)
        self.assertEqual(res["n_members"], 8295)
        self.assertEqual(res["n_verified"], 8295)

    def test_corrupted_member_hash_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._mini_archive(tmp, corrupt_member=True)
            res = archive.verify_archive(p)
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["mismatch"]), 1)

    def test_missing_file_is_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._mini_archive(tmp, missing_entry=True)
            res = archive.verify_archive(p)
        self.assertFalse(res["ok"])
        self.assertIn("data/missing.json", res["missing"])
        self.assertEqual(len(res["extra"]), 1)

    def test_cli_verify_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._mini_archive(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                ok = cli.main(["verify", str(p)])
            self.assertEqual(ok, 0)
            (p / "SHA256SUMS.txt").write_text("0" * 64 + "  data/r.json\n",
                                              encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                bad = cli.main(["verify", str(p)])
            self.assertEqual(bad, 2)


class TestBudgetSLO(unittest.TestCase):
    @staticmethod
    def _rows(fifo_anchors=3):
        rows = []
        for nt in (5, 10, 20):
            rows.append({"policy": "unbounded", "M": 999, "ntask": nt,
                         "lam": 5000, "seed": 0, "n_anchors_final": nt,
                         "final_avg": 70.0 - nt / 10, "ell": 80.0 - nt / 10,
                         "bwt": -3.0 - nt / 10})
            rows.append({"policy": "fifo", "M": 3, "ntask": nt,
                         "lam": 5000, "seed": 0,
                         "n_anchors_final": fifo_anchors,
                         "final_avg": 71.0 - nt / 10, "ell": 82.0 - nt / 10,
                         "bwt": -4.0 - nt / 10})
        return rows

    @staticmethod
    def _write(root, rows, manifest=None):
        p = pathlib.Path(root)
        (p / "runs.json").write_text(json.dumps({"results": rows}),
                                     encoding="utf-8")
        if manifest is not None:
            (p / "manifest.json").write_text(json.dumps(manifest),
                                             encoding="utf-8")
        return p

    def test_unbounded_growth_and_bounded_compliance(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = budget.check_budget(self._write(tmp, self._rows()))
        by_id = {c["id"]: c for c in rep["checks"]}
        self.assertEqual(by_id["B1"]["status"], budget.PASS)
        self.assertEqual(by_id["B2"]["status"], budget.PASS)

    def test_budget_overflow_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = budget.check_budget(self._write(tmp, self._rows(9)))
        by_id = {c["id"]: c for c in rep["checks"]}
        self.assertEqual(by_id["B1"]["status"], budget.FAIL)
        self.assertFalse(rep["ok"])
        self.assertEqual(budget.exit_code(rep), 2)

    def test_undisclosed_gain_cost_tradeoff_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = budget.check_budget(self._write(tmp, self._rows()))
        by_id = {c["id"]: c for c in rep["checks"]}
        self.assertEqual(by_id["B3"]["status"], budget.WARN)
        self.assertIn("+6.5 ell", by_id["B3"]["detail"])
        self.assertEqual(budget.exit_code(rep), 0)
        self.assertEqual(budget.exit_code(rep, strict=True), 1)

    def test_declared_tradeoff_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, self._rows(), {"tradeoff_declared": True})
            rep = budget.check_budget(p)
        by_id = {c["id"]: c for c in rep["checks"]}
        self.assertEqual(by_id["B3"]["status"], budget.PASS)

    def test_slo_ceiling_overrides_a_larger_declared_M(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = budget.check_budget(self._write(tmp, self._rows()),
                                      max_anchors=2)
        by_id = {c["id"]: c for c in rep["checks"]}
        self.assertEqual(by_id["B1"]["status"], budget.FAIL)

    def test_cli_budget_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, self._rows())
            out = pathlib.Path(tmp) / "budget.md"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cli.main(["budget", str(p), "--out", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            self.assertTrue(out.with_suffix(".json").is_file())
            self.assertIn("anchor memory SLO", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

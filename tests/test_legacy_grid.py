"""Contract tests for the D1 lambda-grid loader.

The audit's central finding was that argmaxes pinned to a grid EDGE are truncation
artefacts, not measurements. This test pins the loader contract that feeds that
check: which JSON records are admissible and how gain-calibrated runs are keyed.
"""
import json
import pathlib
import tempfile
import unittest
import warnings

# bench/grid_check.py uses json.load(open(f)) without closing the handle. That is a
# known, harmless technical debt in the ORIGINAL script; we do not edit it here so the
# repository copy stays byte-identical to the copy that produced the published numbers.
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed file")

try:
    from ._loader import load_script
except ImportError:  # python -m unittest discover -s tests
    from _loader import load_script

M = load_script("bench/grid_check.py")


def _write(folder, name, **kw):
    rec = {"arm": "ours", "ntask": 10, "lam": 100.0,
           "final_avg": 70.0, "bwt": -2.0, "hgain": 1.0}
    rec.update(kw)
    p = pathlib.Path(folder) / name
    p.write_text(json.dumps(rec), encoding="utf-8")


class TestGridLoader(unittest.TestCase):
    def setUp(self):
        # unittest resets the warning filters before each test, so re-apply here.
        warnings.simplefilter("ignore", ResourceWarning)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = self.tmp.name

    def test_valid_records_are_grouped_by_arm_task_lambda_gain(self):
        _write(self.folder, "a.json", seed=0)
        _write(self.folder, "b.json", seed=1)
        cells = M.load([str(pathlib.Path(self.folder) / "*.json")])
        self.assertEqual(list(cells), [("ours", 10, 100.0, 1.0)])
        self.assertEqual(len(cells[("ours", 10, 100.0, 1.0)]), 2)

    def test_error_records_are_excluded(self):
        _write(self.folder, "ok.json")
        _write(self.folder, "bad.json", error="cuda oom")
        cells = M.load([str(pathlib.Path(self.folder) / "*.json")])
        self.assertEqual(len(cells), 1)
        self.assertEqual(len(next(iter(cells.values()))), 1)

    def test_incomplete_records_are_excluded(self):
        (pathlib.Path(self.folder) / "partial.json").write_text(
            json.dumps({"arm": "ours", "ntask": 10, "final_avg": 70.0}),
            encoding="utf-8")
        cells = M.load([str(pathlib.Path(self.folder) / "*.json")])
        self.assertEqual(len(cells), 0)

    def test_gain_is_part_of_the_cell_key(self):
        # Averaging an alpha-calibrated run into the gain=1 curve would produce a
        # curve belonging to no model at all.
        _write(self.folder, "a.json")
        _write(self.folder, "b.json", hgain=4.0)
        cells = M.load([str(pathlib.Path(self.folder) / "*.json")])
        self.assertEqual(sorted(g for *_, g in cells), [1.0, 4.0])

    def test_missing_gain_defaults_to_one(self):
        rec = {"arm": "ours", "ntask": 10, "lam": 3.0,
               "final_avg": 70.0, "bwt": -1.0}
        (pathlib.Path(self.folder) / "n.json").write_text(json.dumps(rec),
                                                          encoding="utf-8")
        cells = M.load([str(pathlib.Path(self.folder) / "*.json")])
        self.assertIn(("ours", 10, 3.0, 1.0), cells)


if __name__ == "__main__":
    unittest.main()

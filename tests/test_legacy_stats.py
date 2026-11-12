"""Contract tests for the exact paired t-test used by the 8295-cell report.

`t_paired` returns a THREE-tuple (mean_diff, t, p). An earlier version of this
pipeline unpacked it as two values and silently corrupted downstream reporting,
so the contract is pinned here. These tests import the real bench script by path.
"""
import math
import statistics as st
import unittest

try:
    from ._loader import load_script
except ImportError:  # python -m unittest discover -s tests
    from _loader import load_script

M = load_script("bench/analyze_ladgain.py")


class TestPairedT(unittest.TestCase):
    def test_returns_three_values(self):
        out = M.t_paired([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
        self.assertEqual(len(out), 3)

    def test_known_case_matches_hand_computation(self):
        a = [1, 2, 3, 4, 5]
        b = [0, 0, 0, 0, 0]
        md, t, p = M.t_paired(a, b)
        expected_t = 3.0 / (st.stdev(a) / math.sqrt(5))
        self.assertAlmostEqual(md, 3.0, places=12)
        self.assertAlmostEqual(t, expected_t, places=10)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 0.05)

    def test_symmetry_under_sign_flip(self):
        a, b = [3, 5, 8, 9, 11], [1, 2, 2, 4, 5]
        md1, t1, p1 = M.t_paired(a, b)
        md2, t2, p2 = M.t_paired(b, a)
        self.assertAlmostEqual(md1, -md2, places=12)
        self.assertAlmostEqual(t1, -t2, places=12)
        self.assertAlmostEqual(p1, p2, places=12)

    def test_zero_variance_nonzero_effect(self):
        md, t, p = M.t_paired([2, 2, 2], [1, 1, 1])
        self.assertEqual(md, 1.0)
        self.assertEqual(t, float("inf"))
        self.assertEqual(p, 0.0)

    def test_identical_samples(self):
        md, t, p = M.t_paired([1, 1, 1], [1, 1, 1])
        self.assertEqual(md, 0.0)
        self.assertEqual(t, 0.0)
        self.assertEqual(p, 1.0)

    def test_single_pair_is_undefined(self):
        md, t, p = M.t_paired([1.0], [0.0])
        self.assertEqual(md, 1.0)
        self.assertTrue(math.isnan(t))
        self.assertTrue(math.isnan(p))

    def test_step_order_matches_the_report(self):
        # The 2026-11-11 report attributes +1.42 to ad_prot -> ad_usage at T=20.
        self.assertEqual(M.STEPS[4][:2], ("ad_prot", "ad_usage"))
        self.assertEqual(M.ORDER[-1], "ad_usage_ce")
        self.assertEqual(M.TS, (5, 10, 20))


if __name__ == "__main__":
    unittest.main()

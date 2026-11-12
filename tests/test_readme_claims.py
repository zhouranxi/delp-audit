"""The public README must not drift back to the withdrawn v1 claims.

This is the implementation-vs-declaration consistency check the whole audit is
about, applied to the repository's own prose. Without this gate, the 2026-11-11
re-analysis can be silently undone by an edit that "restores" the old abstract.
"""
import pathlib
import re
import unittest

try:
    from ._loader import ROOT
except ImportError:  # python -m unittest discover -s tests
    from _loader import ROOT

README = (ROOT / "README.md").read_text(encoding="utf-8")
FLAT = re.sub(r"\s+", " ", README)  # markdown soft-wraps lines mid-sentence


class TestReadmeClaims(unittest.TestCase):
    def test_withdrawn_claim_is_gone_as_a_positive_statement(self):
        self.assertNotIn("only one mechanism survives", README.lower())
        # The old figure is allowed to appear ONLY inside an explicit retraction.
        self.assertEqual(README.count("+1.79"), 1)
        i = FLAT.find("+1.79")
        j = FLAT.find("are withdrawn", i)
        self.assertNotEqual(j, -1, "the v1 figure must be marked as withdrawn")
        self.assertLess(j - i, 200)

    def test_final_readings_are_present(self):
        for token in ("8295", "+1.42", "0.041", "-0.06", "+0.08", "+1.33",
                      "+12.03", "+15.35", "+12.76", "0/21", "1185/1185"):
            with self.subTest(token=token):
                self.assertIn(token, README)

    def test_update_note_is_present(self):
        self.assertIn("supersedes the mechanism-ladder claim", README)
        self.assertIn("measurement and fair-tuning methodology", README)

    def test_citation_uses_the_concept_doi(self):
        cite = README.split("## Citation", 1)[1].split("## Layout", 1)[0]
        self.assertIn("https://doi.org/10.5281/zenodo.23572226", cite)
        self.assertIn("zenodo.org/badge/DOI/10.5281/zenodo.23572226.svg", cite)
        # The superseded v1 DOI must never become the badge or citation target.
        self.assertNotIn("zenodo.org/badge/DOI/10.5281/zenodo.23602281", cite)
        # The concept DOI must resolve to the corrected v3, not the withdrawn v2.
        self.assertIn("10.5281/zenodo.24148873", cite)
        self.assertIn("version 3", cite)

    def test_shipped_evidence_is_disclosed_and_really_present(self):
        self.assertIn("Status of results", README)
        self.assertIn("shipped in this repository", FLAT)
        self.assertNotIn("not shipped in this repository", FLAT)
        self.assertIn("results/ladgain8295", README)
        self.assertIn("PASS 3 / WARN 3 / FAIL 0 / SKIP 0", FLAT)
        # A prose claim without the files behind it is exactly the defect this
        # audit is about: the gate checks the evidence, not just the wording.
        d = ROOT / "results" / "ladgain8295"
        for name in ("ladgain8295.tar.gz", "ladgain8295.tar.gz.sha256",
                     "SHA256SUMS.txt", "manifest.json", "cl-audit-report.md"):
            with self.subTest(name=name):
                self.assertTrue((d / name).is_file(), name)
        sums = (d / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(sums), 8295)

    def test_zenodo_staleness_is_disclosed(self):
        # v3 is live; the older versions still carry withdrawn figures and the
        # README must keep saying so instead of pretending they were rewritten.
        self.assertIn("10.5281/zenodo.24148873", FLAT)
        self.assertIn("still quote the withdrawn figures", FLAT)
        self.assertIn("retained unchanged for the historical record", FLAT)


if __name__ == "__main__":
    unittest.main()

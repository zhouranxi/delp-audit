# bench copies — canonical index (2026-11-12)

**Canonical for released experiment scripts: this directory**, i.e.
`2026-10-17-14-29-40/github-repo-stage/bench/`, because it is the only copy under
version control (`git` history goes back to `b0a0834 Initial release`) and the one
the README and `tests/` refer to.

This is an **index, not a deletion**. No bench copy was removed, renamed or
rewritten. The other copies are left in place for provenance and are marked with
a `NON-CANONICAL-COPY.md` notice.

| Copy | py / sh | Role | Status |
|---|---|---|---|
| `github-repo-stage/bench/` | 51 / 23 | Released audit & re-analysis scripts, referenced by README and tests | **canonical** |
| `2026-10-21-02-20-31/bench/` | 46 / 15 | Working copy used for the 2026-11-11/12 D1–D2 runs and the archive pull | marked non-canonical |
| `delp-anchor-budget/bench/` | 43 / 12 | Separate S1–S3 anchor-budget line; 37 py files unique to it | marked non-canonical |
| `2026-10-17-14-29-40/bench/` | 1 / 0 | Top-level leftover (`s3_analyze.py`) | marked non-canonical |
| Server `/mnt/mydisk/xinguan-dendritic/bench/` | 84 / — | Largest run-time copy on the shared 4×3090 host | read-only, not marked (no writes to the shared server) |

Difference counts against the canonical copy (by filename):
`2026-10-21` copy has 2 unique py files and lacks 7; `delp-anchor-budget` has 37
unique and lacks 45. These are genuinely different generations/lines, not
identical files that could be safely deduplicated.

## The two archive-provenance scripts

`2026-10-21-02-20-31/bench/pull_ladgain.py` and `build_ladgain_manifest.py` are
the only py files unique to the working copy. They are **deliberately not copied
here**: both resolve paths relative to their own parent workspace
(`<root>/archive/ladgain8295/` and a sibling `analyze_ladgain.py`), so a literal
copy would be a script that looks runnable but writes to the wrong place.

Their output *is* merged: see
[`results/ladgain8295/`](../results/ladgain8295/README.md) for the published
tarball, SHA256 manifest, recomputed conclusion-level `manifest.json` and the
six-gate report. Re-running the pull is unnecessary now that the archive ships
with the repository; `cl-audit verify results/ladgain8295` reproduces the
integrity check offline.

## Rule going forward

- New **tooling** goes to `cl_audit/` (installable, tested, adapter-based), never
  into a second `bench/` copy.
- `bench/` is frozen historical experiment code. Fixing a released bug belongs
  here, with a test in `tests/`.
- Do not delete old copies; keep them readable and pointed at this index.

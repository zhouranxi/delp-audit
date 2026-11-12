# 8295-cell joint gain × λ result set

Checksummed, full **per-seed** archive behind the 2026-11-11 re-analysis described in
the [top-level README](../../README.md). This is the data on which `cl-audit` was
validated; it is not a synthetic fixture.

- Source: `/mnt/mydisk/xinguan-dendritic/dendritic-lab/results_ladgain` (lab server),
  retrieved read-only on 2026-11-12 by streaming `tar czf - | base64` (the server
  wrote nothing and started no jobs).
- Size: 8295 JSON records, 33 MB uncompressed, 2.6 MB in the tar.gz. 1659 grid cells
  × 5 seeds, 7 arms × {5, 10, 20} tasks, gain ∈ [0.25, 1024], λ ∈ [0.1, 1e5].
- The 8.6 GB figure quoted in earlier inventories refers to **all** experiment results
  on that server, not to this grid.

## Contents

| File | What it is |
|---|---|
| `ladgain8295.tar.gz` | All 8295 per-seed JSON records (directory `results_ladgain/`) |
| `ladgain8295.tar.gz.sha256` | SHA256 of the tarball |
| `SHA256SUMS.txt` | SHA256 of each of the 8295 JSON files inside the tarball |
| `manifest.json` | Conclusion-level declarations + effect table **recomputed from the per-seed records** (paired t-tests, joint-optimum boundary, self-check counts). Not a transcription of the report. |
| `cl-audit-report.md` / `.json` | The six-gate report produced from this archive |

## Verify

```bash
cl-audit verify results/ladgain8295
# or, without installing the CLI:
python -m cl_audit.cli verify results/ladgain8295
```

`verify` checks the tarball hash, parses every line of `SHA256SUMS.txt`, hashes each
tar member **in memory** (nothing is written to disk), and reports any mismatch or
missing/extra file. Exit code 2 on any failure; CI runs this step.

## Reproduce the six gates

```bash
cd results/ladgain8295
tar xzf ladgain8295.tar.gz                 # -> results_ladgain/*.json
cp manifest.json results_ladgain/manifest.json
cl-audit check results_ladgain --out my-report.md
```

Expected: **PASS 3 / WARN 3 / FAIL 0 / SKIP 0**, exit code 0. G1 (declaration–
implementation), G2 (tuning fairness, 0/21 joint optima grid-pinned, cross-checked
against the manifest) and G3 (baseline protocol parity) pass. G4 warns that only
1/3 task horizons are significant, G5 warns that the significant effect worsens BWT
(−1.18, p = 0.033) and therefore **cannot be described as anti-forgetting**, and G6
warns that the largest effect (+15.35 points) comes from removing the hidden sigmoid
— an architectural change, not a protection mechanism.

## Honesty notes

- `cl_audit/fixtures.py` also contains synthetic directories and a conclusion-level
  `FINAL_READINGS` constant. Those exist **only** to regression-test the checker's
  logic and must never be cited as experimental results. This directory is the real
  data.
- The abstract/description of Zenodo v1 and v2 still state the withdrawn figures
  (`+1.79, p = 0.005`, "only one mechanism survives"). They predate this re-analysis;
  see the update note at the top of the top-level README until the preprint is revised.

# Audit of Dendritic Context-Gated Protection (DeLP)

Code, preregistrations, and raw results for the paper *"Where Do the Gains Come From? An
Audit of Dendritic Context-Gated Protection for Continual Learning"*.

The paper audits our earlier claim that a usage-localized elastic penalty on dendritic
segments (DeLP) dominates standard continual-learning baselines on Permuted-MNIST. After
correcting baseline defects, calibration confounds, and protocol errors, DeLP no longer
leads on forgetting, and at ten tasks corrected EWC and MAS exceed its accuracy. Three
preregistered rule-switching experiments show no adaptation advantage of compartmentalized
routing at any cue-conflict intensity.

> **Update (2026-11-11) — supersedes the mechanism-ladder claim in version 1 of the preprint.**
> The original seven-rung ladder tuned lambda only, with the hidden-layer gain fixed at 1;
> 19 of 21 selected gains later hit the grid boundary. Re-running every rung on a joint
> gain x lambda grid (8295 cells: 21 (rung, T) combinations x 1659 (gain, lambda) cells x
> 5 seeds) changed the accounting:
>
> - Usage weighting is **+1.42 accuracy points (p = 0.041) at T = 20 only**, accompanied by
>   **dBWT -1.18 (p = 0.033)** -- forgetting gets *worse*. It cannot be described as an
>   anti-forgetting mechanism.
> - The full method against a dendrite-free control is **-0.06 / +0.08 / +1.33** points at
>   T = 5 / 10 / 20, **none statistically significant**.
> - The only large remaining effect is removing the hidden sigmoid
>   (**+12.03 / +15.35 / +12.76** points), an activation/gain effect, not a protection
>   mechanism.
> - Boundary audit after widening both grids: **0/21** selected gains and **0/21** selected
>   lambdas sit on a grid edge; the implementation-equivalence self-check
>   (`mlp_nosig` == `ad_nocx`) passes **1185/1185**.
>
> The contribution is therefore a **measurement and fair-tuning methodology**, not a
> surviving protection mechanism. The v1 abstract's claim of a single surviving
> mechanism (usage weighting), and its "+1.79, p = 0.005 / +4 to +6 once gain-calibrated"
> figures, are withdrawn. The 8295-cell per-seed result set, its SHA256 manifest and
> the six-gate report are shipped in `results/ladgain8295/` (see Status of results).

## Citation

If you use this code or refer to the audit, please cite the **concept DOI**, which always
resolves to the latest version of the preprint:

> Zhou, R. (2026). *Where Do the Gains Come From? An Audit of Dendritic Context-Gated
> Protection for Continual Learning*. Zenodo. <https://doi.org/10.5281/zenodo.23572226>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23572226.svg)](https://doi.org/10.5281/zenodo.23572226)

DOI note: `10.5281/zenodo.23572226` is the **concept DOI** covering all versions; it
currently resolves to **version 3** (`10.5281/zenodo.24148873`, 2026-11-12), whose
description carries the 2026-11-11 correction. The earlier version records
(`10.5281/zenodo.23572227`, `10.5281/zenodo.23602281`, `10.5281/zenodo.23685788`)
are retained unchanged for the historical record; their abstracts still quote the
withdrawn figures and must not be cited as the current conclusion. The v3 files
(PDF and LaTeX) are inherited unchanged from version 2 — only the description was
corrected. Earlier drafts of this README used the superseded v1 DOI; this section and
the repository history now use the concept DOI consistently.

## Layout

| Directory | Contents | Paper section |
|---|---|---|
| `dendritic-lab/` | Original experiment scripts (exp1–exp12), including the flawed baseline runs that triggered the audit | §III (audit findings), §IV |
| `bench/` | Audit & re-analysis: corrected baselines (`exp6_fixed.py`), seven-rung mechanism ladder (`ladder.py`; the original 945-cell λ-only grid is **superseded** by the 8295-cell joint gain × λ grid described above), gain/α sweeps, gate probes, calibration analyses; second-benchmark replication on Permuted-Fashion-MNIST (`exp6_fixed_fashion.py`, `results_fashion/`) | §IV–§V |
| `dendrite-mve/` | Preregistered rule-switching study: environments, models, training/stats scripts, preregistrations, raw per-seed results | §VI |
| `cl_audit/` | `cl-audit` CLI: the six-gate pre-publication check (implementation/declaration consistency, tuning fairness, mechanism attribution), a `MethodAdapter` layer for non-DeLP methods, CI fixture suite, archive verification and anchor-budget SLO checks | tooling |
| `tests/` | Offline regression tests, including equivalence tests against `bench/analyze_ladgain.py` and deliberately broken fixtures | tooling |
| `results/ladgain8295/` | **Shipped evidence**: checksummed tar.gz of all 8295 per-seed JSON records, per-file SHA256SUMS, recomputed conclusion-level `manifest.json`, and the six-gate report produced from it | §IV (2026-11-11 update) |

## Data provenance for the paper tables

| Paper item | Source of the quoted numbers |
|---|---|
| Table I `orig.` rows (DELP / MAS / PackNet) | `dendritic-lab/exp12_result.json` (archived original table) |
| Table I corrected rows (EWC, SI, MAS corrected, PackNet corrected) | `dendritic-lab/results_ei_main.json` (all 16 table cells: 8 arms × T ∈ {5,10} × 5 seeds, per-seed records with accuracy matrices); the λ sweep used to select each λ is in `dendritic-lab/results_ei_sweep.json` |
| Table II (Permuted-Fashion-MNIST) | `bench/results_fashion/main_*.json` (5-seed main stage; `sweep_*.json` shards are the 2-seed λ screens and are excluded from the table) |
| Table III mechanism ladder | `bench/reports/ladder.md` + `bench/reports/ladder.csv` (980 rows, the complete released batch; `bench/out/ladder.csv` is an earlier partial batch kept for the record) |
| Gain/α sweeps (§V) | `bench/reports/sig_probe.md` |
| Table V preregistered runs | `dendrite-mve/prereg/*.md` result records + raw per-seed JSONs in `dendrite-mve/results*/` |

Note on the two ladder CSVs: `bench/reports/ladder.csv` (980 rows) is the batch behind
Table III; `bench/out/ladder.csv` (945 rows) is an earlier partial run whose T=20 cells
differ (elasticity up to +9.11 there vs +7.94 in the released batch). All paper numbers
follow `bench/reports/`.

Note on λ selection in Table I: the three re-implemented arms (plain EWC, corrected SI,
corrected MAS) select λ on the held-out validation split, because their code records
`val_bwt`/`val_final_avg`. The arms that reuse the original implementation (DeLP itself,
the original SI/MAS/PackNet) keep their original λ, because those implementations do not
record held-out metrics. `audit/verify_lamsel.py` reproduces this check from the released
sweep records.

Note on the 2026-11-11 correction: the provenance table above documents the data behind
the **paper tables as published in versions 1-2 of the preprint**. It remains valid as a
record of where those numbers came from. The *mechanism conclusion* drawn from the
λ-only ladder is superseded by the joint gain × λ re-analysis in the update block at the
top of this README; its 8295 per-seed records, checksums and six-gate report are shipped
in `results/ladgain8295/` (see Status of results). Neither batch of original files was
rewritten or deleted.

## Second benchmark: Permuted-Fashion-MNIST (`bench/`)

The corrected eight-arm comparison was replicated on Permuted-Fashion-MNIST with an identical
protocol (same arms, λ grids, validation-selected λ per arm, 5 seeds, T ∈ {5, 10}):
`exp6_fixed_fashion.py` (dataset adapter over `exp6_fixed.py`), `run_fashion_all.sh`
(sweep → aggregate → main pipeline), `aggregate_fashion.py` (val λ selection),
`analyze_fashion.py` (table + paired tests). Raw shards in `bench/results_fashion/`.
Fashion-MNIST raw IDX files are not committed; download the four `.gz` files from the
[Zalando Research repository](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)
into `fashion_data/`.

Note: `exp6_fixed.py` as initially released contained a defect in the plain-EWC arm
(per-sample label indexing produced a non-scalar loss on newer PyTorch), fixed by casting
the label index to `int`; all other arms were unaffected.

## Preregistered rule-switching experiments (`dendrite-mve/`)

All three experiments (4 models × 8 seeds, 1500 episodes each) are fully preregistered;
hypotheses, thresholds, and validity gates were fixed before running, and result records
were appended without altering the preregistered text.

| Directory | Experiment | Cue conflict | Outcome |
|---|---|---|---|
| `results_full/` | PR-001 | none (no cues) | H1 not supported (7.1% reduction, p=0.335) |
| `results_cue_rerun/`, `results_cue/` | PR-002 | 10% misleading cues | H2 not supported (7.1%, p=0.296) |
| `results_cue_adv/` | PR-003 | 100% adversarial cues | H3 not supported (0.0%, p=0.458) |
| `results_cue_invalid_20261017_local/` | PR-002 first run | — | voided by preregistered gate (logging defect); kept for the record |

Preregistration documents with amendments and result records: `dendrite-mve/prereg/`.
Power analysis: `dendrite-mve/power_analysis.py` (MDE at 80% power = 30% adaptation
reduction for n=8; the PR-003 floor effect raises it to 50%).

## Requirements

The `cl-audit` tooling is **standard library only**, Python 3.9+:

```bash
pip install -e .
cl-audit --help
python -m unittest discover -s tests -v   # offline regression suite
cl-audit selftest                        # known-good / deliberately broken fixtures
cl-audit ci --out ci-summary.md          # CI gate + PR-ready summary
cl-audit verify results/ladgain8295      # verify every shipped SHA256 in memory
```

`cl-audit check <dir> --adapter ewc_generic` shows the non-DeLP path: adapters in
`cl_audit/adapters.py` rename foreign column names into the checker's internal
schema; the gates themselves contain no DeLP-specific field names. `cl-audit
budget <dir>` is a separate anchor-memory SLO check and does not alter the six
publication gates.

Re-running the original GPU experiments additionally needs `torch>=2.0`, `numpy`,
`scipy` (see `dendrite-mve/requirements.txt`, currently unpinned); all runs fit on a
single RTX 3090. Those experiment scripts are standalone and are not installed as a
package by `pip install -e .`.

## Reproducing

```bash
# original experiments (pre-audit)
cd dendritic-lab && python exp9_consistent_table.py

# audit: corrected baselines (8 arms x {5,10} tasks x 5 seeds)
cd bench && python exp6_fixed.py

# audit: seven-rung mechanism ladder, LEGACY lambda-only protocol (945 cells)
cd bench && python ladder.py

# audit: 2026-11-11 protocol, joint gain x lambda grid (8295 cells).
# Jobs are generated by gen_ladgain2.py, dispatched with run_lanes.sh, and
# aggregated with analyze_ladgain.py:
cd bench && python analyze_ladgain.py <results_dir> > ladgain2.md

# preregistered rule-switching runs (single GPU, ~55 min per run)
cd dendrite-mve && python train2.py --model dc_net_cue --seed 0 --device cuda
cd dendrite-mve && python stats2.py   # or stats3.py for PR-003

# six-gate pre-publication check on any result directory
cl-audit check <results_dir> --out report.md
cl-audit selftest

# reproduce the six gates from the SHIPPED 8295-cell archive
cd results/ladgain8295
cl-audit verify .
tar xzf ladgain8295.tar.gz
cp manifest.json results_ladgain/manifest.json
cl-audit check results_ladgain --out my-report.md   # PASS 3 / WARN 3 / FAIL 0 / SKIP 0
```

## Status of results (2026-11-12)

The 8295-cell joint gain × λ result set behind the update above **is shipped in this
repository**, in `results/ladgain8295/`: a 2.6 MB tar.gz of all 8295 per-seed JSON
records, its SHA256, a per-file `SHA256SUMS.txt` (8295 entries), a conclusion-level
`manifest.json` whose effect table was **recomputed from those records** (not copied
from the report), and the six-gate report. `cl-audit verify results/ladgain8295`
checks every hash in memory; CI runs that step. The data were pulled read-only from
the lab server on 2026-11-12.

On that archive the six gates report **PASS 3 / WARN 3 / FAIL 0 / SKIP 0**: declaration
consistency, tuning fairness and protocol parity pass, while the significance,
forgetting-direction and mechanism-attribution gates warn that the one significant
effect is not stable across task horizons, worsens BWT, and is dominated by an
architectural activation change. Synthetic directories in `cl_audit/fixtures.py`
(including a plain-EWC format fixture) exist only to regression-test the checker
and must never be cited as experimental results.

Version 3 of the Zenodo record (under the concept DOI above) carries the 2026-11-11
correction in its description; its files are inherited unchanged from version 2. The
abstracts of the earlier versions (v0.2, v1, v2) predate the re-analysis and still
quote the withdrawn figures; they are retained for the historical record and should be
read together with the v3 correction and the update note at the top of this README.

## License

MIT — see [LICENSE](LICENSE).

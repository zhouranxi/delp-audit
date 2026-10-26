# Audit of Dendritic Context-Gated Protection (DeLP)

Code, preregistrations, and raw results for the paper *"Where Do the Gains Come From? An
Audit of Dendritic Context-Gated Protection for Continual Learning"*.

The paper audits our earlier claim that a usage-localized elastic penalty on dendritic
segments (DeLP) dominates standard continual-learning baselines on Permuted-MNIST. After
correcting baseline defects, calibration confounds, and protocol errors, only one mechanism
survives (usage weighting), and three preregistered rule-switching experiments show no
adaptation advantage of compartmentalized routing at any cue-conflict intensity.

## Citation

If you use this code or refer to the audit, please cite the preprint:

> Zhou, R. (2026). *Where Do the Gains Come From? An Audit of Dendritic Context-Gated
> Protection for Continual Learning*. Zenodo. <https://doi.org/10.5281/zenodo.23572226>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23572226.svg)](https://doi.org/10.5281/zenodo.23572226)

## Layout

| Directory | Contents | Paper section |
|---|---|---|
| `dendritic-lab/` | Original experiment scripts (exp1–exp12), including the flawed baseline runs that triggered the audit | §III (audit findings), §IV |
| `bench/` | Audit & re-analysis: corrected baselines (`exp6_fixed.py`), seven-rung mechanism ladder (`ladder.py`), gain/α sweeps, gate probes, calibration analyses; second-benchmark replication on Permuted-Fashion-MNIST (`exp6_fixed_fashion.py`, `results_fashion/`) | §IV–§V |
| `dendrite-mve/` | Preregistered rule-switching study: environments, models, training/stats scripts, preregistrations, raw per-seed results | §VI |

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

Python 3.10+, `torch>=2.0`, `numpy`, `scipy` (see `dendrite-mve/requirements.txt`).
All runs fit on a single RTX 3090.

## Reproducing

```bash
# original experiments (pre-audit)
cd dendritic-lab && python exp9_consistent_table.py

# audit: corrected baselines (8 arms x {5,10} tasks x 5 seeds)
cd bench && python exp6_fixed.py

# audit: seven-rung mechanism ladder (9 lambda x 5 seeds x {5,10,20} tasks)
cd bench && python ladder.py

# preregistered rule-switching runs (single GPU, ~55 min per run)
cd dendrite-mve && python train2.py --model dc_net_cue --seed 0 --device cuda
# statistics / gates / hypothesis tests
cd dendrite-mve && python stats2.py   # or stats3.py for PR-003
```

## License

MIT — see [LICENSE](LICENSE).

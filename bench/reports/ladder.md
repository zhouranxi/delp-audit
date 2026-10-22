# Ladder ablation: where does the gain actually come from?

Each rung adds exactly one mechanism to the one above it. Data,
permutations, seeds, optimiser, epochs, batch size, kWTA and the
evaluation protocol are identical across rungs; only the mechanism
differs. Every cell is 5 seeds.

## 1. Is the lambda rule well posed?

| T | rung | lam* (val acc) | test acc | test BWT | lam (val BWT) | acc at that lam | elasticity |
|---|---|---|---|---|---|---|---|
| 5 | R0  PlainMLP + EWC (h = kwta(sigmoid(t))) | 3000 | 76.24 | -4.45 | 100000 | 68.91 | +2.72 |
| 5 | R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | 300 | 81.61 | -6.07 | 100000 | 68.64 | +3.70 |
| 5 | R1  + dendrite params, context pinned to 0 | 300 | 81.61 | -6.07 | 100000 | 68.64 | +3.70 |
| 5 | R2  + context modulation | 1000 | 82.22 | -2.97 | 100000 | 70.84 | +3.41 |
| 5 | R3  + branch protection | 300 | 82.11 | -5.71 | 100000 | 70.64 | +3.38 |
| 5 | R4  + usage weighting (= ours) | 1000 | 81.98 | -4.50 | 100000 | 72.59 | +3.43 |
| 5 | R5  R4 with a CE Fisher | 300 | 82.02 | -5.61 | 100000 | 71.54 | +3.19 |
| 10 | R0  PlainMLP + EWC (h = kwta(sigmoid(t))) | 300 | 69.94 | -9.52 | 100000 | 44.69 | +2.55 |
| 10 | R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | 30 | 74.43 | -13.19 | 100000 | 41.50 | +4.22 |
| 10 | R1  + dendrite params, context pinned to 0 | 30 | 74.43 | -13.19 | 100000 | 41.50 | +4.22 |
| 10 | R2  + context modulation | 30 | 74.80 | -13.15 | 100000 | 43.80 | +4.19 |
| 10 | R3  + branch protection | 30 | 74.85 | -13.11 | 100000 | 42.31 | +4.10 |
| 10 | R4  + usage weighting (= ours) | 30 | 74.93 | -13.36 | 100000 | 43.96 | +3.86 |
| 10 | R5  R4 with a CE Fisher | 10 | 74.97 | -13.51 | 100000 | 43.49 | +3.91 |
| 20 | R0  PlainMLP + EWC (h = kwta(sigmoid(t))) | 30 | 59.46 | -20.95 | 100000 | 28.09 | +4.91 |
| 20 | R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | 3 | 61.59 | -26.27 | 100000 | 26.20 | +7.94 |
| 20 | R1  + dendrite params, context pinned to 0 | 3 | 61.59 | -26.27 | 100000 | 26.20 | +7.94 |
| 20 | R2  + context modulation | 10 | 62.94 | -24.71 | 100000 | 27.96 | +7.66 |
| 20 | R3  + branch protection | 3 | 62.57 | -25.56 | 100000 | 26.66 | +7.89 |
| 20 | R4  + usage weighting (= ours) | 10 | 63.60 | -24.28 | 100000 | 27.38 | +7.31 |
| 20 | R5  R4 with a CE Fisher | 3 | 62.84 | -25.38 | 100000 | 27.07 | +7.83 |

If every `lam (val BWT)` lands on the top of the grid while its accuracy is far below the peak, a BWT-only selection rule is degenerate (BWT falls monotonically in lambda, accuracy is hump-shaped). The adopted rule is therefore **held-out average accuracy**, and the paper must state it.

## 2. What does each mechanism buy?

### T = 5

#### 2a. each rung at its own best lambda (tuned on validation)

| rung | lam* | final acc | BWT | n |
|---|---|---|---|---|
| R0  PlainMLP + EWC (h = kwta(sigmoid(t))) | 3000 | 76.24 ± 0.39 | -4.45 ± 0.46 | 5 |
| R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | 300 | 81.61 ± 0.62 | -6.07 ± 0.84 | 5 |
| R1  + dendrite params, context pinned to 0 | 300 | 81.61 ± 0.62 | -6.07 ± 0.84 | 5 |
| R2  + context modulation | 1000 | 82.22 ± 0.72 | -2.97 ± 0.51 | 5 |
| R3  + branch protection | 300 | 82.11 ± 0.71 | -5.71 ± 0.84 | 5 |
| R4  + usage weighting (= ours) | 1000 | 81.98 ± 0.76 | -4.50 ± 0.90 | 5 |
| R5  R4 with a CE Fisher | 300 | 82.02 ± 1.06 | -5.61 ± 1.24 | 5 |

#### 2b. stepwise deltas, paired on seeds

| step | dAcc | p | dBWT | p | reading |
|---|---|---|---|---|---|
| mlp_ewc -> mlp_nosig | +5.37 | 0.000 | -1.62 | 0.003 | the hidden ACTIVATION (drop the sigmoid), not the dendrites |
| mlp_nosig -> ad_nocx | +0.00 | 1.000 | +0.00 | 1.000 | dendrite capacity alone (null control, expect 0) |
| ad_nocx -> ad_ctx | +0.61 | 0.172 | +3.10 | 0.001 | context modulation |
| ad_ctx -> ad_prot | -0.11 | 0.616 | -2.74 | 0.000 | branch protection |
| ad_prot -> ad_usage | -0.13 | 0.563 | +1.21 | 0.015 | usage weighting |
| ad_usage -> ad_usage_ce | +0.04 | 0.842 | -1.10 | 0.008 | CE Fisher instead of the max-logit proxy |

#### 2c. same steps at a COMMON lambda = 10 (no per-rung tuning)

| step | dAcc | p | dBWT | p |
|---|---|---|---|---|
| mlp_ewc -> mlp_nosig | +1.90 | 0.025 | -2.58 | 0.011 |
| mlp_nosig -> ad_nocx | +0.00 | 1.000 | +0.00 | 1.000 |
| ad_nocx -> ad_ctx | +0.87 | 0.033 | +0.59 | 0.184 |
| ad_ctx -> ad_prot | +0.41 | 0.003 | +0.55 | 0.001 |
| ad_prot -> ad_usage | +0.04 | 0.851 | -0.18 | 0.455 |
| ad_usage -> ad_usage_ce | +0.10 | 0.534 | +0.27 | 0.163 |

#### 2d. each rung against the two baselines at its own lambda*

`vs R0` is the claim as the paper states it. `vs R0b` is the same claim once the baseline is given the activation the method uses; the R4 row is the number that survives.

| rung | dAcc vs R0 | p | dAcc vs R0b | p | dBWT vs R0b | p |
|---|---|---|---|---|---|---|
| R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | +5.37 | 0.000 | +0.00 | 1.000 | +0.00 | 1.000 |
| R1  + dendrite params, context pinned to 0 | +5.37 | 0.000 | +0.00 | 1.000 | +0.00 | 1.000 |
| R2  + context modulation | +5.97 | 0.000 | +0.61 | 0.172 | +3.10 | 0.001 |
| R3  + branch protection | +5.86 | 0.000 | +0.49 | 0.221 | +0.35 | 0.393 |
| R4  + usage weighting (= ours) | +5.74 | 0.000 | +0.37 | 0.494 | +1.56 | 0.056 |
| R5  R4 with a CE Fisher | +5.78 | 0.000 | +0.41 | 0.493 | +0.46 | 0.517 |

### T = 10

#### 2a. each rung at its own best lambda (tuned on validation)

| rung | lam* | final acc | BWT | n |
|---|---|---|---|---|
| R0  PlainMLP + EWC (h = kwta(sigmoid(t))) | 300 | 69.94 ± 0.75 | -9.52 ± 1.02 | 5 |
| R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | 30 | 74.43 ± 0.54 | -13.19 ± 0.56 | 5 |
| R1  + dendrite params, context pinned to 0 | 30 | 74.43 ± 0.54 | -13.19 ± 0.56 | 5 |
| R2  + context modulation | 30 | 74.80 ± 0.40 | -13.15 ± 0.41 | 5 |
| R3  + branch protection | 30 | 74.85 ± 0.91 | -13.11 ± 0.92 | 5 |
| R4  + usage weighting (= ours) | 30 | 74.93 ± 0.81 | -13.36 ± 0.94 | 5 |
| R5  R4 with a CE Fisher | 10 | 74.97 ± 0.38 | -13.51 ± 0.51 | 5 |

#### 2b. stepwise deltas, paired on seeds

| step | dAcc | p | dBWT | p | reading |
|---|---|---|---|---|---|
| mlp_ewc -> mlp_nosig | +4.49 | 0.000 | -3.67 | 0.001 | the hidden ACTIVATION (drop the sigmoid), not the dendrites |
| mlp_nosig -> ad_nocx | +0.00 | 1.000 | +0.00 | 1.000 | dendrite capacity alone (null control, expect 0) |
| ad_nocx -> ad_ctx | +0.37 | 0.019 | +0.04 | 0.701 | context modulation |
| ad_ctx -> ad_prot | +0.05 | 0.873 | +0.04 | 0.887 | branch protection |
| ad_prot -> ad_usage | +0.07 | 0.820 | -0.25 | 0.502 | usage weighting |
| ad_usage -> ad_usage_ce | +0.04 | 0.863 | -0.15 | 0.548 | CE Fisher instead of the max-logit proxy |

#### 2c. same steps at a COMMON lambda = 10 (no per-rung tuning)

| step | dAcc | p | dBWT | p |
|---|---|---|---|---|
| mlp_ewc -> mlp_nosig | +3.74 | 0.003 | -0.65 | 0.277 |
| mlp_nosig -> ad_nocx | +0.00 | 1.000 | +0.00 | 1.000 |
| ad_nocx -> ad_ctx | +0.62 | 0.018 | +0.37 | 0.146 |
| ad_ctx -> ad_prot | +0.32 | 0.029 | +0.30 | 0.088 |
| ad_prot -> ad_usage | -0.27 | 0.272 | -0.51 | 0.097 |
| ad_usage -> ad_usage_ce | +0.71 | 0.019 | +1.05 | 0.012 |

#### 2d. each rung against the two baselines at its own lambda*

`vs R0` is the claim as the paper states it. `vs R0b` is the same claim once the baseline is given the activation the method uses; the R4 row is the number that survives.

| rung | dAcc vs R0 | p | dAcc vs R0b | p | dBWT vs R0b | p |
|---|---|---|---|---|---|---|
| R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | +4.49 | 0.000 | +0.00 | 1.000 | +0.00 | 1.000 |
| R1  + dendrite params, context pinned to 0 | +4.49 | 0.000 | +0.00 | 1.000 | +0.00 | 1.000 |
| R2  + context modulation | +4.86 | 0.000 | +0.37 | 0.019 | +0.04 | 0.701 |
| R3  + branch protection | +4.91 | 0.000 | +0.42 | 0.203 | +0.08 | 0.798 |
| R4  + usage weighting (= ours) | +4.98 | 0.000 | +0.49 | 0.083 | -0.17 | 0.508 |
| R5  R4 with a CE Fisher | +5.02 | 0.000 | +0.54 | 0.018 | -0.32 | 0.070 |

### T = 20

#### 2a. each rung at its own best lambda (tuned on validation)

| rung | lam* | final acc | BWT | n |
|---|---|---|---|---|
| R0  PlainMLP + EWC (h = kwta(sigmoid(t))) | 30 | 59.46 ± 0.94 | -20.95 ± 0.95 | 5 |
| R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | 3 | 61.59 ± 0.62 | -26.27 ± 0.57 | 5 |
| R1  + dendrite params, context pinned to 0 | 3 | 61.59 ± 0.62 | -26.27 ± 0.57 | 5 |
| R2  + context modulation | 10 | 62.94 ± 2.00 | -24.71 ± 2.10 | 5 |
| R3  + branch protection | 3 | 62.57 ± 0.75 | -25.56 ± 0.75 | 5 |
| R4  + usage weighting (= ours) | 10 | 63.60 ± 1.12 | -24.28 ± 1.19 | 5 |
| R5  R4 with a CE Fisher | 3 | 62.84 ± 2.04 | -25.38 ± 2.21 | 5 |

#### 2b. stepwise deltas, paired on seeds

| step | dAcc | p | dBWT | p | reading |
|---|---|---|---|---|---|
| mlp_ewc -> mlp_nosig | +2.13 | 0.012 | -5.32 | 0.000 | the hidden ACTIVATION (drop the sigmoid), not the dendrites |
| mlp_nosig -> ad_nocx | +0.00 | 1.000 | +0.00 | 1.000 | dendrite capacity alone (null control, expect 0) |
| ad_nocx -> ad_ctx | +1.35 | 0.260 | +1.56 | 0.214 | context modulation |
| ad_ctx -> ad_prot | -0.37 | 0.602 | -0.85 | 0.284 | branch protection |
| ad_prot -> ad_usage | +1.02 | 0.008 | +1.28 | 0.006 | usage weighting |
| ad_usage -> ad_usage_ce | -0.76 | 0.404 | -1.10 | 0.279 | CE Fisher instead of the max-logit proxy |

#### 2c. same steps at a COMMON lambda = 3 (no per-rung tuning)

| step | dAcc | p | dBWT | p |
|---|---|---|---|---|
| mlp_ewc -> mlp_nosig | +3.06 | 0.018 | +0.08 | 0.909 |
| mlp_nosig -> ad_nocx | +0.00 | 1.000 | +0.00 | 1.000 |
| ad_nocx -> ad_ctx | +1.30 | 0.242 | +0.88 | 0.416 |
| ad_ctx -> ad_prot | -0.32 | 0.672 | -0.18 | 0.822 |
| ad_prot -> ad_usage | +0.52 | 0.167 | +0.14 | 0.693 |
| ad_usage -> ad_usage_ce | -0.26 | 0.584 | +0.04 | 0.939 |

#### 2d. each rung against the two baselines at its own lambda*

`vs R0` is the claim as the paper states it. `vs R0b` is the same claim once the baseline is given the activation the method uses; the R4 row is the number that survives.

| rung | dAcc vs R0 | p | dAcc vs R0b | p | dBWT vs R0b | p |
|---|---|---|---|---|---|---|
| R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | +2.13 | 0.012 | +0.00 | 1.000 | +0.00 | 1.000 |
| R1  + dendrite params, context pinned to 0 | +2.13 | 0.012 | +0.00 | 1.000 | +0.00 | 1.000 |
| R2  + context modulation | +3.48 | 0.028 | +1.35 | 0.260 | +1.56 | 0.214 |
| R3  + branch protection | +3.12 | 0.001 | +0.98 | 0.112 | +0.71 | 0.199 |
| R4  + usage weighting (= ours) | +4.14 | 0.001 | +2.00 | 0.025 | +1.99 | 0.024 |
| R5  R4 with a CE Fisher | +3.38 | 0.013 | +1.25 | 0.328 | +0.89 | 0.491 |

## 3. Best accuracy achievable under a forgetting budget

For each rung: the highest final accuracy among all lambda settings whose |BWT| stays within the budget. A rung wins a column only if it can be tuned to that forgetting level at all.

| T | rung | |BWT|<= 2 | |BWT|<= 4 | |BWT|<= 6 | |BWT|<= 10 | |BWT|<= 20 |
|---|---|---|---|---|---|---|
| 5 | R0  PlainMLP + EWC (h = kwta(sigmoid(t))) | **72.6** (λ=30000, BWT -1.5) | **75.3** (λ=10000, BWT -3.0) | **76.2** (λ=3000, BWT -4.4) | **76.2** (λ=3000, BWT -4.4) | **76.2** (λ=3000, BWT -4.4) |
| 5 | R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | **79.5** (λ=3000, BWT -1.7) | **81.4** (λ=1000, BWT -3.4) | **81.4** (λ=1000, BWT -3.4) | **81.6** (λ=300, BWT -6.1) | **81.6** (λ=300, BWT -6.1) |
| 5 | R1  + dendrite params, context pinned to 0 | **79.5** (λ=3000, BWT -1.7) | **81.4** (λ=1000, BWT -3.4) | **81.4** (λ=1000, BWT -3.4) | **81.6** (λ=300, BWT -6.1) | **81.6** (λ=300, BWT -6.1) |
| 5 | R2  + context modulation | **80.3** (λ=3000, BWT -1.5) | **82.2** (λ=1000, BWT -3.0) | **82.2** (λ=1000, BWT -3.0) | **82.2** (λ=1000, BWT -3.0) | **82.2** (λ=1000, BWT -3.0) |
| 5 | R3  + branch protection | **80.3** (λ=3000, BWT -1.5) | **82.1** (λ=1000, BWT -3.1) | **82.1** (λ=300, BWT -5.7) | **82.1** (λ=300, BWT -5.7) | **82.1** (λ=300, BWT -5.7) |
| 5 | R4  + usage weighting (= ours) | **78.6** (λ=10000, BWT -1.4) | **81.3** (λ=3000, BWT -2.6) | **82.0** (λ=1000, BWT -4.5) | **82.0** (λ=1000, BWT -4.5) | **82.0** (λ=1000, BWT -4.5) |
| 5 | R5  R4 with a CE Fisher | **80.2** (λ=3000, BWT -1.9) | **81.8** (λ=1000, BWT -3.5) | **82.0** (λ=300, BWT -5.6) | **82.0** (λ=300, BWT -5.6) | **82.0** (λ=300, BWT -5.6) |
| 10 | R0  PlainMLP + EWC (h = kwta(sigmoid(t))) | **44.7** (λ=100000, BWT -1.5) | **44.7** (λ=100000, BWT -1.5) | **50.1** (λ=30000, BWT -4.2) | **69.9** (λ=300, BWT -9.5) | **69.9** (λ=300, BWT -9.5) |
| 10 | R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | **47.8** (λ=10000, BWT -1.9) | **47.8** (λ=10000, BWT -1.9) | **54.6** (λ=3000, BWT -4.9) | **60.9** (λ=1000, BWT -10.0) | **74.4** (λ=30, BWT -13.2) |
| 10 | R1  + dendrite params, context pinned to 0 | **47.8** (λ=10000, BWT -1.9) | **47.8** (λ=10000, BWT -1.9) | **54.6** (λ=3000, BWT -4.9) | **60.9** (λ=1000, BWT -10.0) | **74.4** (λ=30, BWT -13.2) |
| 10 | R2  + context modulation | **49.8** (λ=10000, BWT -1.7) | **49.8** (λ=10000, BWT -1.7) | **56.4** (λ=3000, BWT -4.1) | **63.4** (λ=1000, BWT -8.2) | **74.8** (λ=30, BWT -13.2) |
| 10 | R3  + branch protection | **49.1** (λ=10000, BWT -1.7) | **49.1** (λ=10000, BWT -1.7) | **55.2** (λ=3000, BWT -4.6) | **61.8** (λ=1000, BWT -9.2) | **74.9** (λ=30, BWT -13.1) |
| 10 | R4  + usage weighting (= ours) | **47.5** (λ=30000, BWT -1.6) | **52.2** (λ=10000, BWT -3.1) | **52.2** (λ=10000, BWT -3.1) | **57.7** (λ=3000, BWT -7.5) | **74.9** (λ=30, BWT -13.4) |
| 10 | R5  R4 with a CE Fisher | **46.1** (λ=30000, BWT -1.3) | **50.4** (λ=10000, BWT -2.4) | **55.8** (λ=3000, BWT -5.8) | **61.8** (λ=1000, BWT -9.9) | **75.0** (λ=10, BWT -13.5) |
| 20 | R0  PlainMLP + EWC (h = kwta(sigmoid(t))) | **28.1** (λ=100000, BWT -1.1) | **28.1** (λ=100000, BWT -1.1) | **30.4** (λ=30000, BWT -4.2) | **30.4** (λ=30000, BWT -4.2) | **33.0** (λ=10000, BWT -10.9) |
| 20 | R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | **29.5** (λ=10000, BWT -1.1) | **33.1** (λ=3000, BWT -3.8) | **33.1** (λ=3000, BWT -3.8) | **36.9** (λ=1000, BWT -9.5) | **36.9** (λ=1000, BWT -9.5) |
| 20 | R1  + dendrite params, context pinned to 0 | **29.5** (λ=10000, BWT -1.1) | **33.1** (λ=3000, BWT -3.8) | **33.1** (λ=3000, BWT -3.8) | **36.9** (λ=1000, BWT -9.5) | **36.9** (λ=1000, BWT -9.5) |
| 20 | R2  + context modulation | **31.1** (λ=10000, BWT -1.0) | **34.8** (λ=3000, BWT -3.2) | **34.8** (λ=3000, BWT -3.2) | **38.2** (λ=1000, BWT -8.8) | **38.2** (λ=1000, BWT -8.8) |
| 20 | R3  + branch protection | **29.8** (λ=10000, BWT -1.3) | **33.7** (λ=3000, BWT -3.5) | **33.7** (λ=3000, BWT -3.5) | **37.3** (λ=1000, BWT -9.3) | **37.3** (λ=1000, BWT -9.3) |
| 20 | R4  + usage weighting (= ours) | **28.9** (λ=30000, BWT -1.1) | **31.7** (λ=10000, BWT -2.4) | **31.7** (λ=10000, BWT -2.4) | **35.0** (λ=3000, BWT -6.7) | **38.6** (λ=1000, BWT -13.3) |
| 20 | R5  R4 with a CE Fisher | **30.8** (λ=10000, BWT -1.6) | **30.8** (λ=10000, BWT -1.6) | **33.6** (λ=3000, BWT -4.3) | **33.6** (λ=3000, BWT -4.3) | **36.9** (λ=1000, BWT -10.2) |

Grid-floor check at T = 20. `floor` is accuracy at lambda = 3, the added bottom point; `peak` is the best accuracy anywhere on the grid.

| rung | floor | peak | lam at peak | floor - peak |
|---|---|---|---|---|
| R0  PlainMLP + EWC (h = kwta(sigmoid(t))) | 58.53 | 59.46 | 30 | -0.93 |
| R0b hidden activation: kwta(0.5*t) instead of kwta(sigmoid(t)) | 61.59 | 61.59 | 3 | +0.00 |
| R1  + dendrite params, context pinned to 0 | 61.59 | 61.59 | 3 | +0.00 |
| R2  + context modulation | 62.89 | 62.94 | 10 | -0.05 |
| R3  + branch protection | 62.57 | 62.57 | 3 | +0.00 |
| R4  + usage weighting (= ours) | 63.10 | 63.60 | 10 | -0.50 |
| R5  R4 with a CE Fisher | 62.84 | 62.84 | 3 | +0.00 |

A rung whose `lam at peak` is the floor had its step sizes measured at a truncated grid; its deltas in section 2 therefore sit at the weaker end of what it can do. R0 and R4 peak inside the grid (lambda = 30 and 10), so the headline comparisons are unaffected; the branch-protection and CE-Fisher steps at T = 20 are the ones that move, and they are reported at their weaker strength.


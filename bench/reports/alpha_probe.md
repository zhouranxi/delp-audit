# alpha-calibrated DeLP: is the family position a scale artefact?

`alpha` is the hidden-layer gain (`--hgain`); alpha=1 is the published configuration. Each cell is 5 seeds unless stated.

## 1. Best lambda per gain, and where each gain lands on the axis

| T | alpha | best lambda | acc | BWT | log10 A | R | R^2 of slopes |
|---|---|---|---|---|---|---|---|
| 5 | 1 | 1000 | 81.48 | -5.03 | 5.317 | 0.11 | 0.954 |
| 5 | 2 | 1000 | 84.83 | -3.13 | 5.431 | 0.17 | 0.986 |
| 5 | 4 | 1000 | 87.81 | -1.80 | 5.508 | 0.30 | 0.945 |
| 10 | 1 | 100 | 74.43 | -12.49 | 4.713 | 1.22 | 0.975 |
| 10 | 2 | 100 | 78.16 | -10.43 | 4.817 | 1.11 | 0.992 |
| 10 | 4 | 100 | 83.00 | -7.72 | 4.828 | 1.06 | 0.910 |
| 20 | 1 | 30 | 65.50 | -20.99 | 4.527 | 1.11 | 0.980 |
| 20 | 2 | 30 | 69.11 | -19.85 | 4.550 | 1.46 | 0.986 |
| 20 | 4 | 100 | 75.18 | -12.08 | 5.411 | 1.99 | 0.992 |

## 2. Direct effect of the gain, at matched lambda (paired by seed)

Same lambda, same seeds, only alpha differs. This isolates the re-parameterisation effect from any lambda re-placement.

| T | lambda | alpha | d(acc) vs alpha=1 | p | d(BWT) | d(log10 A) |
|---|---|---|---|---|---|---|
| 5 | 30 | 2 | +4.41 | 0.0000 | +3.59 | +0.122 |
| 5 | 100 | 2 | +4.49 | 0.0000 | +3.69 | +0.118 |
| 5 | 1000 | 2 | +3.35 | 0.0000 | +1.91 | +0.115 |
| 5 | 5000 | 2 | +3.09 | 0.0000 | +0.34 | +0.121 |
| 5 | 30 | 4 | +6.65 | 0.0000 | +4.54 | +0.217 |
| 5 | 100 | 4 | +7.23 | 0.0000 | +5.16 | +0.197 |
| 5 | 1000 | 4 | +6.33 | 0.0000 | +3.23 | +0.192 |
| 5 | 5000 | 4 | +6.29 | 0.0000 | +0.63 | +0.186 |
| 10 | 30 | 2 | +3.50 | 0.0000 | +1.74 | +0.092 |
| 10 | 100 | 2 | +3.73 | 0.0000 | +2.06 | +0.104 |
| 10 | 1000 | 2 | +5.14 | 0.0000 | +1.21 | +0.143 |
| 10 | 5000 | 2 | +6.42 | 0.0000 | -0.29 | +0.204 |
| 10 | 30 | 4 | +7.15 | 0.0000 | +3.62 | +0.130 |
| 10 | 100 | 4 | +8.57 | 0.0000 | +4.77 | +0.115 |
| 10 | 1000 | 4 | +11.37 | 0.0000 | +2.04 | +0.161 |
| 10 | 5000 | 4 | +14.36 | 0.0000 | -1.28 | +0.312 |
| 20 | 30 | 2 | +3.62 | 0.0000 | +1.14 | +0.022 |
| 20 | 100 | 2 | +4.24 | 0.0000 | +0.72 | +0.085 |
| 20 | 1000 | 2 | +6.05 | 0.0000 | -0.04 | +0.191 |
| 20 | 5000 | 2 | +4.93 | 0.0000 | -0.70 | +0.241 |
| 20 | 30 | 4 | +8.85 | 0.0000 | +3.97 | -0.011 |
| 20 | 100 | 4 | +11.26 | 0.0000 | +3.60 | +0.032 |
| 20 | 1000 | 4 | +13.22 | 0.0000 | -2.77 | +0.288 |
| 20 | 5000 | 4 | +12.25 | 0.0000 | -2.16 | +0.436 |

## 3. Verdict for contribution C4

The reading that decides C4: if a gain-calibrated DeLP still sits among the expensive families, the family grouping is a mechanism result and can be published. If it moves to the cheap cluster, the grouping was tracking the hidden-layer scale and C4 must be withdrawn.

- T=5: alpha=1 best acc 81.48. alpha=2: acc 84.83 (+3.35), BWT -3.13 (+1.91); alpha=4: acc 87.81 (+6.33), BWT -1.80 (+3.23)
- T=10: alpha=1 best acc 74.43. alpha=2: acc 78.16 (+3.73), BWT -10.43 (+2.06); alpha=4: acc 83.00 (+8.57), BWT -7.72 (+4.77)
- T=20: alpha=1 best acc 65.50. alpha=2: acc 69.11 (+3.62), BWT -19.85 (+1.14); alpha=4: acc 75.18 (+9.68), BWT -12.08 (+8.91)

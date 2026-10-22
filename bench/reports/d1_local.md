# Exchange rate as a chord over lambda* -> 4 x lambda*

Both endpoints are measured. The derivative at lambda* is NOT used: lambda* is the accuracy maximum, so that derivative is zero by construction and the ratio is degenerate (the earlier attempt produced R^2 ~ 0.4 and sign flips, which is the signature of sitting on the singularity, not of noise).

| family | T | lambda* | lambda* x4 | acc: lo -> hi | BWT: lo -> hi | d(acc) | d(BWT) | R |
|---|---|---|---|---|---|---|---|---|
| delp | 10 | 100 | 400 | 74.43 -> 72.75 | -12.49 -> -8.26 | -1.68 | +4.23 | 0.40 |
| ewc_multi | 10 | 10 | 40 | 69.95 -> 69.52 | -13.90 -> -12.02 | -0.44 | +1.88 | 0.23 |
| ewc_online | 5 | 5 | 20 | 75.96 -> 75.53 | -10.84 -> -10.19 | -0.43 | +0.65 | 0.67 |
| ewc_online | 10 | 50 | 200 | 70.11 -> 69.43 | -12.44 -> -10.45 | -0.69 | +1.99 | 0.35 |
| ewc_online | 20 | 300 | 1000 | 61.37 -> 54.54 | -15.04 -> -13.21 | -6.84 | +1.83 | 3.73 |
| mas | 5 | 0.5 | 2 | 77.91 -> 77.35 | -6.81 -> -3.29 | -0.56 | +3.52 | 0.16 |
| mas | 10 | 0.3 | 1.2 | 74.33 -> 68.61 | -8.94 -> -5.72 | -5.73 | +3.22 | 1.78 |
| mas | 20 | 0.1 | 0.4 | 66.35 -> 58.39 | -19.23 -> -17.03 | -7.95 | +2.20 | 3.62 |
| si | 5 | 20 | 100 | 77.82 -> 74.65 | -6.50 -> -1.41 | -3.17 | +5.09 | 0.62 |
| si | 10 | 10 | 40 | 73.02 -> 65.35 | -10.08 -> -6.15 | -7.66 | +3.94 | 1.95 |
| si | 20 | 3 | 10 | 64.68 -> 57.21 | -21.82 -> -20.58 | -7.47 | +1.24 | 6.03 |

## Is R stable across T?

| family | R (T=5 / 10 / 20) | spread | verdict |
|---|---|---|---|
| delp | - / 0.40 / - | 1.0x | stable |
| ewc_multi | - / 0.23 / - | 1.0x | stable |
| ewc_online | 0.67 / 0.35 / 3.73 | 10.8x | unstable |
| mas | 0.16 / 1.78 / 3.62 | 22.8x | unstable |
| si | 0.62 / 1.95 / 6.03 | 9.7x | unstable |

## Ordering at each T

**T = 5** (cheapest first): mas 0.16, si 0.62, ewc_online 0.67

**T = 10** (cheapest first): ewc_multi 0.23, ewc_online 0.35, delp 0.40, mas 1.78, si 1.95

**T = 20** (cheapest first): mas 3.62, ewc_online 3.73, si 6.03

# Is the hidden-scale knob specific to DeLP?

Every number is at that family's own accuracy-optimal lambda, so only the gain varies.

| family | T | gain=1 | gain=4 | gain=16 | best gain | delta at best |
|---|---|---|---|---|---|---|
| ewc_online | 5 | 75.76 | 79.57 | 76.41 | 4 | +3.81 |
| ewc_online | 10 | 70.01 | 71.77 | 66.24 | 4 | +1.76 |
| ewc_online | 20 | 60.59 | 60.61 | 60.51 | 4 | +0.02 |
| ewc_multi | 5 | 75.46 | 78.97 | 75.53 | 4 | +3.51 |
| ewc_multi | 10 | 69.95 | 70.62 | 66.01 | 4 | +0.67 |
| ewc_multi | 20 | 59.48 | 61.53 | 62.50 | 16 | +3.02 |
| si | 5 | 77.49 | 80.84 | 78.03 | 4 | +3.34 |
| si | 10 | 73.02 | 78.99 | 73.38 | 4 | +5.98 |
| si | 20 | 60.16 | 63.26 | 61.43 | 4 | +3.10 |
| mas | 5 | 77.76 | 81.80 | 78.10 | 4 | +4.05 |
| mas | 10 | 74.33 | 77.70 | 71.32 | 4 | +3.37 |
| mas | 20 | 66.35 | 70.92 | nan | 4 | +4.57 |

## DeLP reference (its own best lambda at each gain)

| T | alpha=1 | alpha=2 | alpha=4 |
|---|---|---|---|
| 5 | 81.48 | 84.83 | 87.81 |
| 10 | 74.43 | 78.16 | 83.00 |
| 20 | 65.50 | 69.11 | 75.18 |

## Verdict

| family | gain from its own knob (T=5 / 10 / 20) |
|---|---|
| **delp (alpha)** | +6.33 / +8.57 / +9.68 |
| ewc_online (hsig) | +3.81 / +1.76 / +0.02 |
| ewc_multi (hsig) | +3.51 / +0.67 / +3.02 |
| si (hsig) | +3.34 / +5.98 / +3.10 |
| mas (hsig) | +4.05 / +3.37 / +4.57 |

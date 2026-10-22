# D1 exchange-rate table (lambda grids verified interior)

A = lambda x protected mass. R = |d(acc)/dlog10A| / (d(BWT)/dlog10A), bootstrap over seeds (2000 draws). R is points-of-accuracy per point-of-forgetting.

## T = 5

| family | lambdas | argmax | edge | acc@argmax | BWT@argmax | d(acc)/dlogA | d(BWT)/dlogA | R whole-grid [95% CI] | R descending [95% CI] | R^2 |
|---|---|---|---|---|---|---|---|---|---|---|
| delp | 3..20000 (7) | 1000 | no | 81.48 | -5.03 | +0.37 | +3.48 | 0.11 [0.03, 0.17] | 0.96 [0.57, 1.49] | 0.954 |
| ewc_online | 0.1..10000 (6) | 10 | no | 75.76 | -10.76 | -0.18 | +1.63 | 0.11 [0.01, 0.38] | 0.13 [0.01, 0.42] | 0.906 |
| ewc_multi | 10..10000 (4) | 1000 | no | 75.46 | -6.23 | +0.11 | +2.34 | 0.05 [0.00, 0.20] | - | 0.971 |
| si | 0.1..100 (4) | 10 | no | 77.49 | -8.30 | -0.04 | +3.69 | 0.01 [0.00, 0.23] | - | 0.851 |
| mas | 0.1..100 (4) | 1 | no | 77.76 | -5.38 | -5.41 | +3.73 | 1.45 [1.08, 1.98] | 3.08 [2.23, 4.43] | 0.883 |

## T = 10

| family | lambdas | argmax | edge | acc@argmax | BWT@argmax | d(acc)/dlogA | d(BWT)/dlogA | R whole-grid [95% CI] | R descending [95% CI] | R^2 |
|---|---|---|---|---|---|---|---|---|---|---|
| delp | 3..20000 (7) | 100 | no | 74.43 | -12.49 | -4.82 | +3.96 | 1.22 [1.13, 1.31] | 1.95 [1.75, 2.19] | 0.975 |
| ewc_online | 10..10000 (4) | 100 | no | 70.01 | -11.16 | -3.08 | +3.50 | 0.88 [0.67, 1.14] | 2.01 [1.51, 2.76] | 0.978 |
| ewc_multi | 1..10000 (6) | 10 | no | 69.95 | -13.90 | -2.39 | +2.08 | 1.15 [0.89, 1.45] | 2.88 [2.15, 3.84] | 0.908 |
| si | 0.1..100 (4) | 10 | no | 73.02 | -10.08 | -2.54 | +5.93 | 0.43 [0.31, 0.57] | - | 0.950 |
| mas | 0.01..100 (7) | 0.3 | no | 74.33 | -8.94 | -9.28 | +5.16 | 1.80 [1.70, 1.91] | 4.49 [4.01, 5.14] | 0.927 |

## T = 20

| family | lambdas | argmax | edge | acc@argmax | BWT@argmax | d(acc)/dlogA | d(BWT)/dlogA | R whole-grid [95% CI] | R descending [95% CI] | R^2 |
|---|---|---|---|---|---|---|---|---|---|---|
| delp | 3..20000 (7) | 30 | no | 65.50 | -20.99 | -7.50 | +6.78 | 1.11 [1.05, 1.16] | 1.64 [1.59, 1.70] | 0.980 |
| ewc_online | 10..10000 (4) | 100 | no | 60.59 | -18.97 | -5.85 | +6.07 | 0.96 [0.86, 1.09] | 2.05 [1.85, 2.35] | 0.990 |
| ewc_multi | 1..10000 (6) | 10 | no | 59.48 | -22.94 | -5.51 | +2.58 | 2.14 [1.80, 2.50] | 3.53 [2.83, 4.29] | 0.413 |
| si | 0.1..100 (4) | 1 | no | 60.16 | -27.88 | -5.41 | +9.39 | 0.58 [0.44, 0.72] | 0.93 [0.81, 1.04] | 0.922 |
| mas | 0.01..100 (7) | 0.1 | no | 66.35 | -19.23 | -11.91 | +8.25 | 1.44 [1.35, 1.54] | 2.21 [2.08, 2.36] | 0.947 |

## Summary: does R separate into clusters, and does it move with T?

Whole-grid R first (the unstable definition), then the descending-limb R (lambda >= lambda*), which is the defensible one.

| family | whole-grid R (T=5 / 10 / 20) | spread | descending-limb R (T=5 / 10 / 20) | spread |
|---|---|---|---|---|
| delp | 0.11 / 1.22 / 1.11 | 11.3x | 0.96 / 1.95 / 1.64 | 2.0x |
| ewc_online | 0.11 / 0.88 / 0.96 | 8.5x | 0.13 / 2.01 / 2.05 | 16.3x |
| ewc_multi | 0.05 / 1.15 / 2.14 | 47.3x | nan / 2.88 / 3.53 | 1.2x |
| si | 0.01 / 0.43 / 0.58 | 57.1x | nan / nan / 0.93 | 1.0x |
| mas | 1.45 / 1.80 / 1.44 | 1.2x | 3.08 / 4.49 / 2.21 | 2.0x |

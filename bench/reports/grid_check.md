# lambda-grid adequacy for the D1 families

For each (family, T): the argmax of held-out accuracy, whether that argmax sits on a grid edge, and the accuracy gap to the adjacent interior point. `edge` means the reported optimum is a truncation artefact, not a measurement.

## T = 5

| family | grid | argmax | edge? | acc@argmax | BWT@argmax | gap to neighbour |
|---|---|---|---|---|---|---|
| ad_only | 0 (single) | - | - | 76.56 | -15.13 | - |
| delp | 3..20000 | lam=2000 | no | 81.49 | -3.26 | - |
| ewc_multi | 10..10000 | lam=4000 | no | 76.21 | -4.32 | - |
| ewc_online | 0.1..10000 | lam=5 | no | 75.96 | -10.84 | - |
| lwf | 0.01..100 | lam=0.1 | no | 76.71 | -11.38 | - |
| mas | 0.1..100 | lam=0.5 | no | 77.91 | -6.81 | - |
| naive | 0 (single) | - | - | 75.30 | -12.26 | - |
| packnet | 0 (single) | - | - | 51.53 | -1.31 | - |
| si | 0.1..100 | lam=20 | no | 77.82 | -6.50 | - |

## T = 10

| family | grid | argmax | edge? | acc@argmax | BWT@argmax | gap to neighbour |
|---|---|---|---|---|---|---|
| ad_only | 0 (single) | - | - | 71.40 | -18.69 | - |
| delp | 3..20000 | lam=100 | no | 74.43 | -12.49 | - |
| ewc_multi | 1..10000 | lam=10 | no | 69.95 | -13.90 | - |
| ewc_online | 10..10000 | lam=50 | no | 70.11 | -12.44 | - |
| lwf | 0.01..100 | lam=0.1 | no | 69.55 | -19.01 | - |
| mas | 0.01..100 | lam=0.3 | no | 74.33 | -8.94 | - |
| naive | 0 (single) | - | - | 68.70 | -19.07 | - |
| packnet | 0 (single) | - | - | 31.22 | -0.51 | - |
| si | 0.1..100 | lam=10 | no | 73.02 | -10.08 | - |

## T = 20

| family | grid | argmax | edge? | acc@argmax | BWT@argmax | gap to neighbour |
|---|---|---|---|---|---|---|
| ad_only | 0 (single) | - | - | 58.30 | -31.60 | - |
| delp | 3..20000 | lam=60 | no | 65.92 | -17.27 | - |
| ewc_multi | 1..10000 | lam=40 | no | 59.66 | -20.33 | - |
| ewc_online | 10..10000 | lam=300 | no | 61.37 | -15.04 | - |
| lwf | 0.1..100 | lam=1 | no | 58.42 | -28.66 | - |
| mas | 0.01..100 | lam=0.1 | no | 66.35 | -19.23 | - |
| naive | 0 (single) | - | - | 55.95 | -32.49 | - |
| packnet | 0 (single) | - | - | 20.78 | -0.24 | - |
| si | 0.1..100 | lam=3 | no | 64.68 | -21.82 | - |

## Verdict

Every family's argmax is interior at every T. The grids bracket their optima and the reported numbers are measurements, not truncations.

## Common-lambda comparability

The lambdas that all non-trivial families share, and how many families each covers:

- lambda=0.1: 4 families (ewc_online, lwf, mas, si)
- lambda=1: 5 families (ewc_multi, ewc_online, lwf, mas, si)
- lambda=2.5: 3 families (ewc_multi, ewc_online, si)
- lambda=3: 3 families (delp, ewc_multi, si)
- lambda=5: 3 families (ewc_multi, ewc_online, si)
- lambda=10: 6 families (delp, ewc_multi, ewc_online, lwf, mas, si)
- lambda=20: 3 families (ewc_multi, ewc_online, si)
- lambda=40: 3 families (ewc_multi, ewc_online, si)
- lambda=100: 6 families (delp, ewc_multi, ewc_online, lwf, mas, si)
- lambda=1000: 3 families (delp, ewc_multi, ewc_online)

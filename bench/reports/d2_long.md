# D2: does the dual compartment win when gap >> tau_s?

Accuracy / context probe. Probe = same input, different context. `n` is seeds.

| gap | point neuron | point + gated ctx | point + slow state | ALIF (slow threshold) | dual compartment (add) | dual compartment (mul) | dual compartment (thr) |
|---|---|---|---|---|---|---|---|
| 2 | - | - | 100.0 / 100 (n=3) | 100.0 / 34 (n=3) | 100.0 / 100 (n=3) | 100.0 / 100 (n=3) | 99.9 / 100 (n=3) |
| 5 | - | - | 100.0 / 100 (n=3) | 100.0 / 37 (n=3) | 100.0 / 100 (n=3) | 100.0 / 100 (n=3) | 99.9 / 100 (n=3) |
| 10 | - | - | 100.0 / 100 (n=3) | 100.0 / 30 (n=3) | 100.0 / 100 (n=3) | 100.0 / 100 (n=3) | 100.0 / 100 (n=3) |
| 20 | - | - | 100.0 / 97 (n=3) | 99.6 / 33 (n=3) | 100.0 / 97 (n=3) | 99.9 / 97 (n=3) | 99.9 / 97 (n=3) |
| 40 | - | - | 99.9 / 77 (n=3) | 94.7 / 33 (n=3) | 99.8 / 76 (n=3) | 99.3 / 79 (n=3) | 99.3 / 79 (n=3) |
| 80 | - | - | 99.5 / 42 (n=3) | 81.5 / 32 (n=3) | 98.9 / 43 (n=3) | 94.3 / 43 (n=3) | 96.1 / 43 (n=3) |
| 160 | - | - | 98.2 / 36 (n=3) | 62.8 / 38 (n=3) | 92.9 / 38 (n=3) | - | 84.0 / 34 (n=3) |
| 320 | - | - | 97.1 / 28 (n=3) | 53.4 / 35 (n=3) | 76.7 / 30 (n=3) | - | 66.5 / 28 (n=3) |

## The decisive comparison: dual(add) vs point_slow

| gap | point_slow | dual(add) | dual - point_slow | p (Welch) |
|---|---|---|---|---|
| 2 | 100.00 | 100.00 | +0.00 | 1.000 |
| 5 | 100.00 | 100.00 | +0.00 | 1.000 |
| 10 | 100.00 | 100.00 | +0.00 | 1.000 |
| 20 | 100.00 | 100.00 | +0.00 | 1.000 |
| 40 | 99.93 | 99.83 | -0.10 | 0.034 |
| 80 | 99.50 | 98.90 | -0.60 | 0.000 |
| 160 | 98.23 | 92.93 | -5.30 | 0.000 |
| 320 | 97.13 | 76.70 | -20.43 | 0.000 |

## Does the compartment ever beat ALIF?

ALIF also carries a slow state, so a compartment claim needs to beat it too, not just the plain point neuron.

| gap | dual(add) | alif | dual - alif | p (Welch) |
|---|---|---|---|---|
| 2 | 100.00 | 100.00 | +0.00 | 1.000 |
| 5 | 100.00 | 100.00 | +0.00 | 1.000 |
| 10 | 100.00 | 99.97 | +0.03 | 0.317 |
| 20 | 100.00 | 99.63 | +0.37 | 0.000 |
| 40 | 99.83 | 94.67 | +5.17 | 0.000 |
| 80 | 98.90 | 81.47 | +17.43 | 0.000 |
| 160 | 92.93 | 62.83 | +30.10 | 0.000 |
| 320 | 76.70 | 53.43 | +23.27 | 0.000 |

## Verdict against the pre-registered falsification condition

- gap=160: dual - point_slow = -5.30 (p=0.000); beats alif: True
- gap=320: dual - point_slow = -20.43 (p=0.000); beats alif: True

**Condition does not fire.** The compartment never recovers a >=5 point advantage over a single membrane carrying the same slow state, at any gap tested up to the longest. The narrowed claim in decision 2 stands, and the delay range tested must be stated in the paper.

## Decay from gap 2 to the longest gap

| arm | acc@gap2 | acc@320 | loss |
|---|---|---|---|
| point + slow state | 100.0 | 97.1 | -2.9 |
| ALIF (slow threshold) | 100.0 | 53.4 | -46.6 |
| dual compartment (add) | 100.0 | 76.7 | -23.3 |
| dual compartment (thr) | 99.9 | 66.5 | -33.4 |

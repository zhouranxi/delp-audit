# D2 with the readout leak removed

`whole` = trial-averaged readout (the historical number). `last15` = only the final 15 steps. `probe` = context decodable from the slow state at the end of the gap. chance = 33.3 (3 contexts x 3 templates).

| arm | gap | whole | last15 | leak | probe |
|---|---|---|---|---|---|
| alif | 2 | - | 93.6 | - | 34 |
| alif | 10 | - | 35.3 | - | 32 |
| alif | 40 | - | 34.1 | - | 32 |
| alif | 80 | 80.9 | 30.7 | +50.2 | 32 |
| alif | 160 | - | 35.8 | - | 36 |
| alif | 320 | 47.7 | 31.3 | +16.4 | 34 |
| dual | 2 | - | 99.9 | - | 100 |
| dual | 10 | - | 99.5 | - | 100 |
| dual | 40 | - | 81.1 | - | 76 |
| dual | 80 | 98.9 | 30.6 | +68.3 | 41 |
| dual | 160 | - | 35.9 | - | 36 |
| dual | 320 | 70.0 | 31.3 | +38.7 | 28 |
| point | 2 | - | 93.8 | - | 34 |
| point | 10 | - | 40.7 | - | 28 |
| point | 40 | - | 34.2 | - | 32 |
| point | 80 | 83.8 | 31.2 | +52.6 | 30 |
| point | 160 | - | 35.9 | - | 34 |
| point | 320 | 48.2 | 31.4 | +16.8 | 34 |
| point_slow | 2 | - | 100.0 | - | 100 |
| point_slow | 10 | - | 99.9 | - | 100 |
| point_slow | 40 | - | 81.8 | - | 74 |
| point_slow | 80 | 99.6 | 43.5 | +56.1 | 42 |
| point_slow | 160 | - | 33.6 | - | 34 |
| point_slow | 320 | 96.6 | 29.3 | +67.3 | 28 |

## Restricted-readout retention curve

This is the only column that can support a memory claim.

| arm | gap 2 | gap 10 | gap 40 | gap 80 | gap 160 | gap 320 |
|---|---|---|---|---|---|---|
| alif | 93.6 | 35.3 * | 34.1 * | 30.7 * | 35.8 * | 31.3 * |
| dual | 99.9 | 99.5 | 81.1 | 30.6 * | 35.9 * | 31.3 * |
| point | 93.8 | 40.7 | 34.2 * | 31.2 * | 35.9 * | 31.4 * |
| point_slow | 100.0 | 99.9 | 81.8 | 43.5 | 33.6 * | 29.3 * |

`*` marks values within ~7 points of chance, i.e. no evidence of retention.

## Largest gap at which each arm still beats chance

| arm | largest gap with acc >= 45 | restricted acc there |
|---|---|---|
| alif | 2 | 93.6 |
| dual | 40 | 81.1 |
| point | 2 | 93.8 |
| point_slow | 40 | 81.8 |

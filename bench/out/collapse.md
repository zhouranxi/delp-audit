# C1': universal shape + family offset

For each family f and task count T we fit  BWT = a_f + b_f*log10(A)  and
ell = c_f + d_f*log10(A).  If the slopes agree across families, the frontier
has a family-independent SHAPE and families differ only by an offset
(a horizontal shift in log A = the unit mismatch of their Omega).


## T = 5

| family | n | BWT slope | BWT R^2 | BWT a | ell slope | ell R^2 | ell a |
|---|---|---|---|---|---|---|---|
| ewc_multi | 4 | +2.40 | 0.969 | -17.0 | -1.83 | 0.973 | +88.4 |
| ewc_online | 4 | +2.29 | 0.939 | -16.9 | -2.12 | 0.986 | +89.6 |

  BWT slope spread across families: +2.29 .. +2.40 (sd 0.05) -> shape looks family-independent

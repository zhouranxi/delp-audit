"""Exact paired t-test, extracted unchanged from bench/analyze_ladgain.py.

Extraction contract: `t_paired` returns a THREE-tuple (mean_diff, t, p). An
earlier version of the pipeline unpacked it as two values and silently corrupted
downstream reporting, so tests/test_legacy_stats.py and
tests/test_cl_audit.py both pin this contract and assert numerical equivalence
with the original bench script. Do not "simplify" the return type.

The incomplete-beta implementation avoids a scipy dependency so the audit can
run on a laptop with only the per-seed JSON files.
"""
import math
import statistics as st


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb + a * math.log(x) + b * math.log1p(-x)) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_paired(a, b):
    """Two-sided p for a paired t-test. Returns (mean_diff, t, p)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return (st.mean(d) if d else float("nan"), float("nan"), float("nan"))
    md = st.mean(d)
    sd = st.stdev(d)
    if sd == 0:
        return md, float("inf") if md else 0.0, 0.0 if md else 1.0
    t = md / (sd / math.sqrt(n))
    return md, t, _betai(0.5 * (n - 1), 0.5, (n - 1) / ((n - 1) + t * t))


def sem(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def mean(v):
    return st.mean(v) if v else float("nan")

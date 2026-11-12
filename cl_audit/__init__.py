"""cl-audit: six-gate pre-publication audit for continual-learning claims.

The DeLP audit found that most apparent gains came from (a) an undeclared gain
scalar, (b) hyperparameter grids that did not bracket the optimum, and (c) a
baseline implementation defect. This package turns those findings into a
repeatable, offline check that any result directory must pass before its numbers
are reported.

Standard library only: no torch, no GPU, no lab-server data required.
"""
__version__ = "0.2.0"

from .checks import check_experiment, FAIL, PASS, SKIP, WARN  # noqa: F401,E402

"""The statistics, and the choices behind them.

Two numbers in this file are opinions rather than formulas, and both are stated
in the README because an unlabelled percentile or an undefined "consistency" is
not comparable with anybody else's.
"""

from __future__ import annotations

import math

__all__ = ["consistency", "mean", "percentile"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, inclusive. Always an observed value.

    Linear interpolation — the default in numpy and most stats packages —
    produces a number **between** two observations, which is fine for describing
    a distribution and wrong for a latency budget. p95 answers "how slow is the
    slow case", and a latency that no request ever exhibited is not an answer to
    that question. On twenty examples it is also nearly meaningless: the
    interpolated p95 sits between the 19th and 20th observations and moves with
    the weighting rather than with the data.

    So: sort, take the value at ``ceil(fraction * n) - 1``. Every reported
    latency is a latency that actually happened.
    """
    if not values:
        return 0.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered))
    return ordered[max(rank - 1, 0)]


def consistency(scores: list[float]) -> float | None:
    """Mean pairwise agreement across repeats of the same example, in [0, 1].

    Defined as the mean of ``1 - |a - b|`` over every unordered pair. 1.0 means
    every repeat produced the same score; 0.0 means they were maximally opposed.

    "Variance of the score across repeats", the obvious phrasing, is not usable
    here. For a binary scorer the variance is a deterministic function of the
    mean — ``p(1-p)`` — so it carries no information the accuracy column does
    not already have, and it peaks at 0.25 rather than at anything a reader can
    interpret. Pairwise agreement behaves the same way for binary and continuous
    scorers and reads as a percentage.

    Returns None for a single repeat, where the question is undefined. Reporting
    1.0 there would claim perfect repeatability was measured when nothing was.
    """
    if len(scores) < 2:
        return None
    pairs = [
        1.0 - abs(scores[i] - scores[j])
        for i in range(len(scores))
        for j in range(i + 1, len(scores))
    ]
    return mean(pairs)

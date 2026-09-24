"""Per-key comparison, returning a breakdown as well as an aggregate.

The breakdown is the reason this scorer is the default. An aggregate accuracy of
0.82 tells you model A won. It does not tell you that model A is perfect on
amounts and wrong on currency for every non-INR invoice, which is the sentence
that decides whether you ship it, change the prompt, or narrow the input.

A percentage is a verdict. A failing field is an instruction.
"""

from __future__ import annotations

from .exact import normalise

__all__ = ["field_match"]


def field_match(expected: object, actual: object) -> tuple[float, dict]:
    """Score dicts key by key.

    Returns ``(aggregate, {field: score})``. The aggregate is the mean over the
    keys of ``expected``; keys the model invented are reported in the breakdown
    as ``<key>.unexpected`` but do not dilute the score, because a model adding
    a field is a different problem from a model getting one wrong and averaging
    them together hides both.

    Non-dict expectations fall back to a single all-or-nothing comparison, so
    the scorer is safe to point at any examples file.
    """
    if not isinstance(expected, dict):
        matched = normalise(expected) == normalise(actual)
        return (1.0 if matched else 0.0), {}

    if not isinstance(actual, dict):
        return 0.0, {key: 0.0 for key in expected}

    breakdown = {
        key: (1.0 if normalise(value) == normalise(actual.get(key)) else 0.0)
        for key, value in expected.items()
    }

    for key in actual:
        if key not in expected:
            breakdown[f"{key}.unexpected"] = 0.0

    scored = [breakdown[key] for key in expected]
    aggregate = sum(scored) / len(scored) if scored else 0.0
    return aggregate, breakdown

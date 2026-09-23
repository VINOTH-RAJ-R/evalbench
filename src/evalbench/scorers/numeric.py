"""Numeric comparison within a relative tolerance."""

from __future__ import annotations

__all__ = ["numeric_tolerance"]

DEFAULT_TOLERANCE = 0.01


def numeric_tolerance(
    expected: object, actual: object, *, tolerance: float = DEFAULT_TOLERANCE
) -> float:
    """1.0 when the values agree within a relative tolerance, else 0.0.

    Booleans are rejected rather than treated as 0 and 1. Python says
    ``True == 1``, and a scorer that accepts True for an expected 1 reports a
    model as correct when it returned the wrong type entirely.
    """
    if not _is_number(expected) or not _is_number(actual):
        return 0.0

    expected_value = float(expected)  # type: ignore[arg-type]
    actual_value = float(actual)  # type: ignore[arg-type]

    if expected_value == actual_value:
        return 1.0
    if expected_value == 0:
        return 0.0
    return (
        1.0
        if abs(actual_value - expected_value) / abs(expected_value) <= tolerance
        else 0.0
    )


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

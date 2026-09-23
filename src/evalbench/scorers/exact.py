"""Strict equality, after a normalisation that is deliberately conservative."""

from __future__ import annotations

__all__ = ["exact", "normalise"]


def exact(expected: object, actual: object) -> float:
    """1.0 when the values match after normalisation, 0.0 otherwise."""
    return 1.0 if normalise(expected) == normalise(actual) else 0.0


def normalise(value: object) -> object:
    """Strip whitespace and case from strings; recurse into containers.

    Numbers are not coerced across types beyond int/float equality, which
    Python already gives us. Coercing "4200" to 4200 would hide a real failure:
    a model returning a string where the schema wants an integer is wrong, and
    a scorer that forgives it makes the pipeline break somewhere else instead.
    """
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    return value

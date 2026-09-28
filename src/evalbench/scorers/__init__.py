"""The scorer registry, and the escape hatch.

A scorer takes ``(expected, actual)`` and returns a score in ``[0, 1]``, plus an
optional per-field breakdown. The breakdown is what turns "model A scored 0.82"
into "model A gets the currency wrong on every non-INR invoice", and the second
sentence is the one that changes what you do next.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Union

from .exact import exact
from .field_match import field_match
from .numeric import numeric_tolerance

__all__ = ["REGISTRY", "ScoreResult", "Scorer", "load_scorer"]

ScoreResult = Union[float, "tuple[float, dict]"]
Scorer = Callable[[object, object], ScoreResult]

REGISTRY: dict[str, Scorer] = {
    "exact": exact,
    "field_match": field_match,
    "numeric_tolerance": numeric_tolerance,
}


def load_scorer(spec: str) -> tuple[str, Scorer]:
    """Resolve a scorer by registry name or ``path/to/module.py:function``.

    The path form imports and executes local Python. That is appropriate for a
    developer tool run against your own files and inappropriate to leave
    unmentioned, so the README says so in a line.
    """
    if ":" not in spec:
        if spec not in REGISTRY:
            known = ", ".join(sorted(REGISTRY))
            raise ValueError(f"unknown scorer {spec!r}; available: {known}")
        return spec, REGISTRY[spec]

    module_path, _, function_name = spec.rpartition(":")
    path = Path(module_path)
    if not path.is_file():
        raise ValueError(f"scorer module not found: {module_path}")

    module_spec = importlib.util.spec_from_file_location(path.stem, path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"could not load scorer module: {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    function = getattr(module, function_name, None)
    if function is None or not callable(function):
        raise ValueError(f"{module_path} has no callable named {function_name!r}")
    return spec, function

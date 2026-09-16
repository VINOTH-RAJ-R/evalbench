"""Compare models on your own labelled examples, and catch prompt regressions.

    evalbench run --examples examples/sample.jsonl --models mock-a,mock-b

Runs entirely offline against the mock provider, so the quickstart works with
no API key and no signup.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]

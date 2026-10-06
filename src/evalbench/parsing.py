"""Deciding whether a response was usable at all.

Parse-failure rate is the metric this tool has that public benchmarks do not,
and it is the one that matters most for a pipeline: a model that is 3% more
accurate and fails to return usable output twice as often is a worse model for
anything automated, and no leaderboard will tell you that.

Measuring it needs a parser, and the strictness of that parser changes the
number. So there are two modes and the run file records which was used.

**Strict** — ``json.loads`` on the raw response. The default.

**Recovering** — the escalation ladder from
`llm-json <https://github.com/VINOTH-RAJ-R/llm-json>`_, installed via
``pip install "evalbench[llm-json]"``. This additionally reports a *recovery
rate*: responses that were unusable as returned but salvageable, which is more
actionable than a binary failure rate because it separates "the model cannot do
this" from "the model wraps its output in prose".
"""

from __future__ import annotations

import json
from typing import Any, NamedTuple

__all__ = ["ParseOutcome", "parse_response", "recovery_available"]


class ParseOutcome(NamedTuple):
    value: Any
    parsed: bool
    recovered: bool


def recovery_available() -> bool:
    """True when the llm-json extra is installed."""
    try:
        import llm_json  # noqa: F401
    except ImportError:
        return False
    return True


def parse_response(text: str, *, recover: bool = True) -> ParseOutcome:
    """Interpret a model response as JSON.

    ``recover=True`` uses llm-json when it is available and falls back to strict
    parsing when it is not, so the tool works either way and the extra is a
    capability upgrade rather than a requirement.
    """
    try:
        return ParseOutcome(json.loads(text), True, False)
    except ValueError:
        pass

    if not recover:
        return ParseOutcome(None, False, False)

    try:
        from llm_json import ParseFailed, parse
    except ImportError:
        return ParseOutcome(None, False, False)

    try:
        return ParseOutcome(parse(text, repair=True), True, True)
    except ParseFailed:
        return ParseOutcome(None, False, False)

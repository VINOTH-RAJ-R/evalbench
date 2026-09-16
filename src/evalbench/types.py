"""The value types that move between the runner, the scorers and the report.

Everything here is frozen and JSON-representable. A run file written today has
to be diffable against one written in six weeks, which rules out anything whose
serialisation depends on the code that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "Example",
    "ModelSummary",
    "Response",
    "Result",
    "RunSummary",
]


@dataclass(frozen=True)
class Example:
    """One labelled case from the examples file."""

    id: str
    input: str
    expected: Any

    @classmethod
    def from_json(cls, raw: dict, line_number: int) -> Example:
        missing = [key for key in ("id", "input", "expected") if key not in raw]
        if missing:
            raise ValueError(
                f"line {line_number}: example is missing {', '.join(missing)}"
            )
        if not isinstance(raw["id"], str) or not raw["id"]:
            raise ValueError(f"line {line_number}: id must be a non-empty string")
        return cls(id=raw["id"], input=raw["input"], expected=raw["expected"])


@dataclass(frozen=True)
class Response:
    """What a provider returned, with the numbers needed to cost it."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Result:
    """One model's attempt at one example, on one repeat."""

    example_id: str
    model: str
    repeat: int
    score: float
    parsed: bool
    """False when the response could not be interpreted at all. Distinct from
    a score of 0.0, which means it was understood and was wrong."""
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    recovered: bool = False
    """True when the response was unusable as returned but salvageable. Only
    ever set when the llm-json extra is installed."""
    error: str | None = None
    fields: dict[str, float] = field(default_factory=dict)
    """Per-field scores, when the scorer produces them."""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelSummary:
    """Aggregate metrics for one model across every example and repeat."""

    model: str
    examples: int
    accuracy: float
    parse_failure_rate: float
    recovery_rate: float
    consistency: float | None
    latency_p50: float
    latency_p95: float
    prompt_tokens: int
    completion_tokens: int
    cost: float | None
    """None when no pricing table was supplied. The tool never estimates a
    price it was not given."""

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunSummary:
    """A complete run: every result, every metric, and how it was produced."""

    timestamp: str
    scorer: str
    models: list[str]
    repeats: int
    concurrency: int
    examples_file: str
    summaries: list[ModelSummary]
    results: list[Result]

    def to_json(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "scorer": self.scorer,
            "models": self.models,
            "repeats": self.repeats,
            "concurrency": self.concurrency,
            "examples_file": self.examples_file,
            "summaries": [s.to_json() for s in self.summaries],
            "results": [r.to_json() for r in self.results],
        }

    @classmethod
    def from_json(cls, raw: dict) -> RunSummary:
        return cls(
            timestamp=raw["timestamp"],
            scorer=raw["scorer"],
            models=raw["models"],
            repeats=raw["repeats"],
            concurrency=raw.get("concurrency", 1),
            examples_file=raw.get("examples_file", ""),
            summaries=[ModelSummary(**s) for s in raw["summaries"]],
            results=[Result(**r) for r in raw["results"]],
        )

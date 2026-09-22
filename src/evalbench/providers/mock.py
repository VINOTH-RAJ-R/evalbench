"""A deterministic offline provider.

Two reasons this is not a test fixture but a shipped component.

A reviewer who cannot run the test suite does not trust it, and a suite that
needs an API key cannot be run by anybody evaluating the repository. Every test
here runs offline with no secret present.

It also makes the README quickstart real. "Install it and run this command" that
actually works, with no signup, is worth more than a screenshot of output.

Determinism comes from ``sha256`` rather than ``hash()``. Python salts string
hashing per process, so a suite built on ``hash()`` would pass locally and fail
intermittently in CI for reasons nobody enjoys tracking down.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from ..types import Response

__all__ = ["MockProvider"]

_QUALITY = {
    "mock-a": 0.90,
    "mock-b": 0.70,
    "mock-c": 0.50,
}
_DEFAULT_QUALITY = 0.75


class MockProvider:
    """Answers deterministically, with a per-model quality and failure profile.

    ``mock-a`` is the strong model, ``mock-b`` middling, ``mock-c`` weak, and
    ``mock-c`` additionally emits unparseable output some of the time so the
    parse-failure column has something to report.
    """

    def __init__(self, name: str, *, seed: str = "") -> None:
        self.name = name
        self._seed = seed
        self._quality = _QUALITY.get(name, _DEFAULT_QUALITY)
        self._calls: dict[str, int] = {}
        self._lock = threading.Lock()

    def complete(self, prompt: str) -> Response:
        attempt = self._next_attempt(prompt)
        roll = self._roll(prompt, f"answer:{attempt}")
        expected = _expected_from_prompt(prompt)

        if self._emits_garbage(prompt, attempt):
            text = "I'm sorry, I can't produce structured output for that."
        elif roll < self._quality:
            text = json.dumps(expected)
        else:
            text = json.dumps(
                _degrade(expected, self._roll(prompt, f"degrade:{attempt}"))
            )

        if self._wraps_in_fence(prompt):
            text = f"```json\n{text}\n```"

        return Response(
            text=text,
            prompt_tokens=max(len(prompt) // 4, 1),
            completion_tokens=max(len(text) // 4, 1),
            latency_ms=self._latency(prompt),
        )

    def _next_attempt(self, prompt: str) -> int:
        """How many times this exact prompt has been seen.

        Repeats of one example must be allowed to differ, or the consistency
        metric is measuring nothing. Varying on a call counter keeps the run
        reproducible at the default concurrency of 1; above that the interleave
        is what it is, which is also true of the real providers this stands in
        for.
        """
        with self._lock:
            seen = self._calls.get(prompt, 0)
            self._calls[prompt] = seen + 1
        return seen

    def _roll(self, prompt: str, purpose: str) -> float:
        material = f"{self.name}|{self._seed}|{purpose}|{prompt}".encode()
        digest = hashlib.sha256(material).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _emits_garbage(self, prompt: str, attempt: int) -> bool:
        if self.name != "mock-c":
            return False
        return self._roll(prompt, f"garbage:{attempt}") < 0.15

    def _wraps_in_fence(self, prompt: str) -> bool:
        return self._roll(prompt, "fence") < 0.25

    def _latency(self, prompt: str) -> float:
        # A long right tail, so p50 and p95 are visibly different numbers.
        roll = self._roll(prompt, "latency")
        base = 120 + roll * 240
        if roll > 0.9:
            base *= 4
        return round(base, 1)


def _expected_from_prompt(prompt: str) -> Any:
    """Recover the labelled answer the harness embedded in the prompt.

    The mock is answering questions it has been shown the answer to, which is
    the only way an offline provider can produce a meaningful accuracy figure.
    """
    marker = "<<expected>>"
    if marker in prompt:
        return json.loads(prompt.split(marker, 1)[1].split("<</expected>>", 1)[0])
    return {}


def _degrade(expected: Any, roll: float) -> Any:
    """Produce a plausible wrong answer rather than noise.

    A wrong answer that is structurally identical and differs in one field is
    what a real model produces, and it is what makes the per-field breakdown
    worth printing.
    """
    if isinstance(expected, dict) and expected:
        keys = sorted(expected)
        target = keys[int(roll * len(keys)) % len(keys)]
        corrupted = dict(expected)
        value = corrupted[target]
        if isinstance(value, bool):
            corrupted[target] = not value
        elif isinstance(value, (int, float)):
            corrupted[target] = round(value * 1.1, 2)
        elif isinstance(value, str):
            corrupted[target] = value.upper() if value.islower() else value.lower()
        else:
            corrupted[target] = None
        return corrupted
    return expected

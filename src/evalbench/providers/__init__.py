"""The provider boundary.

One method. Adding a provider is thirty lines, which is the point: the value of
this tool is the measurement, not the integrations, and a plugin surface that
takes an afternoon to implement against is a plugin surface nobody uses.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Response

__all__ = ["Provider", "get_provider"]


@runtime_checkable
class Provider(Protocol):
    """Anything that can turn a prompt into a response with token counts.

    ``latency_ms`` is measured around the call itself, not around the queueing,
    so that a run with concurrency above 1 still reports per-request latency
    rather than contention.
    """

    name: str

    def complete(self, prompt: str) -> Response: ...


def get_provider(model: str) -> Provider:
    """Resolve a model name to a provider instance.

    Names beginning ``mock`` resolve to the deterministic offline provider, so
    the entire test suite and the README quickstart run with no API key. Every
    other name goes to the OpenAI-compatible HTTP adapter.
    """
    if model.startswith("mock"):
        from .mock import MockProvider

        return MockProvider(model)

    # Imported only on the branch that needs it, so a mock-only run never pulls
    # in httpx and never requires the HTTP adapter to be present at all.
    from .openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(model)

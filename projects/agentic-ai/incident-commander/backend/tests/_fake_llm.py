"""Test doubles for the model seam.

`FakeLLMClient` is the offline agent from `src/offline_llm.py` with no delay, so
the loop's control flow (routing, the evidence loop, the Critic, the approval
gate, halt conditions, compound faults) runs for free and deterministically.

`FailingLLMClient` and `ScriptedLLMClient` cover the unhappy paths: a model that
is down, and a model that returns something specific for one agent (used to test
the Critic veto and schema-violation handling).
"""

from __future__ import annotations

from typing import Any

from src.offline_llm import OfflineLLM


class FakeLLMClient(OfflineLLM):
    """The offline agent, instant. Records `(agent, schema)` for every call in `calls`."""

    def model_name_for(self, agent: str) -> str:  # pragma: no cover
        return "fake-model"


class FailingLLMClient:
    """Always raises - used to prove the loop degrades instead of crashing."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("model is down")
        self.calls = 0

    def model_name_for(self, agent: str) -> str:  # pragma: no cover
        return "fake-model"

    def structured(self, *args: Any, **kwargs: Any):
        self.calls += 1
        raise self.exc


class ScriptedLLMClient(FakeLLMClient):
    """A FakeLLMClient with specific schemas overridden.

    Used to force a path the competent stub would never take - a Critic veto on a
    sound action, say - without contorting the sensible default behaviour.
    """

    def __init__(self, overrides: dict[type, Any] | None = None) -> None:
        super().__init__()
        self.overrides = overrides or {}

    def structured(self, schema, *args: Any, **kwargs: Any):
        if schema in self.overrides:
            self.calls.append((kwargs.get("agent", "unknown"), schema.__name__))
            value = self.overrides[schema]
            return value(*args, **kwargs) if callable(value) else value
        return super().structured(schema, *args, **kwargs)

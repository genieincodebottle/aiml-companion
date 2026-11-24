"""Model provider adapters - the stub/live seam.

`Provider.complete(system, user, schema)` returns a validated instance of a Pydantic
schema plus token usage. Two implementations:

  - StubProvider: deterministic, offline, free. It does NOT call an LLM; it produces
    schema-valid outputs from simple heuristics over the prompt so the whole platform
    (gateway, policy, runtime, HITL, evals, tracing) runs end-to-end with no key. It
    is a faithful stand-in, not a mock - it returns the same *shape* a live model
    returns, so every downstream plane behaves identically.
  - GeminiProvider: the real model via langchain-google-genai structured output.

The gateway holds a primary and a fallback provider and fails over between them.
"""

from __future__ import annotations

import json
import logging
from typing import Type, TypeVar

from pydantic import BaseModel

from ..config import Settings
from ..models.schemas import GatewayUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a call (so the gateway can fail over)."""


class Provider:
    name: str = "base"

    def complete(self, system: str, user: str, schema: Type[T]) -> tuple[T, GatewayUsage]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Stub provider - the offline default
# --------------------------------------------------------------------------- #
class StubProvider(Provider):
    """Deterministic schema-valid outputs from prompt heuristics. No network."""

    name = "offline"

    def complete(self, system: str, user: str, schema: Type[T]) -> tuple[T, GatewayUsage]:
        from ..tenants.fincrime.stub_brain import synthesize

        obj = synthesize(schema, system, user)
        # Deterministic pseudo-usage so budget accounting is exercised in stub mode.
        in_tok = max(1, len(system) + len(user)) // 4
        out_tok = max(1, len(json.dumps(obj.model_dump(), default=str))) // 4
        usage = GatewayUsage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=_stub_cost(in_tok, out_tok),
            model="offline",
            provider="offline",
        )
        return obj, usage


def _stub_cost(in_tok: int, out_tok: int) -> float:
    # A nominal price so cost ceilings and per-case cost tracking are visibly exercised.
    return (in_tok / 1_000_000) * 0.30 + (out_tok / 1_000_000) * 2.50


# --------------------------------------------------------------------------- #
# Gemini provider - the live seam
# --------------------------------------------------------------------------- #
# Approximate Gemini pricing (USD per 1M tokens); kept here so cost is transparent.
_PRICE = {
    "gemini-pro-latest": (1.25, 10.0),
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-flash-lite-latest": (0.10, 0.40),
}


class GeminiProvider(Provider):
    """Real Gemini via langchain-google-genai with structured output."""

    name = "gemini"

    def __init__(self, model: str, settings: Settings) -> None:
        self.model = model
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._client = ChatGoogleGenerativeAI(
                model=model,
                temperature=settings.temperature,
                google_api_key=settings.google_api_key,
                timeout=settings.llm_timeout_s,
                # One SDK retry; the runtime adds one step-level retry on top, then
                # dead-letters. Unbounded retries would hide an outage as a slow case.
                max_retries=1,
            )
        except Exception as exc:  # pragma: no cover - import/runtime guard
            raise ProviderError(f"Failed to initialize Gemini ({model}): {exc}") from exc

    def complete(  # pragma: no cover - needs key
        self, system: str, user: str, schema: Type[T]
    ) -> tuple[T, GatewayUsage]:
        from langchain_core.messages import HumanMessage, SystemMessage

        bound = self._client.with_structured_output(schema, include_raw=True)
        try:
            resp = bound.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except Exception as exc:
            raise ProviderError(f"Gemini call failed: {exc}") from exc
        parsed = resp["parsed"] if isinstance(resp, dict) else resp
        if parsed is None:
            raise ProviderError("Gemini returned no parseable structured output")
        in_tok, out_tok = _gemini_usage(resp)
        pin, pout = _PRICE.get(self.model, (1.25, 10.0))
        usage = GatewayUsage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=(in_tok / 1_000_000) * pin + (out_tok / 1_000_000) * pout,
            model=self.model,
            provider="gemini",
        )
        return parsed, usage


def _gemini_usage(resp) -> tuple[int, int]:  # pragma: no cover - needs key
    try:
        raw = resp.get("raw") if isinstance(resp, dict) else None
        meta = getattr(raw, "usage_metadata", None) if raw is not None else None
        if meta:
            return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    except Exception:
        pass
    return 0, 0

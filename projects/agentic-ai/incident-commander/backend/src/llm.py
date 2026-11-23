"""Gemini access with structured output, model tiering, budgets, and resilience.

This module is the single seam between the agents and the model. Everything that
makes an LLM call *safe to run unattended* lives here rather than in the agents:

- **Structured output.** Every call is bound to a Pydantic schema via
  `with_structured_output`, so agents receive validated objects, never raw text.
- **Model tiering.** Classification-shaped agents (Triage, Verifier, Postmortem)
  can run on a cheaper model than the reasoning agents (Diagnostician, Remediation,
  Critic). See `config.FAST_TIER_AGENTS`.
- **Budget accounting.** `TokenTracker` accumulates usage per incident *and per
  agent*, so the loop can enforce a hard cost ceiling and a learner can see which
  agent is actually expensive.
- **Cost pre-flight.** `estimate_cost` gives a pre-call estimate so the loop can
  refuse a call that would breach the ceiling, instead of discovering the breach
  after paying for it.
- **Timeouts, bounded retries with jittered backoff, and a circuit breaker.**
  A model that is down should fail fast for the whole process, not have every
  incident independently retry it into the ground.

`get_llm()` returns either this Gemini client or the offline agent
(`src/offline_llm.py`), depending on `settings.llm_mode`. The choice is explicit
and visible (health endpoint, console banner, incident record), never a silent
fallback after a failure: if Gemini is selected and goes down mid-incident, the
incident escalates rather than switching brains (ADR-0009).
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from .config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Approximate published pricing (USD per 1M tokens), kept here so the cost ceiling
# is transparent and easy to update. Unknown models fall back to _DEFAULT_PRICE.
_PRICING: dict[str, tuple[float, float]] = {
    # model prefix: (input $/1M, output $/1M). Approximate; check Google's
    # pricing page before relying on the cost ceiling for real spend.
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-pro-latest": (1.25, 10.00),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}
_DEFAULT_PRICE = (0.30, 2.50)

#: Rough characters-per-token ratio used only for pre-flight cost estimation.
_CHARS_PER_TOKEN = 4
#: Assumed output size for a structured response when estimating pre-flight cost.
_ASSUMED_OUTPUT_TOKENS = 400


def price_for(model: str) -> tuple[float, float]:
    """(input, output) $/1M tokens for a model name, longest-prefix match."""
    for prefix in sorted(_PRICING, key=len, reverse=True):
        if model.startswith(prefix):
            return _PRICING[prefix]
    return _DEFAULT_PRICE


def cost_of(model: str, in_tok: int, out_tok: int) -> float:
    p_in, p_out = price_for(model)
    return (in_tok / 1_000_000) * p_in + (out_tok / 1_000_000) * p_out


def estimate_cost(model: str, system: str, user: str, out_tokens: int = _ASSUMED_OUTPUT_TOKENS) -> float:
    """Pre-flight cost estimate for a call. Deliberately conservative (rounds up)."""
    in_tok = (len(system) + len(user)) // _CHARS_PER_TOKEN + 1
    return cost_of(model, in_tok, out_tokens)


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


@dataclass
class AgentUsage:
    """Per-agent usage inside one incident."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retries: int = 0
    latency_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens": self.input_tokens + self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "retries": self.retries,
            "latency_s": round(self.latency_s, 3),
            "mean_latency_s": round(self.latency_s / self.calls, 3) if self.calls else 0.0,
        }


@dataclass
class TokenTracker:
    """Process-wide usage ledger, bucketed by incident and by agent.

    Thread-safe because agents run in worker threads (`asyncio.to_thread`).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    _by_incident: dict[str, dict[str, AgentUsage]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        incident_id: str,
        in_tok: int,
        out_tok: int,
        *,
        agent: str = "unknown",
        model: str = "",
        retries: int = 0,
        latency_s: float = 0.0,
    ) -> None:
        cost = cost_of(model or get_settings().model_name, in_tok, out_tok)
        with self._lock:
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            self.calls += 1
            bucket = self._by_incident.setdefault(incident_id, {})
            usage = bucket.setdefault(agent, AgentUsage())
            usage.calls += 1
            usage.input_tokens += in_tok
            usage.output_tokens += out_tok
            usage.cost_usd += cost
            usage.retries += retries
            usage.latency_s += latency_s

    def incident_cost(self, incident_id: str) -> tuple[int, float]:
        """(total tokens, total USD) for one incident."""
        with self._lock:
            bucket = self._by_incident.get(incident_id, {})
            tokens = sum(u.input_tokens + u.output_tokens for u in bucket.values())
            cost = sum(u.cost_usd for u in bucket.values())
        return tokens, cost

    def incident_breakdown(self, incident_id: str) -> dict[str, dict]:
        """Per-agent usage for one incident - the view that shows *where* cost goes."""
        with self._lock:
            bucket = self._by_incident.get(incident_id, {})
            return {agent: usage.as_dict() for agent, usage in sorted(bucket.items())}

    def forget(self, incident_id: str) -> None:
        """Drop an incident's ledger (called when the incident is evicted)."""
        with self._lock:
            self._by_incident.pop(incident_id, None)

    @staticmethod
    def cost(in_tok: int, out_tok: int) -> float:
        """Cost at the configured default model's price."""
        return cost_of(get_settings().model_name, in_tok, out_tok)


token_tracker = TokenTracker()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMConfigError(RuntimeError):
    """Raised when Gemini cannot be initialized (missing key, bad install, etc.)."""


class LLMBudgetError(RuntimeError):
    """Raised when a call is refused because it would breach the cost ceiling."""


class LLMUnavailableError(RuntimeError):
    """Raised when the circuit breaker is open, or all retries were exhausted."""


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreaker:
    """Trips after N consecutive failures; stays open for a cooldown window.

    Why: without this, every concurrent incident independently retries a dead
    model, multiplying latency and quota burn during an outage. Failing fast and
    escalating to a human is the correct behaviour for an SRE tool.
    """

    threshold: int
    cooldown_s: float
    failures: int = 0
    opened_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        with self._lock:
            if self.opened_at is None:
                return
            elapsed = time.monotonic() - self.opened_at
            if elapsed < self.cooldown_s:
                raise LLMUnavailableError(
                    f"LLM circuit breaker is open after {self.failures} consecutive failures; "
                    f"retrying in {self.cooldown_s - elapsed:.0f}s. Escalate to a human."
                )
            # Cooldown elapsed: half-open. Calls may flow again, but one more
            # failure re-opens the breaker immediately - that is what makes the
            # next call a probe. Resetting `failures` to zero here (the original
            # code) silently implemented full reset: a still-dead model earned a
            # whole fresh threshold of retries after every cooldown.
            self.opened_at = None
            self.failures = self.threshold - 1

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold and self.opened_at is None:
                self.opened_at = time.monotonic()
                logger.error("LLM circuit breaker OPEN after %d consecutive failures", self.failures)

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self.opened_at is not None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin resilient wrapper around ChatGoogleGenerativeAI with structured output.

    Construction FAILS LOUDLY when no API key is configured.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.llm_enabled:
            raise LLMConfigError(
                "GOOGLE_API_KEY is not configured. This is an agentic system and Gemini "
                "is required for every decision. Set GOOGLE_API_KEY in backend/.env "
                "(free key at https://aistudio.google.com/apikey)."
            )
        self.breaker = CircuitBreaker(
            threshold=self.settings.llm_breaker_threshold,
            cooldown_s=self.settings.llm_breaker_cooldown_s,
        )
        self._models: dict[str, object] = {}
        # Build the default model eagerly so misconfiguration surfaces at startup
        # rather than mid-incident.
        self._model_for_name(self.settings.model_name)
        logger.info(
            "Gemini initialized: strong=%s fast=%s tiering=%s",
            self.settings.model_name,
            self.settings.model_fast or "(disabled)",
            self.settings.model_tiering,
        )

    # --- model construction / tiering --------------------------------------

    def _model_for_name(self, model_name: str):
        cached = self._models.get(model_name)
        if cached is not None:
            return cached
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.settings.temperature,
                google_api_key=self.settings.google_api_key,
                timeout=self.settings.llm_timeout_s,
                max_retries=0,  # we own the retry policy (see `structured`)
            )
        except LLMConfigError:
            raise
        except Exception as exc:  # pragma: no cover - import/runtime guard
            raise LLMConfigError(f"Failed to initialize Gemini ({model_name}): {exc}") from exc
        self._models[model_name] = model
        return model

    def model_name_for(self, agent: str) -> str:
        return self.settings.model_for(agent)

    # --- the one call every agent makes ------------------------------------

    def structured(
        self,
        schema: Type[T],
        system: str,
        user: str,
        incident_id: str = "global",
        max_retries: int | None = None,
        *,
        agent: str = "unknown",
        budget_remaining_usd: float | None = None,
    ) -> T:
        """Call Gemini and return a validated instance of `schema`.

        `agent` selects the model tier and labels the usage ledger.
        `budget_remaining_usd`, when given, makes this call refuse to run if its
        pre-flight estimate exceeds what is left of the incident's cost ceiling.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        model_name = self.model_name_for(agent)

        # --- pre-flight budget check (refuse before paying) -----------------
        if budget_remaining_usd is not None:
            projected = estimate_cost(model_name, system, user)
            if projected > budget_remaining_usd:
                raise LLMBudgetError(
                    f"{agent}: estimated call cost ${projected:.5f} exceeds remaining incident "
                    f"budget ${budget_remaining_usd:.5f}; refusing the call."
                )

        self.breaker.check()

        retries = max_retries if max_retries is not None else self.settings.llm_max_retries
        bound = self._model_for_name(model_name).with_structured_output(schema, include_raw=True)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]

        last_exc: Exception | None = None
        started = time.monotonic()
        for attempt in range(retries + 1):
            try:
                resp = bound.invoke(messages)
                parsed = resp["parsed"] if isinstance(resp, dict) else resp
                self._record_usage(
                    resp,
                    incident_id,
                    agent=agent,
                    model=model_name,
                    retries=attempt,
                    latency_s=time.monotonic() - started,
                )
                if parsed is None:
                    raise ValueError("Model returned no parseable structured output")
                self.breaker.record_success()
                return parsed
            except (ValidationError, ValueError) as exc:
                # Schema violation: structured decoding is stochastic, so one retry is
                # worth it, but we do not retry forever on a prompt the model cannot satisfy.
                last_exc = exc
                logger.warning("%s: structured output invalid (attempt %d): %s", agent, attempt + 1, exc)
                if attempt < retries:
                    time.sleep(self._backoff(attempt))
            except Exception as exc:  # noqa: BLE001 - transport / rate-limit / transient
                last_exc = exc
                logger.warning("%s: LLM call failed (attempt %d): %s", agent, attempt + 1, exc)
                if attempt < retries:
                    time.sleep(self._backoff(attempt))

        self.breaker.record_failure()
        raise LLMUnavailableError(f"{agent}: LLM call failed after {retries + 1} attempts: {last_exc}")

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff with full jitter (avoids synchronized retry storms)."""
        return random.uniform(0.0, 0.5 * (2**attempt))

    def _record_usage(
        self,
        resp,
        incident_id: str,
        *,
        agent: str,
        model: str,
        retries: int,
        latency_s: float,
    ) -> None:
        try:
            raw = resp.get("raw") if isinstance(resp, dict) else None
            meta = getattr(raw, "usage_metadata", None) if raw is not None else None
            if meta:
                in_tok = int(meta.get("input_tokens", 0))
                out_tok = int(meta.get("output_tokens", 0))
            else:
                in_tok, out_tok = 0, 0
        except Exception:  # pragma: no cover - usage metadata is best-effort
            in_tok, out_tok = 0, 0
        token_tracker.record(
            incident_id,
            in_tok,
            out_tok,
            agent=agent,
            model=model,
            retries=retries,
            latency_s=latency_s,
        )


_client = None


def get_llm():
    """The process-wide model client: Gemini, or the offline agent.

    Raises LLMConfigError when Gemini mode is forced (`IC_LLM_MODE=gemini`) and no
    key is configured.
    """
    global _client
    if _client is None:
        settings = get_settings()
        if settings.llm_mode == "offline":
            from .offline_llm import OfflineLLM

            logger.info("offline mode: agents are the deterministic rule-based runbook, not Gemini")
            _client = OfflineLLM(step_delay_s=settings.offline_step_delay_s)
        else:
            _client = LLMClient()
    return _client


def reset_llm_cache() -> None:
    """Test hook: drop the memoized client so settings changes take effect."""
    global _client
    _client = None

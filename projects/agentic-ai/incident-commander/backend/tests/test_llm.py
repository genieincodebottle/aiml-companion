"""LLM seam: pricing, budgets, the circuit breaker, and per-agent accounting.

None of this touches a real model. `LLMClient` is only constructed where a fake
model factory is injected, because the point is to test the *resilience policy*,
which is ours, not the provider's.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from src.config import FAST_TIER_AGENTS, STRONG_TIER_AGENTS, get_settings
from src.llm import (
    CircuitBreaker,
    LLMBudgetError,
    LLMConfigError,
    LLMUnavailableError,
    TokenTracker,
    cost_of,
    estimate_cost,
    price_for,
)
from src.models.schemas import TriageResult


class _Tiny(BaseModel):
    value: str = "x"


# --- pricing ----------------------------------------------------------------


def test_pricing_uses_longest_prefix_match():
    """flash-lite must not be priced as flash: the prefix order is load-bearing."""
    assert price_for("gemini-2.5-flash-lite") == (0.10, 0.40)
    assert price_for("gemini-2.5-flash") == (0.30, 2.50)
    assert price_for("gemini-2.5-pro") == (1.25, 10.00)


def test_unknown_model_falls_back_to_a_conservative_price():
    """Under-pricing an unknown model would silently defeat the cost ceiling."""
    assert price_for("some-future-model") == (0.30, 2.50)


def test_cost_arithmetic():
    assert cost_of("gemini-2.5-flash", 1_000_000, 0) == pytest.approx(0.30)
    assert cost_of("gemini-2.5-flash", 0, 1_000_000) == pytest.approx(2.50)


def test_estimate_is_conservative():
    """A pre-flight estimate that under-reads is worse than none."""
    estimate = estimate_cost("gemini-2.5-flash", "system " * 100, "user " * 100)
    assert estimate > 0


# --- per-agent accounting ---------------------------------------------------


def test_usage_is_bucketed_by_incident_and_agent():
    """ "It cost $0.008" is not actionable; "the Diagnostician is 71% of it" is."""
    tracker = TokenTracker()
    tracker.record("inc_1", 1000, 500, agent="Diagnostician", model="gemini-2.5-flash", latency_s=1.2)
    tracker.record("inc_1", 200, 100, agent="Triage", model="gemini-2.5-flash", latency_s=0.3)
    tracker.record("inc_2", 999, 999, agent="Triage", model="gemini-2.5-flash")

    tokens, cost = tracker.incident_cost("inc_1")
    assert tokens == 1800
    assert cost > 0

    breakdown = tracker.incident_breakdown("inc_1")
    assert set(breakdown) == {"Diagnostician", "Triage"}
    assert breakdown["Diagnostician"]["tokens"] == 1500
    assert breakdown["Diagnostician"]["cost_usd"] > breakdown["Triage"]["cost_usd"]
    assert breakdown["Diagnostician"]["mean_latency_s"] == pytest.approx(1.2)

    # Incidents must not leak into each other's ledgers.
    assert tracker.incident_cost("inc_2")[0] == 1998


def test_forget_releases_an_evicted_incidents_ledger():
    """The ledger is another unbounded-growth path in a long-running process."""
    tracker = TokenTracker()
    tracker.record("inc_1", 10, 10, agent="Triage")
    tracker.forget("inc_1")
    assert tracker.incident_cost("inc_1") == (0, 0.0)


def test_unknown_incident_costs_nothing():
    assert TokenTracker().incident_cost("never_seen") == (0, 0.0)


# --- model tiering ----------------------------------------------------------


def test_tiering_is_disabled_unless_a_fast_model_is_configured(env):
    """Single-model out of the box; tiering is opt-in, so the default is predictable."""
    env(IC_MODEL_NAME="gemini-2.5-pro", IC_MODEL_FAST="")
    settings = get_settings()
    assert settings.model_for("Triage") == "gemini-2.5-pro"
    assert settings.model_for("Diagnostician") == "gemini-2.5-pro"


def test_tiering_routes_cheap_agents_to_the_cheap_model(env):
    env(IC_MODEL_NAME="gemini-2.5-pro", IC_MODEL_FAST="gemini-2.5-flash-lite")
    settings = get_settings()
    for agent in FAST_TIER_AGENTS:
        assert settings.model_for(agent) == "gemini-2.5-flash-lite", agent
    for agent in STRONG_TIER_AGENTS:
        assert settings.model_for(agent) == "gemini-2.5-pro", agent


def test_every_agent_is_assigned_to_exactly_one_tier():
    """An agent in neither set silently gets the expensive model forever."""
    assert not (FAST_TIER_AGENTS & STRONG_TIER_AGENTS)
    from src.models.state import AgentName

    named = {a.value for a in AgentName} - {"Orchestrator"}
    assert named <= (FAST_TIER_AGENTS | STRONG_TIER_AGENTS)


# --- circuit breaker --------------------------------------------------------


def test_breaker_opens_after_consecutive_failures_and_then_recovers():
    """During a provider outage, every incident retrying independently is the problem."""
    breaker = CircuitBreaker(threshold=3, cooldown_s=0.2)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.is_open
    with pytest.raises(LLMUnavailableError, match="circuit breaker"):
        breaker.check()

    time.sleep(0.25)
    breaker.check()  # half-open: one call is let through
    assert not breaker.is_open


def test_a_success_resets_the_failure_count():
    breaker = CircuitBreaker(threshold=3, cooldown_s=10)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert not breaker.is_open


# --- construction and budget refusal ----------------------------------------


def test_client_refuses_to_construct_without_a_key(env):
    """No key means no incident. There is deliberately no non-agentic fallback."""
    env(GOOGLE_API_KEY="")
    from src.llm import LLMClient

    with pytest.raises(LLMConfigError, match="GOOGLE_API_KEY"):
        LLMClient()


def test_placeholder_key_is_not_treated_as_configured(env):
    """The .env.example placeholder must not look like a real key."""
    env(GOOGLE_API_KEY="your_google_api_key_here")
    assert get_settings().llm_enabled is False


def test_a_call_is_refused_before_it_breaches_the_budget(env, monkeypatch):
    """Pre-flight refusal, not post-hoc discovery.

    The original loop checked cost only at the top of an iteration, so one pass
    could blow well past the ceiling before anyone noticed. This refuses first.
    """
    env(GOOGLE_API_KEY="test-key")
    from src.llm import LLMClient

    monkeypatch.setattr(LLMClient, "_model_for_name", lambda self, name: object())
    client = LLMClient()
    with pytest.raises(LLMBudgetError, match="exceeds remaining incident budget"):
        client.structured(
            TriageResult,
            "system prompt " * 500,
            "user prompt " * 500,
            agent="Triage",
            budget_remaining_usd=0.0,
        )


def test_backoff_is_jittered():
    """Un-jittered backoff synchronizes retries into a thundering herd."""
    from src.llm import LLMClient

    delays = {LLMClient._backoff(2) for _ in range(50)}  # noqa: SLF001
    assert len(delays) > 1, "backoff must not be a constant"
    assert all(0 <= d <= 2.0 for d in delays)

"""Gateway: redaction, semantic cache, routing, budget enforcement, audit."""

from __future__ import annotations

import dataclasses

import pytest

from src.config import get_settings
from src.gateway import Gateway
from src.gateway.cache import SemanticCache
from src.gateway.core import BudgetExceeded
from src.gateway.redaction import redact
from src.models.schemas import ModelTier, TriageResult


def test_redaction_masks_and_is_stable():
    r = redact("email a@b.com twice: a@b.com and ssn 123-45-6789")
    assert "a@b.com" not in r.text
    assert "123-45-6789" not in r.text
    # Same value -> same placeholder (co-reference preserved).
    assert r.text.count("[EMAIL_1]") == 2
    assert r.count == 2


def test_semantic_cache_hits_near_duplicates_and_misses_novel():
    cache = SemanticCache(threshold=0.9)
    cache.store("suspicious transfers to shell company", "k", '{"x":1}')
    assert cache.lookup("suspicious transfers to shell company", "k") == '{"x":1}'
    assert cache.lookup("weather forecast for tuesday", "k") is None


def test_cache_never_crosses_routing_keys():
    cache = SemanticCache(threshold=0.9)
    cache.store("same prompt", "model-a", "A")
    assert cache.lookup("same prompt", "model-b") is None


def test_gateway_routes_tier_and_tracks_cost(settings):
    gw = Gateway(settings)
    obj, usage = gw.complete(
        workflow_id="wf1", system="s", user="Alert with structuring", schema=TriageResult, tier=ModelTier.CHEAP
    )
    assert isinstance(obj, TriageResult)
    assert usage.cost_usd > 0
    assert gw.cost_of("wf1") == pytest.approx(usage.cost_usd)


def test_gateway_second_identical_call_is_cached_and_free(settings):
    gw = Gateway(settings)
    gw.complete(workflow_id="wf1", system="s", user="same alert text", schema=TriageResult)
    _, usage2 = gw.complete(workflow_id="wf1", system="s", user="same alert text", schema=TriageResult)
    assert usage2.cached is True
    assert usage2.cost_usd == 0.0


def test_gateway_cache_never_crosses_workflows(settings):
    """Two cases with near-identical templated prompts must not share responses:
    cross-serving one case's screening/narrative to another is a correctness bug."""
    gw = Gateway(settings)
    gw.complete(workflow_id="wf-case-a", system="s", user="same alert text", schema=TriageResult)
    _, usage = gw.complete(workflow_id="wf-case-b", system="s", user="same alert text", schema=TriageResult)
    assert usage.cached is False


def test_gateway_enforces_cost_budget():
    tiny = dataclasses.replace(get_settings(), max_cost_usd=0.0)
    gw = Gateway(tiny)
    with pytest.raises(BudgetExceeded):
        gw.complete(workflow_id="wf1", system="s", user="u", schema=TriageResult)

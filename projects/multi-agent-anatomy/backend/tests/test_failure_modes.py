"""The four failure modes, asserted.

Every test here corresponds to one toggle in the failure panel. If a test fails,
the demo it backs is lying.
"""

from __future__ import annotations

import pytest

from app.failures import FailureSwitches
from app.pipeline import run_request
from app.tools.db import init_db

QUESTION = "Where is my order ORD-4412 and can I still return it?"


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db(force=True)


def _spans(result, **match):
    return [
        s
        for s in result["trace"]["spans"]
        if all(s.get(k) == v for k, v in match.items())
    ]


@pytest.mark.asyncio
async def test_killed_agent_still_produces_an_answer():
    result = await run_request(
        QUESTION, switches=FailureSwitches(kill_shipping_agent=True)
    )
    assert result["reply"], "a dead branch must not empty the reply"
    assert any(s["status"] == "killed" for s in result["trace"]["spans"])
    # The gap is declared rather than filled with a plausible tracking number.
    assert any("courier" in g or "tracking" in g for g in result["gaps"])
    assert "MX-" not in result["reply"]


@pytest.mark.asyncio
async def test_slow_tool_is_cut_by_its_budget_not_by_the_request_deadline():
    result = await run_request(
        QUESTION,
        switches=FailureSwitches(slow_tool_seconds=12.0, slow_tool_target="get_shipment"),
    )
    tool = _spans(result, name="tool:get_shipment")[0]
    assert tool["status"] == "timeout"
    # 12s of latency, cut well inside the 30s whole-request deadline.
    assert tool["duration_ms"] < 12_000
    assert result["reply"]


@pytest.mark.asyncio
async def test_saga_runs_its_undos_in_reverse():
    result = await run_request(
        "Please book the return for ORD-4412 and refund it",
        switches=FailureSwitches(fail_order_update=True),
    )
    saga = [s for s in result["trace"]["spans"] if s["kind"] == "saga"]
    order = [s["name"] for s in saga if s["name"] != "saga:book-return"]
    assert order == [
        "book_courier",
        "charge_fee",
        "update_order",
        "undo:charge_fee",
        "undo:book_courier",
    ]


@pytest.mark.asyncio
async def test_corrupt_passage_leaves_every_span_green_and_the_answer_wrong():
    """The one that matters.

    This is the observability section made executable: status is not truth.
    """
    good = await run_request(QUESTION)
    bad = await run_request(QUESTION, switches=FailureSwitches(corrupt_passage=True))

    assert "30 days" in good["reply"]
    assert "14 days" in bad["reply"]

    # Nothing failed. Not one span.
    assert bad["all_spans_green"] is True
    assert all(s["status"] == "ok" for s in bad["trace"]["spans"])

    # The only signal is index age, and it is a warning rather than an error,
    # because the request genuinely succeeded.
    assert bad["semantic_warnings"]
    assert any("index age" in w for w in bad["semantic_warnings"])


@pytest.mark.asyncio
async def test_cache_off_costs_more_than_cache_on():
    warm = await run_request(QUESTION, reset_cache=True)
    warm = await run_request(QUESTION)
    cold = await run_request(QUESTION, switches=FailureSwitches(disable_prompt_cache=True))
    assert warm["trace"]["summary"]["cached_tokens"] > 0
    assert cold["trace"]["summary"]["cached_tokens"] == 0
    assert cold["trace"]["summary"]["total_cost_usd"] > warm["trace"]["summary"]["total_cost_usd"]


@pytest.mark.asyncio
async def test_cross_tenant_attempt_returns_nothing_from_the_other_tenant():
    result = await run_request(
        QUESTION, switches=FailureSwitches(attempt_cross_tenant=True)
    )
    retrieval = [s for s in result["trace"]["spans"] if s["kind"] == "retrieval"][0]
    assert retrieval["detail"]["cross_tenant_attempted"] is True
    assert retrieval["detail"]["cross_tenant_passages_returned"] == 0
    assert "Contoso" not in result["reply"]

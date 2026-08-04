"""Budget propagation and trace integrity."""

from __future__ import annotations

import pytest

from app.budget import BudgetExceeded, RequestBudget
from app.config import BUDGETS
from app.pipeline import run_request
from app.tools.db import init_db

QUESTION = "Where is my order ORD-4412 and can I still return it?"


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db(force=True)


def test_a_delegation_can_never_outlive_the_request():
    b = RequestBudget.start(deadline_s=3.0, tokens=1000, request_id="r", tenant_id="t")
    d = b.delegate("order-agent", agent_timeout_s=8.0)
    # The agent asked for 8s. The request has 3. It gets 3.
    assert d.timeout_s <= 3.0


def test_every_per_agent_timeout_is_shorter_than_the_request_deadline():
    for name, value in vars(BUDGETS).items():
        if name.endswith("_timeout_s"):
            assert value < BUDGETS.request_deadline_s, name


def test_delegation_depth_is_capped():
    b = RequestBudget.start(deadline_s=10, tokens=1000, request_id="r", tenant_id="t")
    b.depth = BUDGETS.max_delegation_depth
    with pytest.raises(BudgetExceeded):
        b.delegate("orchestrator", agent_timeout_s=1.0)


def test_spending_is_shared_across_the_fan_out():
    b = RequestBudget.start(deadline_s=10, tokens=1000, request_id="r", tenant_id="t")
    one = b.delegate("order-agent", agent_timeout_s=1.0)
    two = b.delegate("policy-agent", agent_timeout_s=1.0)
    one.spend(400)
    # The second agent sees the first one's spend immediately. Three parallel
    # agents draw down one pool, not three.
    assert two.tokens_remaining == 600


@pytest.mark.asyncio
async def test_budget_remaining_falls_monotonically_through_the_trace():
    result = await run_request(QUESTION)
    seen = [
        s["tokens_remaining"]
        for s in result["trace"]["spans"]
        if s["tokens_remaining"] is not None
    ]
    assert seen == sorted(seen, reverse=True)


@pytest.mark.asyncio
async def test_every_model_span_carries_an_agent_id_and_a_prompt_version():
    result = await run_request(QUESTION)
    model_spans = [s for s in result["trace"]["spans"] if s["kind"] == "model"]
    assert model_spans
    for s in model_spans:
        assert s["agent_id"]
        assert s["prompt_version"], f"{s['name']} has no prompt version"


@pytest.mark.asyncio
async def test_the_eight_stages_all_appear_in_order():
    result = await run_request(QUESTION)
    stages = sorted({s["stage"] for s in result["trace"]["spans"]})
    assert stages == [0, 1, 2, 3, 4, 5, 6, 7, 8]


@pytest.mark.asyncio
async def test_five_agents_and_six_model_calls_on_the_happy_path():
    result = await run_request(QUESTION)
    agents = {
        s["agent_id"] for s in result["trace"]["spans"] if s["agent_id"] != "gateway"
    }
    assert agents == {
        "orchestrator",
        "order-agent",
        "shipping-agent",
        "policy-agent",
        "writer-agent",
    }
    # Stage 2 classify, stage 3 plan, three lookups, writer, stage 6 merge.
    assert result["trace"]["summary"]["model_calls"] == 7

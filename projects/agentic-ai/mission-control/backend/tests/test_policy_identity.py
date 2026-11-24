"""Policy engine (least privilege, fail-closed) and workload identity (scope, expiry)."""

from __future__ import annotations

from src.identity import IdentityBroker
from src.models.schemas import AgentCard, ModelTier, PolicyEffect
from src.policy import PolicyEngine
from src.registry import AgentRegistry


def _card(tools, side_effecting=False):
    return AgentCard(
        name="a",
        role="r",
        description="d",
        model_tier=ModelTier.CHEAP,
        allowed_tools=tools,
        side_effecting=side_effecting,
    )


def test_tool_not_on_card_is_denied():
    eng = PolicyEngine()
    d = eng.evaluate(_card(["graph.expand"]), "case.file_sar")
    assert d.effect == PolicyEffect.DENY
    assert d.rule == "deny-tool-not-on-card"


def test_read_only_tool_is_allowed():
    eng = PolicyEngine()
    d = eng.evaluate(_card(["graph.expand"]), "graph.expand")
    assert d.effect == PolicyEffect.ALLOW


def test_risky_action_requires_approval():
    eng = PolicyEngine()
    card = _card(["case.file_sar"], side_effecting=True)
    d = eng.evaluate(card, "case.file_sar")
    assert d.effect == PolicyEffect.REQUIRE_APPROVAL


def test_large_amount_requires_approval():
    eng = PolicyEngine()
    card = _card(["case.close"], side_effecting=True)
    d = eng.evaluate(card, "case.close", {"amount_usd": 250_000})
    assert d.effect == PolicyEffect.REQUIRE_APPROVAL


def test_disabled_policy_allows_everything():
    eng = PolicyEngine(disabled=True)
    assert eng.evaluate(_card([]), "anything").effect == PolicyEffect.ALLOW


def test_identity_scope_and_tamper_detection():
    broker = IdentityBroker(b"test-key")
    card = AgentRegistry().get("fincrime.graph")
    token = broker.mint(card)
    ok, _ = broker.verify(token, "graph.expand")
    assert ok
    bad, reason = broker.verify(token, "case.file_sar")
    assert not bad and "scope" in reason
    tampered = token.__class__(
        agent=token.agent, scopes=("case.file_sar",), expires_at=token.expires_at, signature=token.signature
    )
    ok2, reason2 = broker.verify(tampered, "case.file_sar")
    assert not ok2 and "signature" in reason2


def test_identity_expiry():
    broker = IdentityBroker(b"test-key")
    card = AgentRegistry().get("fincrime.graph")
    token = broker.mint(card, ttl_s=10, now=1000.0)
    ok, reason = broker.verify(token, "graph.expand", now=2000.0)
    assert not ok and "expired" in reason

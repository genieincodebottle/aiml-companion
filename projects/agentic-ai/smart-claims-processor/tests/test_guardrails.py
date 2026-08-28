"""Tests for the GuardrailsManager."""

import pytest
from unittest.mock import patch
from src.guardrails.manager import GuardrailsManager, GuardrailsViolation


def test_agent_call_limit():
    manager = GuardrailsManager("CLM-TEST-001")
    manager.cfg["max_agent_calls"] = 3

    manager.post_check("agent_a", None)
    manager.post_check("agent_b", None)
    manager.post_check("agent_c", None)

    with pytest.raises(GuardrailsViolation, match="call limit"):
        manager.pre_check("agent_d")


def test_cost_limit():
    manager = GuardrailsManager("CLM-TEST-002")
    manager.cfg["max_cost_usd"] = 0.10
    manager.total_cost = 0.11

    with pytest.raises(GuardrailsViolation, match="Cost budget"):
        manager.pre_check("any_agent")


def test_token_limit():
    manager = GuardrailsManager("CLM-TEST-003")
    manager.cfg["max_tokens_per_claim"] = 1000
    manager.total_tokens = 1001

    with pytest.raises(GuardrailsViolation, match="Token budget"):
        manager.pre_check("any_agent")


def test_loop_detection():
    manager = GuardrailsManager("CLM-TEST-004")
    manager.cfg["max_loop_iterations"] = 3

    for _ in range(3):
        manager._agent_call_history.append("repeat_agent")

    with pytest.raises(GuardrailsViolation, match="Loop detected"):
        manager.pre_check("repeat_agent")


def test_timeout_soft_stop():
    manager = GuardrailsManager("CLM-TEST-005")
    manager.cfg["max_execution_seconds"] = 0  # Already expired
    manager.start_time = 0  # Very old start time

    result = manager.pre_check("any_agent")
    assert result is False
    assert len(manager.violations) > 0


def test_usage_summary():
    manager = GuardrailsManager("CLM-TEST-006")
    manager.agent_call_count = 5
    manager.total_tokens = 12000
    manager.total_cost = 0.004

    summary = manager.get_usage_summary()
    assert summary["agent_call_count"] == 5
    assert summary["total_tokens_used"] == 12000
    assert summary["guardrails_passed"] is True


def test_low_confidence_warning():
    manager = GuardrailsManager("CLM-TEST-007")

    class MockOutput:
        confidence = 0.30

    result = manager._check_confidence("intake", MockOutput())
    assert result is False
    assert len(manager.violations) > 0


def test_empty_reasoning_flagged():
    manager = GuardrailsManager("CLM-TEST-008")

    class MockOutput:
        intake_notes = ""  # Empty reasoning

    result = manager._check_hallucination("intake", MockOutput())
    assert result is False

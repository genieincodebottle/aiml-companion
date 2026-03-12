"""Tests for the guardrails manager."""

import time
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.guardrails.manager import GuardrailManager, PII_PATTERNS


class TestGuardrailPreChecks:
    def setup_method(self):
        self.gm = GuardrailManager(config={
            "max_agent_calls": 5,
            "max_reasoning_iterations": 3,
            "max_execution_time_seconds": 10,
            "hallucination_check_enabled": True,
            "confidence_threshold": 0.6,
            "pii_detection_enabled": True,
            "source_grounding_required": True,
        })
        self.gm.start_pipeline()

    def test_passes_initially(self):
        result = self.gm.pre_check("financial_analyst")
        assert result["passed"] is True

    def test_blocks_after_call_limit(self):
        for _ in range(5):
            self.gm.pre_check("test_agent")

        result = self.gm.pre_check("test_agent")
        assert result["passed"] is False
        assert "call limit" in result["reason"].lower()

    def test_blocks_agent_loop(self):
        for _ in range(3):
            self.gm.pre_check("loop_agent")

        result = self.gm.pre_check("loop_agent")
        assert result["passed"] is False
        assert "loop limit" in result["reason"].lower()

    def test_tracks_violations(self):
        for _ in range(5):
            self.gm.pre_check("test_agent")
        self.gm.pre_check("test_agent")  # This should fail

        violations = self.gm.get_violations()
        assert len(violations) >= 1
        assert violations[0]["agent"] == "test_agent"


class TestGuardrailPostChecks:
    def setup_method(self):
        self.gm = GuardrailManager(config={
            "max_agent_calls": 30,
            "max_reasoning_iterations": 10,
            "max_execution_time_seconds": 300,
            "hallucination_check_enabled": True,
            "confidence_threshold": 0.6,
            "pii_detection_enabled": True,
            "source_grounding_required": True,
        })
        self.gm.start_pipeline()

    def test_tracks_token_usage(self):
        self.gm.post_check("agent1", "output text", tokens_used=500, cost_usd=0.01)
        assert self.gm.state.total_tokens_used == 500
        assert self.gm.state.total_cost_usd == 0.01

    def test_cumulative_tracking(self):
        self.gm.post_check("agent1", "text", tokens_used=500, cost_usd=0.01)
        self.gm.post_check("agent2", "text", tokens_used=300, cost_usd=0.005)
        assert self.gm.state.total_tokens_used == 800
        assert abs(self.gm.state.total_cost_usd - 0.015) < 0.0001

    def test_detects_pii_in_output(self):
        result = self.gm.post_check("agent1", "Contact john@example.com for details")
        assert len(result["warnings"]) > 0
        assert "PII" in result["warnings"][0]


class TestPIIDetection:
    def test_detects_email(self):
        found = GuardrailManager.detect_pii("Email: john@example.com")
        assert "email" in found

    def test_detects_phone(self):
        found = GuardrailManager.detect_pii("Call 555-123-4567")
        assert "phone_us" in found

    def test_detects_ssn(self):
        found = GuardrailManager.detect_pii("SSN: 123-45-6789")
        assert "ssn" in found

    def test_detects_credit_card(self):
        found = GuardrailManager.detect_pii("Card: 4111-1111-1111-1111")
        assert "credit_card" in found

    def test_no_pii_in_clean_text(self):
        found = GuardrailManager.detect_pii("Tesla reported strong Q4 earnings with revenue growth.")
        assert len(found) == 0

    def test_masks_pii(self):
        masked = GuardrailManager.mask_pii("Email john@test.com or call 555-123-4567")
        assert "[EMAIL_REDACTED]" in masked
        assert "[PHONE_REDACTED]" in masked
        assert "john@test.com" not in masked


class TestBudgetStatus:
    def test_initial_status(self):
        gm = GuardrailManager()
        gm.start_pipeline()
        status = gm.get_budget_status()
        assert status["tokens_used"] == 0
        assert status["agent_calls"] == 0
        assert status["violations"] == 0

    def test_status_after_activity(self):
        gm = GuardrailManager()
        gm.start_pipeline()
        gm.pre_check("agent1")
        gm.post_check("agent1", "text", tokens_used=1000, cost_usd=0.05)

        status = gm.get_budget_status()
        assert status["tokens_used"] == 1000
        assert status["cost_usd"] == 0.05
        assert status["agent_calls"] == 1


class TestGuardrailDisable:
    def test_disabled_always_passes(self):
        gm = GuardrailManager(config={"max_agent_calls": 1, "max_reasoning_iterations": 1, "max_execution_time_seconds": 0})
        gm.start_pipeline()
        gm.disable()

        # Even with zero limits, disabled guardrails should pass
        for _ in range(10):
            result = gm.pre_check("test")
            assert result["passed"] is True

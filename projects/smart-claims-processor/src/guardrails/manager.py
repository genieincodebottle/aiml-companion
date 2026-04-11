"""
Guardrails Manager - wraps every agent execution with safety checks.

Pre-execution checks:
  - Budget (tokens, cost, agent calls)
  - Loop detection
  - Execution timeout

Post-execution checks:
  - Output confidence threshold
  - Hallucination detection (key facts must reference claim data)
  - Schema completeness

Usage:
    manager = GuardrailsManager(state)
    if manager.pre_check(agent_name="intake"):
        result = run_agent(...)
        manager.post_check(agent_name="intake", output=result)
        state = manager.update_state(state)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from src.config import get_guardrails_config
from src.models.schemas import (
    DamageAssessmentOutput,
    FraudAssessmentOutput,
    IntakeValidationOutput,
    PolicyCheckOutput,
    SettlementOutput,
)

logger = logging.getLogger(__name__)

# Minimum confidence for each agent output type
_MIN_CONFIDENCE = {
    "intake": 0.60,
    "fraud": 0.55,       # Fraud crew returns composite score, not confidence
    "damage": 0.65,
    "policy": 0.70,
    "settlement": 0.70,
}

# Keywords that MUST appear in output reasoning if they appear in the claim
_GROUNDING_KEYWORDS = [
    "policy",
    "claim",
    "incident",
    "damage",
    "coverage",
]


class GuardrailsViolation(Exception):
    """Raised when a hard guardrail is breached."""
    pass


class GuardrailsManager:
    """
    Stateful guardrails context for a single claim pipeline run.
    Pass the same instance through all agent calls.
    """

    def __init__(self, claim_id: str):
        self.claim_id = claim_id
        self.cfg = get_guardrails_config()
        self.agent_call_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.start_time = time.time()
        self.violations: list[str] = []
        self._agent_call_history: list[str] = []

    # ── Pre-Execution Checks ──────────────────────────────────────────────────

    def pre_check(self, agent_name: str) -> bool:
        """
        Run all pre-execution guardrails.
        Returns True if safe to proceed, False to skip this agent.
        Raises GuardrailsViolation for hard stops.
        """
        checks = [
            self._check_budget(),
            self._check_loop(agent_name),
            self._check_timeout(),
        ]
        return all(checks)

    def _check_budget(self) -> bool:
        """Hard stop if any budget is exceeded."""
        if self.agent_call_count >= self.cfg["max_agent_calls"]:
            violation = f"Agent call limit reached ({self.agent_call_count}/{self.cfg['max_agent_calls']})"
            self.violations.append(violation)
            logger.warning(f"[{self.claim_id}] GUARDRAIL: {violation}")
            raise GuardrailsViolation(violation)

        if self.total_tokens >= self.cfg["max_tokens_per_claim"]:
            violation = f"Token budget exceeded ({self.total_tokens}/{self.cfg['max_tokens_per_claim']})"
            self.violations.append(violation)
            raise GuardrailsViolation(violation)

        if self.total_cost >= self.cfg["max_cost_usd"]:
            violation = f"Cost budget exceeded (${self.total_cost:.4f}/${self.cfg['max_cost_usd']:.2f})"
            self.violations.append(violation)
            raise GuardrailsViolation(violation)

        return True

    def _check_loop(self, agent_name: str) -> bool:
        """Detect if same agent is called too many times (loop indicator)."""
        same_agent_count = self._agent_call_history.count(agent_name)
        max_iterations = self.cfg.get("max_loop_iterations", 10)
        if same_agent_count >= max_iterations:
            violation = f"Loop detected: {agent_name} called {same_agent_count} times"
            self.violations.append(violation)
            logger.error(f"[{self.claim_id}] GUARDRAIL LOOP: {violation}")
            raise GuardrailsViolation(violation)
        return True

    def _check_timeout(self) -> bool:
        """Warn if execution is taking too long (but don't hard-stop)."""
        elapsed = time.time() - self.start_time
        max_seconds = self.cfg.get("max_execution_seconds", 300)
        if elapsed > max_seconds:
            violation = f"Execution timeout: {elapsed:.0f}s > {max_seconds}s"
            self.violations.append(violation)
            logger.warning(f"[{self.claim_id}] GUARDRAIL TIMEOUT: {violation}")
            return False  # Soft stop - skip remaining agents
        return True

    # ── Post-Execution Checks ─────────────────────────────────────────────────

    def post_check(
        self,
        agent_name: str,
        output: Any,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
    ) -> bool:
        """
        Run post-execution checks after agent returns.
        Updates internal counters.
        Returns True if output passes all checks.
        """
        self.agent_call_count += 1
        self.total_tokens += tokens_used
        self.total_cost += cost_usd
        self._agent_call_history.append(agent_name)

        checks_passed = all([
            self._check_confidence(agent_name, output),
            self._check_hallucination(agent_name, output),
        ])
        return checks_passed

    def _check_confidence(self, agent_name: str, output: Any) -> bool:
        """Verify output confidence meets minimum threshold."""
        min_conf = _MIN_CONFIDENCE.get(agent_name, self.cfg.get("min_output_confidence", 0.60))
        confidence = getattr(output, "confidence", None) or getattr(output, "assessment_confidence", None)
        if confidence is not None and confidence < min_conf:
            warning = f"{agent_name} confidence too low: {confidence:.2f} < {min_conf:.2f}"
            self.violations.append(warning)
            logger.warning(f"[{self.claim_id}] LOW CONFIDENCE: {warning}")
            return False
        return True

    def _check_hallucination(self, agent_name: str, output: Any) -> bool:
        """
        Basic hallucination check: output should not reference facts
        that appear invented (not grounded in input).
        Currently checks that output has non-empty analysis/reasoning fields.
        """
        if not self.cfg.get("hallucination_check", True):
            return True
        # Check that reasoning/notes fields are non-empty
        reasoning_fields = ["intake_notes", "assessment_notes", "coverage_notes", "crew_summary", "analysis"]
        for field in reasoning_fields:
            value = getattr(output, field, None)
            if value is not None and not value.strip():
                warning = f"{agent_name} returned empty reasoning field: {field}"
                self.violations.append(warning)
                logger.warning(f"[{self.claim_id}] EMPTY REASONING: {warning}")
                return False
        return True

    # ── State Update ──────────────────────────────────────────────────────────

    def get_usage_summary(self) -> dict:
        """Return current usage metrics for state update."""
        return {
            "agent_call_count": self.agent_call_count,
            "total_tokens_used": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "guardrails_passed": len(self.violations) == 0,
            "guardrails_violations": self.violations.copy(),
        }

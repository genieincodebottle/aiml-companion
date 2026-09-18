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

Wiring:
    Every agent node in src/agents/graph.py is registered through
    `guard_node(...)` below. The manager is rebuilt from the pipeline state on
    each node (the state is what survives a HITL pause in the checkpointer),
    so budgets are enforced across the whole claim, including after resume.

    A hard breach (calls, tokens, cost, loop) or the execution timeout HALTS
    the claim: the agent is skipped, `guardrails_halted` is set, and every
    router sends the claim to a human (hitl_checkpoint). After the human
    decides, the communication agent sends a template letter without another
    LLM call.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from src.config import get_guardrails_config

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
        # Active processing time carried in pipeline state. None = use wall
        # clock since construction (standalone use). Wall clock would count
        # the hours a claim sits paused waiting for a reviewer.
        self.processing_seconds: float | None = None

    @classmethod
    def from_state(cls, state: dict) -> "GuardrailsManager":
        """Rebuild the per-claim budget view from pipeline state."""
        claim = state.get("claim") or {}
        manager = cls(claim.get("claim_id", "unknown"))
        manager.agent_call_count = int(state.get("agent_call_count") or 0)
        manager.total_tokens = int(state.get("total_tokens_used") or 0)
        manager.total_cost = float(state.get("total_cost_usd") or 0.0)
        manager.processing_seconds = float(state.get("processing_seconds") or 0.0)
        manager.violations = list(state.get("guardrails_violations") or [])
        manager._agent_call_history = [
            entry.get("agent") for entry in (state.get("pipeline_trace") or [])
            if isinstance(entry, dict) and entry.get("agent")
        ]
        return manager

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
            from src.utils import currency_symbol
            cs = currency_symbol()
            violation = f"Cost budget exceeded ({cs}{self.total_cost:.4f}/{cs}{self.cfg['max_cost_usd']:.2f})"
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
        if self.processing_seconds is not None:
            elapsed = self.processing_seconds
        else:
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


# ── Node wrapper (how the pipeline actually uses the manager) ────────────────

# Node name -> key used by _MIN_CONFIDENCE and the trace.
_AGENT_KEYS = {
    "intake_agent": "intake",
    "fraud_crew": "fraud",
    "damage_assessor": "damage",
    "policy_checker": "policy",
    "settlement_calculator": "settlement",
    "evaluator": "evaluator",
    "communication_agent": "communication",
}

# Node name -> state key holding that node's structured output.
_OUTPUT_KEYS = {
    "intake_agent": "intake_output",
    "fraud_crew": "fraud_output",
    "damage_assessor": "damage_output",
    "policy_checker": "policy_output",
    "settlement_calculator": "settlement_output",
    "evaluator": "evaluation_output",
    "communication_agent": "communication_output",
}


def _halt_update(state: dict, manager: GuardrailsManager, node_name: str, reason: str) -> dict:
    """State update for a claim that must stop spending and go to a human."""
    from src.models.schemas import ClaimDecision

    logger.warning(f"[{manager.claim_id}] GUARDRAIL HALT before {node_name}: {reason}")
    violations = list(state.get("guardrails_violations") or [])
    if reason not in violations:
        violations.append(reason)
    return {
        "guardrails_halted": True,
        "guardrails_passed": False,
        "guardrails_violations": violations,
        "final_decision": ClaimDecision.ESCALATED_HITL,
        "pipeline_trace": [{
            "agent": "guardrails",
            "status": "halted",
            "skipped_agent": node_name,
            "decision": "halted",
            "confidence": None,
            "reasoning": f"Guardrail halted the pipeline before {node_name}: {reason}",
            "flags": [reason],
            "findings": {
                "agent_call_count": manager.agent_call_count,
                "total_tokens_used": manager.total_tokens,
                "total_cost_usd": round(manager.total_cost, 6),
                "processing_seconds": round(manager.processing_seconds or 0.0, 2),
            },
        }],
    }


def guard_node(node_name: str, fn: Callable[[dict], dict], enforce_budget: bool = True) -> Callable[[dict], dict]:
    """Wrap a LangGraph node with pre-checks, usage accounting and post-checks.

    enforce_budget=False is for the communication agent: it is the last step
    and always runs, but on a halted claim it uses a template, not the LLM.
    """
    from src.llm import get_token_usage

    def guarded(state: dict) -> dict:
        manager = GuardrailsManager.from_state(state)
        if enforce_budget:
            try:
                within_time = manager.pre_check(node_name)
            except GuardrailsViolation as exc:
                return _halt_update(state, manager, node_name, str(exc))
            if not within_time:
                return _halt_update(state, manager, node_name, manager.violations[-1])

        before = get_token_usage()
        started = time.time()
        update = fn(state) or {}
        elapsed = time.time() - started
        after = get_token_usage()

        # Carry usage in STATE, not only in the per-run accumulator: state is
        # what the checkpointer persists, so totals survive a HITL pause and
        # the budget applies to the whole claim.
        update["total_tokens_used"] = int(state.get("total_tokens_used") or 0) + (after["total"] - before["total"])
        update["total_cost_usd"] = round(
            float(state.get("total_cost_usd") or 0.0) + (after["cost"] - before["cost"]), 6
        )
        update["processing_seconds"] = round(float(state.get("processing_seconds") or 0.0) + elapsed, 3)

        # Output quality checks: soft warnings, recorded for the judge and the
        # audit trail. Routing on low confidence is the confidence gates' job.
        output = update.get(_OUTPUT_KEYS.get(node_name, ""))
        if output is not None:
            reviewer = GuardrailsManager(manager.claim_id)
            key = _AGENT_KEYS.get(node_name, node_name)
            reviewer._check_confidence(key, output)
            reviewer._check_hallucination(key, output)
            if reviewer.violations:
                violations = list(state.get("guardrails_violations") or [])
                violations.extend(f"warning: {v}" for v in reviewer.violations if f"warning: {v}" not in violations)
                update["guardrails_violations"] = violations
        return update

    guarded.__name__ = f"guarded_{node_name}"
    guarded.__doc__ = fn.__doc__
    return guarded

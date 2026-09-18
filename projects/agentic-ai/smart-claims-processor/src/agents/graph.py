"""
LangGraph Main Workflow - Smart Claims Processor

Orchestrates all agents in a conditional, state-driven pipeline.

Workflow paths:

PATH A - Normal (low fraud):
  intake -> fraud_crew -> damage_assessor -> policy_checker
  -> settlement -> evaluator -> communication

PATH B - HITL Required (fraud score >= hitl.triggers.fraud_score):
  intake -> fraud_crew -> hitl_checkpoint -> [paused for reviewer]
    approved: -> damage_assessor -> policy_checker -> settlement -> evaluator -> communication
    denied / fraud_investigation / other: -> communication

PATH B2 - HITL Required (evaluator quality gate failed, or a high-value claim
          that needs sign-off on the proposed settlement):
  intake -> fraud_crew -> damage_assessor -> policy_checker
  -> settlement -> evaluator -> hitl_checkpoint -> [paused] -> communication

PATH C - Auto-Reject (confirmed fraud: composite score >= auto_reject_threshold,
         narrative judged fabricated, AND the claimant's own words admit intent):
  intake -> fraud_crew -> auto_reject -> communication

PATH D - Intake Failure (invalid claim):
  intake -> [invalid] -> communication (denial)

PATH E - Fast Mode (amount < pipeline.fast_mode.max_amount):
  intake -> settlement -> evaluator -> communication

Confidence gates: any agent whose confidence is below its
confidence_gates.per_agent threshold -> hitl_after_<agent> [paused] -> next agent.

Guardrail halt: a hard budget or timeout breach before any agent ->
hitl_checkpoint [paused] -> communication (template letter, no further LLM calls).

Conditional routing functions determine which path to take at each junction.
"""

from __future__ import annotations  # for Python 3.10 compatibility with forward references in type hints

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

# SqliteSaver lives in the separate `langgraph-checkpoint-sqlite` package.
# Fall back to MemorySaver if it isn't installed - the pipeline still works,
# the only thing lost is durability across restarts of a paused (HITL) claim.
try:
    from langgraph.checkpoint.sqlite import SqliteSaver   # type: ignore
    _HAS_SQLITE_SAVER = True
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver
    SqliteSaver = None  # type: ignore
    _HAS_SQLITE_SAVER = False

from src.agents.communication_agent import run_communication_agent
from src.agents.damage_assessor import run_damage_assessor
from src.agents.fraud_crew import run_fraud_crew
from src.agents.intake_agent import run_intake_agent
from src.agents.policy_checker import run_policy_checker
from src.agents.settlement_calculator import run_settlement_calculator
from src.config import (
    get_confidence_gate_config,
    get_evaluation_config,
    get_hitl_config,
    get_pipeline_config,
)
from src.evaluation.evaluator import run_evaluator
from src.guardrails.manager import guard_node
from src.hitl.checkpoint import check_hitl_required, format_hitl_brief
from src.hitl.queue import enqueue_claim
from src.models.schemas import ClaimDecision, FraudRiskLevel
from src.models.state import ClaimsState, initial_state, ClaimInput
from src.utils import currency_symbol

logger = logging.getLogger(__name__)


# ------- Confidence gates -----------------------------------------------------

def _gate_threshold(agent_key: str) -> float:
    cfg = get_confidence_gate_config()
    return float(cfg.get("per_agent", {}).get(agent_key, cfg.get("default_threshold", 0.60)))


def _check_confidence_gate(agent_key: str, confidence: float | None) -> bool:
    """Return True if confidence is below threshold (HITL needed)."""
    cfg = get_confidence_gate_config()
    if not cfg.get("enabled", False) or confidence is None:
        return False
    threshold = _gate_threshold(agent_key)
    below = confidence < threshold
    if below:
        logger.info(f"Confidence gate: {agent_key} confidence {confidence:.2f} < {threshold:.2f} -> HITL")
    return below


# Which agent each confidence-gate HITL node is guarding, and how to read that
# agent's confidence from state.
_GATE_NODES = {
    "hitl_after_intake": ("intake_agent", lambda s: getattr(s.get("intake_output"), "confidence", None)),
    "hitl_after_fraud": ("fraud_crew", lambda s: getattr(s.get("fraud_output"), "confidence", None)),
    "hitl_after_damage": ("damage_assessor", lambda s: getattr(s.get("damage_output"), "assessment_confidence", None)),
    "hitl_after_policy": ("policy_checker", lambda s: getattr(s.get("policy_output"), "confidence", None)),
    "hitl_after_settlement": ("settlement_calculator", lambda s: getattr(s.get("settlement_output"), "confidence", None)),
}


def _routing_reasons(node_name: str, state: ClaimsState) -> list[str]:
    """Why the router sent this claim to THIS HITL node.

    The router already decided a human is needed. These reasons make that
    decision explicit in the ticket, and they guarantee the node pauses: the
    generic trigger rules in check_hitl_required know nothing about a
    per-agent confidence gate, so on their own they could wave a
    low-confidence claim straight through.
    """
    reasons: list[str] = []
    if node_name in _GATE_NODES:
        agent_key, read_conf = _GATE_NODES[node_name]
        conf = read_conf(state)
        if conf is not None:
            reasons.append(
                f"Confidence gate: {agent_key} confidence {conf:.2f} below {_gate_threshold(agent_key):.2f}"
            )
        else:
            reasons.append(f"Confidence gate: {agent_key} returned no confidence")
        return reasons

    # hitl_checkpoint: guardrail halt, failed quality gate, or high fraud score
    if state.get("guardrails_halted"):
        last = (state.get("guardrails_violations") or ["budget or timeout exceeded"])[-1]
        reasons.append(f"Guardrail halt: {last}")
    if state.get("evaluation_passed") is False:
        evaluation = state.get("evaluation_output")
        score = getattr(evaluation, "overall_score", None)
        min_score = get_evaluation_config().get("min_score_to_release", 0.70)
        if score is not None:
            reasons.append(f"Evaluator quality gate failed (score {score:.2f} < {min_score:.2f})")
        else:
            reasons.append("Evaluator quality gate failed")
    triggers_cfg = get_hitl_config()["triggers"]
    amount = float(state["claim"].get("estimated_amount", 0) or 0) if state.get("claim") else 0.0
    min_amount = float(triggers_cfg.get("min_amount", 10000))
    if state.get("settlement_output") is not None and amount >= min_amount and not state.get("hitl_required"):
        reasons.append(
            f"High-value claim: {currency_symbol()}{amount:,.0f} >= {currency_symbol()}{min_amount:,.0f} "
            "needs human sign-off on the proposed settlement"
        )
    fraud = state.get("fraud_output")
    review_threshold = float(triggers_cfg.get("fraud_score", 0.45))
    if fraud and fraud.fraud_score >= review_threshold:
        reasons.append(
            f"Fraud score {fraud.fraud_score:.2f} >= review threshold {review_threshold:.2f} "
            f"(risk: {fraud.fraud_risk_level.value})"
        )
    return reasons


# ------- HITL Node -------------------------------------------------------------

def hitl_checkpoint_node(state: ClaimsState, config: RunnableConfig) -> dict:
    """
    HITL checkpoint - enqueues the claim for manual review and PAUSES the graph
    via LangGraph's `interrupt()` primitive. The pipeline does not resume until
    an approver calls `resume_claim(...)` with a decision payload.

    One function backs six graph nodes (hitl_checkpoint + five confidence
    gates); it reads its own node name from the run config to know why it was
    called.

    On resume, LangGraph re-runs this function from the first line and
    `interrupt()` returns the reviewer's decision instead of pausing. Every
    side effect before `interrupt()` therefore runs twice per pause. The
    ticket insert is idempotent (keyed on run + node + pause number), so the
    second run returns the ticket the first run created.
    """
    node_name = (config or {}).get("metadata", {}).get("langgraph_node", "hitl_checkpoint")
    claim = state["claim"]
    claim_id = claim["claim_id"]
    fraud_output = state.get("fraud_output")
    damage_output = state.get("damage_output")
    intake_output = state.get("intake_output")

    logger.info(f"[{claim_id}] HITL checkpoint triggered at {node_name}")

    agent_confidences = [
        c for c in (
            getattr(intake_output, "confidence", None),
            getattr(damage_output, "assessment_confidence", None),
            getattr(state.get("policy_output"), "confidence", None),
            getattr(state.get("settlement_output"), "confidence", None),
        ) if c is not None
    ]

    _, generic_triggers, priority, priority_score = check_hitl_required(
        claim=dict(claim),
        intake_output=intake_output,
        fraud_output=fraud_output,
        damage_assessed_usd=damage_output.assessed_damage_usd if damage_output else 0,
        agent_confidence_scores=agent_confidences,
    )
    routing_reasons = _routing_reasons(node_name, state)
    # Drop generic triggers that restate a routing reason (both rules report
    # the fraud score, for example); keep the routing reason's wording.
    reason_heads = {" ".join(r.split()[:2]) for r in routing_reasons}
    triggers = routing_reasons + [
        t for t in generic_triggers if " ".join(t.split()[:2]) not in reason_heads
    ]
    if not triggers:
        triggers = [f"Routed to human review by the pipeline at {node_name}"]

    review_brief = format_hitl_brief(
        claim=dict(state.get("masked_claim") or claim),
        triggers=triggers,
        priority=priority,
        fraud_output=fraud_output,
        damage_assessed_usd=damage_output.assessed_damage_usd if damage_output else 0,
    )

    state_snapshot = {
        "claim_id": claim_id,
        "incident_type": claim.get("incident_type"),
        "estimated_amount": claim.get("estimated_amount"),
        "fraud_score": fraud_output.fraud_score if fraud_output else 0,
        "fraud_risk": fraud_output.fraud_risk_level.value if fraud_output else "unknown",
        "assessed_damage": damage_output.assessed_damage_usd if damage_output else 0,
        "ai_settlement": state.get("final_amount_usd", 0),
        "ai_decision": state.get("final_decision").value if state.get("final_decision") else "pending",
        "paused_at": node_name,
        "guardrails_halted": bool(state.get("guardrails_halted")),
    }

    prior_pauses_here = sum(
        1 for entry in (state.get("pipeline_trace") or [])
        if isinstance(entry, dict) and entry.get("agent") == "hitl_checkpoint" and entry.get("node") == node_name
    )
    idempotency_key = f"{claim_id}:{state.get('execution_start_time')}:{node_name}:{prior_pauses_here}"

    ticket_id = enqueue_claim(
        claim_id=claim_id,
        priority=priority,
        priority_score=priority_score,
        triggers=triggers,
        review_brief=review_brief,
        state_snapshot=state_snapshot,
        idempotency_key=idempotency_key,
    )

    logger.info(f"[{claim_id}] Pausing pipeline for manual approval (ticket={ticket_id})")

    # Hard pause. Execution suspends here until an approver supplies a decision
    # via Command(resume={...}) from the /api/hitl/decide endpoint.
    human_result: dict = interrupt({
        "ticket_id": ticket_id,
        "claim_id": claim_id,
        "node": node_name,
        "priority": priority.value,
        "priority_score": priority_score,
        "triggers": triggers,
        "review_brief": review_brief,
        "state_snapshot": state_snapshot,
    })

    decision_str = (human_result or {}).get("decision") or ClaimDecision.ESCALATED_HITL.value
    try:
        decision_enum = ClaimDecision(decision_str)
    except ValueError:
        decision_enum = ClaimDecision.ESCALATED_HITL

    override_amount = (human_result or {}).get("settlement_override_usd")
    result_update = {
        "hitl_required": True,
        "hitl_triggers": triggers,
        "hitl_priority": priority,
        "hitl_priority_score": priority_score,
        "hitl_ticket_id": ticket_id,
        "human_decision": decision_str,
        "human_reviewer_id": (human_result or {}).get("reviewer_id"),
        "human_notes": (human_result or {}).get("notes", ""),
        "human_override": bool((human_result or {}).get("override_ai", False)),
        "final_decision": decision_enum,
        "pipeline_trace": [{
            "agent": "hitl_checkpoint",
            "node": node_name,
            "ticket_id": ticket_id,
            "priority": priority.value,
            "priority_score": priority_score,
            "triggers": triggers,
            "human_decision": decision_str,
            "reviewer_id": (human_result or {}).get("reviewer_id"),
            "decision": decision_str,
            "confidence": None,
            "reasoning": (human_result or {}).get("notes", ""),
            "flags": triggers,
            "findings": {"paused_at": node_name, "priority": priority.value},
        }],
    }
    if override_amount is not None:
        # Kept in its own field as well: if the pipeline continues after this
        # pause, the settlement agent would otherwise overwrite final_amount_usd.
        result_update["final_amount_usd"] = float(override_amount)
        result_update["human_settlement_override_usd"] = float(override_amount)
    logger.info(f"[{claim_id}] Resumed from HITL ({node_name}) with decision={decision_str}")
    return result_update


def auto_reject_node(state: ClaimsState) -> dict:
    """Auto-reject path for confirmed fraud (rules and narrative agree)."""
    claim_id = state["claim"]["claim_id"]
    fraud = state.get("fraud_output")
    logger.warning(f"[{claim_id}] AUTO-REJECT: Confirmed fraud (score={fraud.fraud_score:.2f})")
    return {
        "final_decision": ClaimDecision.AUTO_REJECTED,
        "final_amount_usd": 0.0,
        "pipeline_trace": [{
            "agent": "auto_reject",
            "fraud_score": fraud.fraud_score if fraud else 0,
            "narrative_risk": fraud.narrative_risk if fraud else None,
            "reason": "Confirmed fraud - auto rejected",
            "decision": ClaimDecision.AUTO_REJECTED.value,
            "confidence": fraud.confidence if fraud else None,
            "reasoning": "Composite fraud score and narrative risk crossed the auto-reject line, "
                         "and the claimant's own description admits an intent to deceive.",
            "flags": fraud.primary_concerns if fraud else [],
            "findings": {},
        }],
    }


# ------- Routing Functions ----------------------------------------------------

def route_after_intake(state: ClaimsState) -> Literal[
    "fraud_crew", "communication_agent", "settlement_calculator", "hitl_after_intake", "hitl_checkpoint"
]:
    """Route after intake: valid claims proceed, invalid ones go straight to communication."""
    if state.get("guardrails_halted"):
        return "hitl_checkpoint"

    intake = state.get("intake_output")

    if not intake or not intake.is_valid:
        logger.info("Routing to denial: intake invalid")
        return "communication_agent"

    # Confidence gate
    if _check_confidence_gate("intake_agent", intake.confidence):
        return "hitl_after_intake"

    # Fast mode: small claims skip fraud + damage
    pipeline_cfg = get_pipeline_config()
    fast_mode = pipeline_cfg.get("fast_mode", {})
    if (
        fast_mode.get("enabled", False)
        and float(state["claim"].get("estimated_amount", 0)) < fast_mode.get("max_amount", 500)
    ):
        logger.info("Fast mode: routing directly to settlement")
        return "settlement_calculator"

    return "fraud_crew"


def route_after_fraud(state: ClaimsState) -> Literal[
    "damage_assessor", "auto_reject", "hitl_checkpoint", "hitl_after_fraud"
]:
    """Route after fraud assessment.

    Order matters:
      1. CONFIRMED -> auto_reject. The crew only returns CONFIRMED when the
         composite score is >= agents.fraud_crew.auto_reject_threshold, the
         narrative risk is >= auto_reject_min_narrative_risk, and the
         validator reports an explicit admission. See
         fraud_crew.score_fraud_signals.
      2. Composite >= hitl.triggers.fraud_score -> a human reviews.
      3. The crew doesn't trust its own assessment (e.g. no readable verdict
         from the consistency validator) -> confidence-gate HITL.
    """
    if state.get("guardrails_halted"):
        return "hitl_checkpoint"

    fraud = state.get("fraud_output")
    if not fraud:
        return "damage_assessor"

    if fraud.fraud_risk_level == FraudRiskLevel.CONFIRMED:
        return "auto_reject"

    review_threshold = float(get_hitl_config()["triggers"].get("fraud_score", 0.45))
    if fraud.fraud_score >= review_threshold:
        return "hitl_checkpoint"

    if _check_confidence_gate("fraud_crew", fraud.confidence):
        return "hitl_after_fraud"

    return "damage_assessor"


def route_after_damage(state: ClaimsState) -> Literal[
    "policy_checker", "hitl_after_damage", "hitl_checkpoint"
]:
    """Confidence gate after damage assessment."""
    if state.get("guardrails_halted"):
        return "hitl_checkpoint"
    damage = state.get("damage_output")
    if damage and _check_confidence_gate("damage_assessor", damage.assessment_confidence):
        return "hitl_after_damage"
    return "policy_checker"


def route_after_policy(state: ClaimsState) -> Literal[
    "settlement_calculator", "hitl_after_policy", "hitl_checkpoint"
]:
    """Confidence gate after policy check."""
    if state.get("guardrails_halted"):
        return "hitl_checkpoint"
    policy = state.get("policy_output")
    if policy and _check_confidence_gate("policy_checker", policy.confidence):
        return "hitl_after_policy"
    return "settlement_calculator"


def route_after_settlement(state: ClaimsState) -> Literal[
    "evaluator", "hitl_after_settlement", "hitl_checkpoint"
]:
    """Confidence gate after settlement calculation."""
    if state.get("guardrails_halted"):
        return "hitl_checkpoint"
    settlement = state.get("settlement_output")
    if settlement and _check_confidence_gate("settlement_calculator", settlement.confidence):
        return "hitl_after_settlement"
    return "evaluator"


def route_after_evaluation(state: ClaimsState) -> Literal[
    "hitl_checkpoint", "communication_agent"
]:
    """If evaluation FAILED the quality gate, route to HITL before release.

    Three states, not two: True (judged, passed), False (judged, failed), None
    (never judged -- the sampler skipped it). Only an explicit False diverts to
    a human; an unevaluated claim continues, which is what sampling is for.
    Writing this as `if not evaluation_passed` would send every skipped claim to
    HITL and defeat the sampling entirely.
    """
    if state.get("guardrails_halted"):
        return "hitl_checkpoint"
    evaluation_passed = state.get("evaluation_passed")
    if evaluation_passed is False:
        logger.info("Evaluation failed quality gate - routing to HITL")
        return "hitl_checkpoint"
    if _high_value_needs_signoff(state):
        logger.info("High-value claim - routing to HITL for settlement sign-off")
        return "hitl_checkpoint"
    if evaluation_passed is None:
        logger.info("Claim was not evaluated (sampled out) - continuing unjudged")
    return "communication_agent"


_CONTINUE_DECISIONS = {ClaimDecision.APPROVED.value, ClaimDecision.APPROVED_PARTIAL.value}


def _high_value_needs_signoff(state: ClaimsState) -> bool:
    """High-value claims (hitl.triggers.min_amount, per country) need a human
    to sign off on the proposed settlement, unless a human already reviewed
    this claim earlier in the run."""
    if state.get("hitl_required"):
        return False
    min_amount = float(get_hitl_config()["triggers"].get("min_amount", 10000))
    return float((state.get("claim") or {}).get("estimated_amount", 0) or 0) >= min_amount


def route_after_hitl_checkpoint(state: ClaimsState) -> Literal[
    "damage_assessor", "communication_agent"
]:
    """Route after hitl_checkpoint resume.

    - Guardrail halt: the human's decision stands; no more agent spend.
    - Reviewer did not approve (denied, fraud investigation, pending documents,
      escalated): nothing left to calculate, write the letter.
    - Fraud-triggered pause that the reviewer approved: damage hasn't run yet,
      so continue the pipeline to compute the settlement.
    - Eval-triggered pause: everything already ran, go to communication.
    """
    if state.get("guardrails_halted"):
        logger.info("HITL resume after guardrail halt -> communication_agent")
        return "communication_agent"
    decision = state.get("human_decision")
    if decision is not None and decision not in _CONTINUE_DECISIONS:
        logger.info(f"HITL resume: reviewer decided {decision} -> communication_agent")
        return "communication_agent"
    if state.get("damage_output") is None:
        logger.info("HITL resume: fraud-triggered, continuing to damage_assessor")
        return "damage_assessor"
    logger.info("HITL resume: eval-triggered, continuing to communication_agent")
    return "communication_agent"


# ------- Checkpointer (shared across invocations so interrupt/resume works) ---

_CHECKPOINT_DB = str(Path(__file__).resolve().parent.parent / "data" / "claims_checkpoints.db")
_checkpointer_ctx = None   # must stay alive so the SQLite connection isn't garbage-collected
_checkpointer = None
_compiled_graph = None


def _get_checkpointer():
    """Lazy, process-lifetime checkpointer so interrupted claims can resume.

    Prefers SqliteSaver (durable across restarts). Falls back to MemorySaver
    if langgraph-checkpoint-sqlite isn't installed, so the pipeline still runs.
    """
    global _checkpointer, _checkpointer_ctx
    if _checkpointer is not None:
        return _checkpointer

    if not _HAS_SQLITE_SAVER:
        logger.warning(
            "langgraph-checkpoint-sqlite not installed - falling back to "
            "MemorySaver. Paused (HITL) claims will NOT survive a server "
            "restart. Run: uv pip install langgraph-checkpoint-sqlite"
        )
        _checkpointer = MemorySaver()
        return _checkpointer

    Path(_CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)
    _checkpointer_ctx = SqliteSaver.from_conn_string(_CHECKPOINT_DB)
    _checkpointer = _checkpointer_ctx.__enter__()
    return _checkpointer


def get_compiled_graph():
    """Return the process-wide compiled graph (built once, cached)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph_object().compile(checkpointer=_get_checkpointer())
    return _compiled_graph


def _build_graph_object() -> StateGraph:
    """
    Build and return the LangGraph StateGraph (compiled by get_compiled_graph).

    Every agent node runs inside guard_node(), which enforces the per-claim
    budget before the agent and records tokens, cost and time after it.

    START
      └─ intake_agent
           ├─ [invalid] -------------------------------------------------> communication_agent
           ├─ [low_confidence] ------------------------------------------> hitl_after_intake -> fraud_crew
           ├─ [fast_mode] -----------------------------------------------> settlement_calculator
           └─ fraud_crew
                ├─ [confirmed_fraud] ------------------------------------> auto_reject -> communication_agent
                ├─ [high_fraud] -----------------------------------------> hitl_checkpoint
                ├─ [low_confidence] -------------------------------------> hitl_after_fraud -> damage_assessor
                └─ damage_assessor
                     ├─ [low_confidence] --------------------------------> hitl_after_damage -> policy_checker
                     └─ policy_checker
                          ├─ [low_confidence] ---------------------------> hitl_after_policy -> settlement_calculator
                          └─ settlement_calculator
                               ├─ [low_confidence] ---------------------> hitl_after_settlement -> evaluator
                               └─ evaluator
                                    ├─ [passed / not sampled] -------------> communication_agent
                                    └─ [failed] ----------------------------> hitl_checkpoint

    hitl_checkpoint resumes to damage_assessor (approved fraud review) or
    communication_agent (everything else). Any router sends a guardrail-halted
    claim to hitl_checkpoint.
    """
    graph = StateGraph(ClaimsState)

    # ------- Core agent nodes (each wrapped in the guardrails) ---------------------
    graph.add_node("intake_agent", guard_node("intake_agent", run_intake_agent))
    graph.add_node("fraud_crew", guard_node("fraud_crew", run_fraud_crew))
    graph.add_node("damage_assessor", guard_node("damage_assessor", run_damage_assessor))
    graph.add_node("policy_checker", guard_node("policy_checker", run_policy_checker))
    graph.add_node("settlement_calculator", guard_node("settlement_calculator", run_settlement_calculator))
    graph.add_node("evaluator", guard_node("evaluator", run_evaluator))
    graph.add_node("auto_reject", auto_reject_node)
    graph.add_node(
        "communication_agent",
        guard_node("communication_agent", run_communication_agent, enforce_budget=False),
    )

    # ------- HITL nodes (same function, different node names -> different resume targets)
    graph.add_node("hitl_checkpoint", hitl_checkpoint_node)       # fraud / eval / guardrail halt
    graph.add_node("hitl_after_intake", hitl_checkpoint_node)     # intake low confidence
    graph.add_node("hitl_after_fraud", hitl_checkpoint_node)      # fraud crew low confidence
    graph.add_node("hitl_after_damage", hitl_checkpoint_node)     # damage low confidence
    graph.add_node("hitl_after_policy", hitl_checkpoint_node)     # policy low confidence
    graph.add_node("hitl_after_settlement", hitl_checkpoint_node) # settlement low confidence

    # ------- Entry ------------------------------------------------------------------
    graph.add_edge(START, "intake_agent")

    graph.add_conditional_edges("intake_agent", route_after_intake, {
        "fraud_crew": "fraud_crew",
        "communication_agent": "communication_agent",
        "settlement_calculator": "settlement_calculator",
        "hitl_after_intake": "hitl_after_intake",
        "hitl_checkpoint": "hitl_checkpoint",
    })

    graph.add_conditional_edges("fraud_crew", route_after_fraud, {
        "damage_assessor": "damage_assessor",
        "auto_reject": "auto_reject",
        "hitl_checkpoint": "hitl_checkpoint",
        "hitl_after_fraud": "hitl_after_fraud",
    })

    graph.add_conditional_edges("damage_assessor", route_after_damage, {
        "policy_checker": "policy_checker",
        "hitl_after_damage": "hitl_after_damage",
        "hitl_checkpoint": "hitl_checkpoint",
    })

    graph.add_conditional_edges("policy_checker", route_after_policy, {
        "settlement_calculator": "settlement_calculator",
        "hitl_after_policy": "hitl_after_policy",
        "hitl_checkpoint": "hitl_checkpoint",
    })

    graph.add_conditional_edges("settlement_calculator", route_after_settlement, {
        "evaluator": "evaluator",
        "hitl_after_settlement": "hitl_after_settlement",
        "hitl_checkpoint": "hitl_checkpoint",
    })

    graph.add_conditional_edges("evaluator", route_after_evaluation, {
        "hitl_checkpoint": "hitl_checkpoint",
        "communication_agent": "communication_agent",
    })

    # ------- HITL resume targets: each gate resumes to the correct NEXT agent -------
    graph.add_edge("hitl_after_intake", "fraud_crew")
    graph.add_edge("hitl_after_fraud", "damage_assessor")
    graph.add_edge("hitl_after_damage", "policy_checker")
    graph.add_edge("hitl_after_policy", "settlement_calculator")
    graph.add_edge("hitl_after_settlement", "evaluator")
    graph.add_conditional_edges("hitl_checkpoint", route_after_hitl_checkpoint, {
        "damage_assessor": "damage_assessor",
        "communication_agent": "communication_agent",
    })

    # ------- Terminal edges ---------------------------------------------------------
    graph.add_edge("auto_reject", "communication_agent")
    graph.add_edge("communication_agent", END)
    return graph


# Backwards-compat alias (other code/tests may still import this).
def build_claims_graph():
    return get_compiled_graph()


# ------- Public Entry Points --------------------------------------------------

def _thread_config(claim_id: str) -> dict:
    return {"configurable": {"thread_id": claim_id}}


def _is_paused(final_state: dict) -> bool:
    """A paused graph surfaces __interrupt__ in the last state."""
    return bool(final_state.get("__interrupt__"))


def _interrupt_payload(final_state: dict) -> dict:
    interrupts = final_state.get("__interrupt__") or []
    return interrupts[0].value if interrupts else {}


def _store_to_memory(final_state: dict) -> None:
    """After pipeline completion, persist the claim to long-term + episodic memory.

    This is what makes the system learn from its own decisions over time.
    Future claims retrieve these as "similar past claims" via the memory
    tools, giving agents historical context for better decisions.

    Everything written here is PII-masked first. Memory outlives the claim and
    is read back into future prompts, so an unmasked description would leak
    one claimant's details into another claimant's LLM call.
    """
    try:
        from src.memory.manager import memory
        from src.security.pii_masker import mask_claim, mask_text

        claim = final_state.get("claim", {})
        claim_id = claim.get("claim_id", "unknown")
        masked = final_state.get("masked_claim") or mask_claim(dict(claim))
        description = masked.get("incident_description", "")
        decision = final_state.get("final_decision")
        decision_str = decision.value if hasattr(decision, "value") else str(decision or "unknown")

        fraud_output = final_state.get("fraud_output")
        fraud_score = float(getattr(fraud_output, "fraud_score", 0)) if fraud_output else None

        # Long-term memory - store every completed claim
        memory.store_claim_outcome(
            claim_id=claim_id,
            description=description,
            metadata={
                "incident_type": claim.get("incident_type", ""),
                "estimated_amount": claim.get("estimated_amount", 0),
                "decision": decision_str,
                "settlement_amount": final_state.get("final_amount_usd", 0),
                "fraud_score": fraud_score,
                "hitl_required": final_state.get("hitl_required", False),
                "human_override": final_state.get("human_override", False),
            },
        )

        # Episodic memory - store notable events
        if final_state.get("human_override"):
            memory.store_episode(
                claim_id=claim_id,
                narrative=(
                    f"Claim {claim_id} ({claim.get('incident_type', '?')}): "
                    f"AI recommended {decision_str} but reviewer overrode to "
                    f"{final_state.get('human_decision', '?')}. "
                    f"Notes: {mask_text(final_state.get('human_notes') or 'none')}"
                ),
                event_type="human_override",
                metadata={"fraud_score": fraud_score, "reviewer": final_state.get("human_reviewer_id")},
            )

        if decision_str == ClaimDecision.AUTO_REJECTED.value:
            memory.store_episode(
                claim_id=claim_id,
                narrative=(
                    f"Claim {claim_id} auto-rejected. Fraud score: {(fraud_score or 0):.2f}. "
                    f"Description: {description[:200]}"
                ),
                event_type="fraud_confirmed",
                metadata={"fraud_score": fraud_score},
            )

        eval_output = final_state.get("evaluation_output")
        if eval_output and not getattr(eval_output, "passed", True):
            memory.store_episode(
                claim_id=claim_id,
                narrative=(
                    f"Claim {claim_id} failed quality gate. Eval score: "
                    f"{getattr(eval_output, 'overall_score', '?')}. "
                    f"Routed to HITL for human review."
                ),
                event_type="quality_gate_failed",
                metadata={"eval_score": getattr(eval_output, "overall_score", None)},
            )

        logger.debug("[%s] Stored in long-term + episodic memory", claim_id)
    except Exception as e:
        # Memory storage should never crash the pipeline
        logger.warning("Memory storage skipped: %s", e)


def process_claim(claim_input: ClaimInput) -> dict:
    """
    Run the pipeline. Returns one of:
      { "paused": True,  "state": {...}, "interrupt": {...} }   # awaiting approver
      { "paused": False, "state": {...} }                        # completed
    Callers MUST check "paused" and, if True, persist status=pending_human_review.

    Token, cost and processing-time totals are carried in the graph state by
    the guardrails wrapper, so they are correct for the whole claim even
    across a HITL pause.
    """
    from src.llm import reset_token_tracking, get_token_usage

    graph = get_compiled_graph()
    claim_id = claim_input["claim_id"]
    state = initial_state(claim_input)
    state["execution_start_time"] = datetime.now(timezone.utc).isoformat()
    reset_token_tracking()

    logger.info(f"[{claim_id}] Pipeline starting")
    cfg = _thread_config(claim_id)

    try:
        final_state = graph.invoke(state, config=cfg)
    except Exception as e:
        logger.error(f"[{claim_id}] Pipeline crashed: {e}", exc_info=True)
        state["error_log"] = [f"Pipeline crash: {str(e)}"]
        state["final_decision"] = ClaimDecision.ESCALATED_HITL
        usage = get_token_usage()
        state["total_tokens_used"] = usage["total"]
        state["total_cost_usd"] = usage["cost"]
        return {"paused": False, "state": state}

    if _is_paused(final_state):
        logger.info(f"[{claim_id}] Pipeline paused at HITL checkpoint")
        return {"paused": True, "state": final_state, "interrupt": _interrupt_payload(final_state)}

    cs = currency_symbol()
    logger.info(
        f"[{claim_id}] Pipeline complete: decision={final_state.get('final_decision')}, "
        f"amount={cs}{(final_state.get('final_amount_usd') or 0):,.2f}, "
        f"tokens={final_state.get('total_tokens_used', 0)}, "
        f"cost=${(final_state.get('total_cost_usd') or 0):.6f}"
    )
    _store_to_memory(final_state)
    return {"paused": False, "state": final_state}


def resume_claim(claim_id: str, decision: dict) -> dict:
    """
    Resume a paused pipeline with an approver's decision.

    decision = {
        "decision": "approved" | "denied" | "escalated_human_review" | ...,
        "reviewer_id": str,
        "notes": str,
        "override_ai": bool,
        "settlement_override_usd": float | None,
    }
    """
    from src.llm import reset_token_tracking

    graph = get_compiled_graph()
    cfg = _thread_config(claim_id)
    reset_token_tracking()
    logger.info(f"[{claim_id}] Resuming pipeline with decision={decision.get('decision')}")
    final_state = graph.invoke(Command(resume=decision), config=cfg)

    if _is_paused(final_state):
        # Legitimate: e.g. fraud HITL approved, then the evaluator quality gate
        # fails -> a second pause on the same thread, with a NEW ticket.
        logger.info(f"[{claim_id}] Pipeline paused again after resume (e.g. eval quality gate)")
        return {"paused": True, "state": final_state, "interrupt": _interrupt_payload(final_state)}

    logger.info(f"[{claim_id}] Pipeline completed after human approval")
    _store_to_memory(final_state)
    return {"paused": False, "state": final_state}

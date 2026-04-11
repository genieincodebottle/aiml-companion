"""
LangGraph Main Workflow - Smart Claims Processor

Orchestrates all agents in a conditional, state-driven pipeline.

Workflow paths:

PATH A - Normal (low fraud, low value):
  intake → fraud_crew → damage_assessor → policy_checker
  → settlement → evaluator → [evaluator_passed? → communication] | [failed → hitl]

PATH B - HITL Required (high fraud/value/low confidence):
  intake → fraud_crew → damage_assessor → policy_checker
  → settlement → evaluator → hitl_checkpoint → [wait for human] → communication

PATH C - Auto-Reject (confirmed fraud, score >= 0.90):
  intake → fraud_crew → auto_reject → communication

PATH D - Intake Failure (invalid claim):
  intake → [invalid] → communication (denial)

PATH E - Fast Mode (amount < $500, clean history):
  intake → settlement → communication

Conditional routing functions determine which path to take at each junction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.communication_agent import run_communication_agent
from src.agents.damage_assessor import run_damage_assessor
from src.agents.fraud_crew import run_fraud_crew
from src.agents.intake_agent import run_intake_agent
from src.agents.policy_checker import run_policy_checker
from src.agents.settlement_calculator import run_settlement_calculator
from src.config import get_hitl_config, get_pipeline_config
from src.evaluation.evaluator import run_evaluator
from src.guardrails.manager import GuardrailsManager, GuardrailsViolation
from src.hitl.checkpoint import check_hitl_required, format_hitl_brief
from src.hitl.queue import enqueue_claim, get_human_decision
from src.models.schemas import ClaimDecision, FraudRiskLevel, HITLPriority
from src.models.state import ClaimsState, initial_state, ClaimInput
from src.security.audit_log import log_agent_action

logger = logging.getLogger(__name__)


# ── Wrapped Node Functions (with guardrails) ──────────────────────────────────

def _wrap_with_guardrails(agent_fn, agent_name: str):
    """Higher-order function: wraps an agent with pre/post guardrail checks."""
    def wrapped(state: ClaimsState) -> dict:
        manager = state.get("_guardrails_manager")
        if manager is None:
            return agent_fn(state)

        try:
            can_proceed = manager.pre_check(agent_name)
            if not can_proceed:
                logger.warning(f"Guardrail soft-stop for {agent_name}")
                return {
                    "guardrails_violations": manager.violations,
                    "pipeline_trace": [{"agent": agent_name, "status": "guardrail_skipped"}],
                }
        except GuardrailsViolation as e:
            return {
                "guardrails_passed": False,
                "guardrails_violations": manager.violations,
                "error_log": [f"GUARDRAIL HARD STOP ({agent_name}): {str(e)}"],
                "pipeline_trace": [{"agent": agent_name, "status": "guardrail_hard_stop", "error": str(e)}],
            }

        result = agent_fn(state)
        manager.post_check(agent_name=agent_name, output=state.get(f"{agent_name}_output"))

        usage = manager.get_usage_summary()
        result.update({
            "agent_call_count": usage["agent_call_count"],
            "total_tokens_used": usage["total_tokens_used"],
            "total_cost_usd": usage["total_cost_usd"],
            "guardrails_passed": usage["guardrails_passed"],
            "guardrails_violations": usage["guardrails_violations"],
        })
        return result

    return wrapped


# ── HITL Node ─────────────────────────────────────────────────────────────────

def hitl_checkpoint_node(state: ClaimsState) -> dict:
    """
    HITL checkpoint node. Enqueues claim for human review.
    In a real deployment, this would use LangGraph's interrupt() to pause
    execution until the human submits their decision via the FastAPI endpoint.
    For learning purposes, we demonstrate the pattern and check for a decision.
    """
    claim = state["claim"]
    claim_id = claim["claim_id"]
    fraud_output = state.get("fraud_output")
    damage_output = state.get("damage_output")
    intake_output = state.get("intake_output")

    logger.info(f"[{claim_id}] HITL checkpoint triggered")

    # Determine if HITL is needed (re-evaluate with full context now available)
    agent_confidences = []
    if intake_output and intake_output.confidence:
        agent_confidences.append(intake_output.confidence)
    if damage_output and damage_output.assessment_confidence:
        agent_confidences.append(damage_output.assessment_confidence)

    requires_hitl, triggers, priority, priority_score = check_hitl_required(
        claim=dict(claim),
        intake_output=intake_output,
        fraud_output=fraud_output,
        damage_assessed_usd=damage_output.assessed_damage_usd if damage_output else 0,
        agent_confidence_scores=agent_confidences,
    )

    if not requires_hitl:
        return {
            "hitl_required": False,
            "pipeline_trace": [{"agent": "hitl_checkpoint", "status": "not_required"}],
        }

    review_brief = format_hitl_brief(
        claim=dict(claim),
        triggers=triggers,
        priority=priority,
        fraud_output=fraud_output,
        damage_assessed_usd=damage_output.assessed_damage_usd if damage_output else 0,
    )

    # Build a safe snapshot (no PII) for the reviewer
    state_snapshot = {
        "claim_id": claim_id,
        "incident_type": claim.get("incident_type"),
        "estimated_amount": claim.get("estimated_amount"),
        "fraud_score": fraud_output.fraud_score if fraud_output else 0,
        "fraud_risk": fraud_output.fraud_risk_level.value if fraud_output else "unknown",
        "assessed_damage": damage_output.assessed_damage_usd if damage_output else 0,
        "ai_settlement": state.get("final_amount_usd", 0),
        "ai_decision": state.get("final_decision").value if state.get("final_decision") else "pending",
    }

    ticket_id = enqueue_claim(
        claim_id=claim_id,
        priority=priority,
        priority_score=priority_score,
        triggers=triggers,
        review_brief=review_brief,
        state_snapshot=state_snapshot,
    )

    # Poll for human decision (non-blocking - returns None if pending)
    # In production: use LangGraph interrupt() + resume pattern
    human_result = get_human_decision(ticket_id)

    result = {
        "hitl_required": True,
        "hitl_triggers": triggers,
        "hitl_priority": priority,
        "hitl_priority_score": priority_score,
        "hitl_ticket_id": ticket_id,
        "pipeline_trace": [{
            "agent": "hitl_checkpoint",
            "ticket_id": ticket_id,
            "priority": priority.value,
            "priority_score": priority_score,
            "triggers": triggers,
        }],
    }

    if human_result:
        result.update({
            "human_decision": human_result["decision"],
            "human_reviewer_id": human_result["reviewer_id"],
            "human_notes": human_result["notes"],
            "human_override": human_result["override_ai"],
            "final_decision": ClaimDecision(human_result["decision"]),
        })
        logger.info(f"[{claim_id}] Human decision received: {human_result['decision']}")
    else:
        # No human decision yet - mark as escalated
        result["final_decision"] = ClaimDecision.ESCALATED_HITL
        logger.info(f"[{claim_id}] Awaiting human review (ticket: {ticket_id})")

    return result


def auto_reject_node(state: ClaimsState) -> dict:
    """Auto-reject path for confirmed fraud (score >= 0.90)."""
    claim_id = state["claim"]["claim_id"]
    fraud = state.get("fraud_output")
    logger.warning(f"[{claim_id}] AUTO-REJECT: Confirmed fraud (score={fraud.fraud_score:.2f})")
    return {
        "final_decision": ClaimDecision.AUTO_REJECTED,
        "final_amount_usd": 0.0,
        "pipeline_trace": [{
            "agent": "auto_reject",
            "fraud_score": fraud.fraud_score if fraud else 0,
            "reason": "Confirmed fraud - auto rejected",
        }],
    }


# ── Routing Functions ─────────────────────────────────────────────────────────

def route_after_intake(state: ClaimsState) -> Literal[
    "fraud_crew", "communication_agent", "settlement_calculator"
]:
    """Route after intake: valid claims proceed, invalid ones go straight to communication."""
    intake = state.get("intake_output")

    if not intake or not intake.is_valid:
        logger.info(f"Routing to denial: intake invalid")
        return "communication_agent"

    # Fast mode: tiny claims with clean history skip fraud + damage
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
    "damage_assessor", "auto_reject", "hitl_checkpoint"
]:
    """Route after fraud assessment."""
    fraud = state.get("fraud_output")
    if not fraud:
        return "damage_assessor"

    cfg = get_hitl_config()
    auto_reject_threshold = cfg.get("triggers", {}).get("fraud_score", 0.65)

    # Auto-reject: confirmed fraud with very high confidence
    if fraud.fraud_score >= 0.90 and fraud.fraud_risk_level == FraudRiskLevel.CONFIRMED:
        return "auto_reject"

    # HITL: high fraud score but not auto-reject
    if fraud.fraud_score >= auto_reject_threshold:
        return "hitl_checkpoint"

    return "damage_assessor"


def route_after_evaluation(state: ClaimsState) -> Literal[
    "hitl_checkpoint", "communication_agent"
]:
    """If evaluation fails quality gate, route to HITL before release."""
    evaluation_passed = state.get("evaluation_passed", True)
    if not evaluation_passed:
        logger.info("Evaluation failed quality gate - routing to HITL")
        return "hitl_checkpoint"
    return "communication_agent"


def route_after_hitl(state: ClaimsState) -> Literal["communication_agent"]:
    """After HITL (always proceeds to communication)."""
    return "communication_agent"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_claims_graph() -> StateGraph:
    """
    Build and return the compiled LangGraph StateGraph.

    Graph structure:
    START
      └─ intake_agent
           ├─ [invalid] ──────────────────────────────────► communication_agent
           ├─ [fast_mode] ───────────────────────────────► settlement_calculator
           └─ fraud_crew
                ├─ [confirmed_fraud] ─────────────────────► auto_reject
                │                                               └─► communication_agent
                ├─ [high_fraud] ──────────────────────────► hitl_checkpoint
                │                                               └─► communication_agent
                └─ damage_assessor
                     └─ policy_checker
                          └─ settlement_calculator
                               └─ evaluator
                                    ├─ [passed] ──────────► communication_agent
                                    └─ [failed] ──────────► hitl_checkpoint
                                                                └─► communication_agent
    """
    graph = StateGraph(ClaimsState)

    # Add all nodes
    graph.add_node("intake_agent", run_intake_agent)
    graph.add_node("fraud_crew", run_fraud_crew)
    graph.add_node("damage_assessor", run_damage_assessor)
    graph.add_node("policy_checker", run_policy_checker)
    graph.add_node("settlement_calculator", run_settlement_calculator)
    graph.add_node("evaluator", run_evaluator)
    graph.add_node("hitl_checkpoint", hitl_checkpoint_node)
    graph.add_node("auto_reject", auto_reject_node)
    graph.add_node("communication_agent", run_communication_agent)

    # Entry point
    graph.add_edge(START, "intake_agent")

    # Conditional routing after intake
    graph.add_conditional_edges(
        "intake_agent",
        route_after_intake,
        {
            "fraud_crew": "fraud_crew",
            "communication_agent": "communication_agent",
            "settlement_calculator": "settlement_calculator",
        },
    )

    # Conditional routing after fraud
    graph.add_conditional_edges(
        "fraud_crew",
        route_after_fraud,
        {
            "damage_assessor": "damage_assessor",
            "auto_reject": "auto_reject",
            "hitl_checkpoint": "hitl_checkpoint",
        },
    )

    # Sequential edges
    graph.add_edge("damage_assessor", "policy_checker")
    graph.add_edge("policy_checker", "settlement_calculator")
    graph.add_edge("settlement_calculator", "evaluator")

    # Conditional routing after evaluation
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "hitl_checkpoint": "hitl_checkpoint",
            "communication_agent": "communication_agent",
        },
    )

    # HITL and auto-reject always go to communication
    graph.add_edge("hitl_checkpoint", "communication_agent")
    graph.add_edge("auto_reject", "communication_agent")
    graph.add_edge("communication_agent", END)

    return graph.compile()


# ── Public Entry Point ────────────────────────────────────────────────────────

def process_claim(claim_input: ClaimInput) -> ClaimsState:
    """
    Process a single insurance claim through the full pipeline.

    Args:
        claim_input: Raw claim data dict

    Returns:
        Final ClaimsState with all agent outputs, decision, and audit trail
    """
    graph = build_claims_graph()
    state = initial_state(claim_input)
    state["execution_start_time"] = datetime.now(timezone.utc).isoformat()

    claim_id = claim_input["claim_id"]
    logger.info(f"[{claim_id}] Pipeline starting")

    try:
        final_state = graph.invoke(state)
        logger.info(
            f"[{claim_id}] Pipeline complete: "
            f"decision={final_state.get('final_decision')}, "
            f"amount=${(final_state.get('final_amount_usd') or 0):,.2f}, "
            f"agents={(final_state.get('agent_call_count') or 0)}, "
            f"cost=${(final_state.get('total_cost_usd') or 0):.4f}"
        )
        return final_state
    except Exception as e:
        logger.error(f"[{claim_id}] Pipeline crashed: {e}", exc_info=True)
        state["error_log"] = [f"Pipeline crash: {str(e)}"]
        state["final_decision"] = ClaimDecision.ESCALATED_HITL
        return state

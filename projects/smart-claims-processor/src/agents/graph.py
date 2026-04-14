"""
LangGraph Main Workflow - Smart Claims Processor

Orchestrates all agents in a conditional, state-driven pipeline.

Workflow paths:

PATH A - Normal (low fraud, low value):
  intake -> fraud_crew -> damage_assessor -> policy_checker
  -> settlement -> evaluator -> [evaluator_passed? -> communication] | [failed -> hitl]

PATH B - HITL (Human-In-The-Loop) Required (high fraud score):
  intake -> fraud_crew -> hitl_checkpoint -> [wait for human]
  -> damage_assessor -> policy_checker -> settlement -> evaluator -> communication

PATH B2 - HITL Required (eval quality gate failed):
  intake -> fraud_crew -> damage_assessor -> policy_checker
  -> settlement -> evaluator -> hitl_checkpoint -> [wait for human] -> communication

PATH C - Auto-Reject (confirmed fraud, score >= 0.90):
  intake -> fraud_crew -> auto_reject -> communication

PATH D - Intake Failure (invalid claim):
  intake -> [invalid] -> communication (denial)

PATH E - Fast Mode (amount < $500, clean history):
  intake -> settlement -> communication

Conditional routing functions determine which path to take at each junction.
"""

from __future__ import annotations # for Python 3.10 compatibility with forward references in type hints

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

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
from src.config import get_confidence_gate_config, get_hitl_config, get_pipeline_config
from src.evaluation.evaluator import run_evaluator
from src.hitl.checkpoint import check_hitl_required, format_hitl_brief
from src.hitl.queue import enqueue_claim
from src.models.schemas import ClaimDecision, FraudRiskLevel
from src.models.state import ClaimsState, initial_state, ClaimInput
from src.utils import currency_symbol

logger = logging.getLogger(__name__)


# ------- HITL Node -------------------------------------------─

def hitl_checkpoint_node(state: ClaimsState) -> dict:
    """
    HITL checkpoint - enqueues the claim for manual review and PAUSES the graph
    via LangGraph's `interrupt()` primitive. The pipeline does not resume until
    an approver calls `resume_claim(...)` with a decision payload.

    Return value from `interrupt(...)` is the reviewer's decision dict, supplied
    through `Command(resume={...})` when the approval endpoint re-invokes the
    graph.
    """
    claim = state["claim"]
    claim_id = claim["claim_id"]
    fraud_output = state.get("fraud_output")
    damage_output = state.get("damage_output")
    intake_output = state.get("intake_output")

    logger.info(f"[{claim_id}] HITL checkpoint triggered")

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

    logger.info(f"[{claim_id}] Pausing pipeline for manual approval (ticket={ticket_id})")

    # Hard pause. Execution suspends here until an approver supplies a decision
    # via Command(resume={...}) from the /api/hitl/decide endpoint.
    human_result: dict = interrupt({
        "ticket_id": ticket_id,
        "claim_id": claim_id,
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
            "ticket_id": ticket_id,
            "priority": priority.value,
            "priority_score": priority_score,
            "triggers": triggers,
            "human_decision": decision_str,
            "reviewer_id": (human_result or {}).get("reviewer_id"),
        }],
    }
    if override_amount is not None:
        result_update["final_amount_usd"] = float(override_amount)
    logger.info(f"[{claim_id}] Resumed from HITL with decision={decision_str}")
    return result_update


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


# ------- Routing Functions ----------------------------------------------------------------

def _check_confidence_gate(agent_key: str, confidence: float | None) -> bool:
    """Return True if confidence is below threshold (HITL needed)."""
    cfg = get_confidence_gate_config()
    if not cfg.get("enabled", False) or confidence is None:
        return False
    threshold = cfg.get("per_agent", {}).get(agent_key, cfg.get("default_threshold", 0.60))
    below = confidence < threshold
    if below:
        logger.info(f"Confidence gate: {agent_key} confidence {confidence:.2f} < {threshold:.2f} -> HITL")
    return below


def route_after_intake(state: ClaimsState) -> Literal[
    "fraud_crew", "communication_agent", "settlement_calculator", "hitl_after_intake"
]:
    """Route after intake: valid claims proceed, invalid ones go straight to communication."""
    intake = state.get("intake_output")

    if not intake or not intake.is_valid:
        logger.info(f"Routing to denial: intake invalid")
        return "communication_agent"

    # Confidence gate
    if _check_confidence_gate("intake_agent", intake.confidence):
        return "hitl_after_intake"

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
    "damage_assessor", "auto_reject", "hitl_checkpoint", "hitl_after_fraud"
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

    # Confidence gate: crew not confident in its own analysis
    if _check_confidence_gate("fraud_crew", fraud.consistency_score):
        return "hitl_after_fraud"

    return "damage_assessor"


def route_after_damage(state: ClaimsState) -> Literal[
    "policy_checker", "hitl_after_damage"
]:
    """Confidence gate after damage assessment."""
    damage = state.get("damage_output")
    if damage and _check_confidence_gate("damage_assessor", damage.assessment_confidence):
        return "hitl_after_damage"
    return "policy_checker"


def route_after_policy(state: ClaimsState) -> Literal[
    "settlement_calculator", "hitl_after_policy"
]:
    """Confidence gate after policy check."""
    policy = state.get("policy_output")
    if policy and _check_confidence_gate("policy_checker", policy.confidence):
        return "hitl_after_policy"
    return "settlement_calculator"


def route_after_settlement(state: ClaimsState) -> Literal[
    "evaluator", "hitl_after_settlement"
]:
    """Confidence gate after settlement calculation."""
    settlement = state.get("settlement_output")
    if settlement and _check_confidence_gate("settlement_calculator", settlement.confidence):
        return "hitl_after_settlement"
    return "evaluator"


def route_after_evaluation(state: ClaimsState) -> Literal[
    "hitl_checkpoint", "communication_agent"
]:
    """If evaluation fails quality gate, route to HITL before release."""
    evaluation_passed = state.get("evaluation_passed", True)
    if not evaluation_passed:
        logger.info("Evaluation failed quality gate - routing to HITL")
        return "hitl_checkpoint"
    return "communication_agent"


def route_after_hitl_checkpoint(state: ClaimsState) -> Literal[
    "damage_assessor", "communication_agent"
]:
    """Route after hitl_checkpoint resume.

    If damage_assessor hasn't run yet (fraud-triggered HITL), continue the
    pipeline through the remaining agents.  If it has already run
    (eval-triggered HITL), go straight to communication.
    """
    if state.get("damage_output") is None:
        logger.info("HITL resume: fraud-triggered, continuing to damage_assessor")
        return "damage_assessor"
    logger.info("HITL resume: eval-triggered, continuing to communication_agent")
    return "communication_agent"


# ------- Graph Builder ---------------------------------------------------------------------

# ------- Checkpointer (shared across invocations so interrupt/resume works) ---------------

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

    from pathlib import Path
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
    Build and return the compiled LangGraph StateGraph.

    Graph structure with confidence gates:
    START
      └─ intake_agent
           ├─ [invalid] -------------------------------------------------► communication_agent
           ├─ [low_confidence] ------------------------------------------► hitl_after_intake -> fraud_crew
           ├─ [fast_mode] -----------------------------------------------► settlement_calculator
           └─ fraud_crew
                ├─ [confirmed_fraud] ------------------------------------► auto_reject -> communication_agent
                ├─ [high_fraud] -----------------------------------------► hitl_checkpoint -> damage_assessor (continues pipeline)
                ├─ [low_confidence] -------------------------------------► hitl_after_fraud -> damage_assessor
                └─ damage_assessor
                     ├─ [low_confidence] --------------------------------► hitl_after_damage -> policy_checker
                     └─ policy_checker
                          ├─ [low_confidence] ---------------------------► hitl_after_policy -> settlement_calculator
                          └─ settlement_calculator
                               ├─ [low_confidence] ---------------------► hitl_after_settlement -> evaluator
                               └─ evaluator
                                    ├─ [passed] ----------------------------► communication_agent
                                    └─ [failed] ----------------------------► hitl_checkpoint -> communication_agent (eval HITL)
    """
    graph = StateGraph(ClaimsState)

    # ------- Core agent nodes ---------------------------------------------------------
    graph.add_node("intake_agent", run_intake_agent)
    graph.add_node("fraud_crew", run_fraud_crew)
    graph.add_node("damage_assessor", run_damage_assessor)
    graph.add_node("policy_checker", run_policy_checker)
    graph.add_node("settlement_calculator", run_settlement_calculator)
    graph.add_node("evaluator", run_evaluator)
    graph.add_node("auto_reject", auto_reject_node)
    graph.add_node("communication_agent", run_communication_agent)

    # ------- HITL nodes (same function, different node names -> different resume targets)
    graph.add_node("hitl_checkpoint", hitl_checkpoint_node)       # fraud/eval triggered
    graph.add_node("hitl_after_intake", hitl_checkpoint_node)     # intake low confidence
    graph.add_node("hitl_after_fraud", hitl_checkpoint_node)      # fraud crew low confidence
    graph.add_node("hitl_after_damage", hitl_checkpoint_node)     # damage low confidence
    graph.add_node("hitl_after_policy", hitl_checkpoint_node)     # policy low confidence
    graph.add_node("hitl_after_settlement", hitl_checkpoint_node) # settlement low confidence

    # ------- Entry -------------------------------------------
    graph.add_edge(START, "intake_agent")

    # ------- After intake: invalid -> deny, low confidence -> HITL, else -> fraud -------
    graph.add_conditional_edges("intake_agent", route_after_intake, {
        "fraud_crew": "fraud_crew",
        "communication_agent": "communication_agent",
        "settlement_calculator": "settlement_calculator",
        "hitl_after_intake": "hitl_after_intake",
    })

    # ------- After fraud: auto-reject, high fraud -> HITL, low confidence -> HITL, else -> damage
    graph.add_conditional_edges("fraud_crew", route_after_fraud, {
        "damage_assessor": "damage_assessor",
        "auto_reject": "auto_reject",
        "hitl_checkpoint": "hitl_checkpoint",
        "hitl_after_fraud": "hitl_after_fraud",
    })

    # ------- After damage: confidence gate -----------------------------------------------
    graph.add_conditional_edges("damage_assessor", route_after_damage, {
        "policy_checker": "policy_checker",
        "hitl_after_damage": "hitl_after_damage",
    })

    # ------- After policy: confidence gate ----------------------------------------------
    graph.add_conditional_edges("policy_checker", route_after_policy, {
        "settlement_calculator": "settlement_calculator",
        "hitl_after_policy": "hitl_after_policy",
    })

    # ------- After settlement: confidence gate ------------------------------------------
    graph.add_conditional_edges("settlement_calculator", route_after_settlement, {
        "evaluator": "evaluator",
        "hitl_after_settlement": "hitl_after_settlement",
    })

    # ------- After evaluation: quality gate ----------------------------------------------
    graph.add_conditional_edges("evaluator", route_after_evaluation, {
        "hitl_checkpoint": "hitl_checkpoint",
        "communication_agent": "communication_agent",
    })

    # ------- HITL resume targets: each gate resumes to the correct NEXT agent --------------
    graph.add_edge("hitl_after_intake", "fraud_crew")
    graph.add_edge("hitl_after_fraud", "damage_assessor")
    graph.add_edge("hitl_after_damage", "policy_checker")
    graph.add_edge("hitl_after_policy", "settlement_calculator")
    graph.add_edge("hitl_after_settlement", "evaluator")
    graph.add_conditional_edges("hitl_checkpoint", route_after_hitl_checkpoint, {
        "damage_assessor": "damage_assessor",
        "communication_agent": "communication_agent",
    })

    # ------- Terminal edges ------------------------------------------------------------
    graph.add_edge("auto_reject", "communication_agent")
    graph.add_edge("communication_agent", END)
    return graph


# Backwards-compat alias (other code/tests may still import this).
def build_claims_graph():
    return get_compiled_graph()


# ------- Public Entry Points ------------------------------------------------------------

def _thread_config(claim_id: str) -> dict:
    return {"configurable": {"thread_id": claim_id}}


def _is_paused(final_state: dict) -> bool:
    """A paused graph surfaces __interrupt__ in the last state."""
    return bool(final_state.get("__interrupt__"))


def _finalize(final_state: dict) -> dict:
    """Strip internal keys and return a clean state dict."""
    if "_guardrails_manager" in final_state:
        final_state.pop("_guardrails_manager", None)
    return final_state


def _store_to_memory(final_state: dict) -> None:
    """After pipeline completion, persist the claim to long-term + episodic memory.

    This is what makes the system learn from its own decisions over time.
    Future claims will retrieve these as "similar past claims" via the
    memory tools, giving agents historical context for better decisions.
    """
    try:
        from src.memory.manager import memory

        claim = final_state.get("claim", {})
        claim_id = claim.get("claim_id", "unknown")
        description = claim.get("incident_description", "")
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
                    f"Notes: {final_state.get('human_notes', 'none')}"
                ),
                event_type="human_override",
                metadata={"fraud_score": fraud_score, "reviewer": final_state.get("human_reviewer_id")},
            )

        if decision_str == "auto_rejected":
            memory.store_episode(
                claim_id=claim_id,
                narrative=(
                    f"Claim {claim_id} auto-rejected. Fraud score: {fraud_score:.2f}. "
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
        logger.debug("Memory storage skipped: %s", e)


def process_claim(claim_input: ClaimInput) -> dict:
    """
    Run the pipeline. Returns one of:
      { "paused": True,  "state": {...}, "interrupt": {...} }   # awaiting approver
      { "paused": False, "state": {...} }                        # completed
    Callers MUST check "paused" and, if True, persist status=pending_human_review.
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

    # Inject accumulated token/cost usage into the final state
    usage = get_token_usage()
    final_state["total_tokens_used"] = usage["total"]
    final_state["total_cost_usd"] = usage["cost"]

    if _is_paused(final_state):
        interrupts = final_state.get("__interrupt__") or []
        payload = interrupts[0].value if interrupts else {}
        logger.info(f"[{claim_id}] Pipeline paused at HITL checkpoint")
        return {"paused": True, "state": _finalize(final_state), "interrupt": payload}

    cs = currency_symbol()
    logger.info(
        f"[{claim_id}] Pipeline complete: decision={final_state.get('final_decision')}, "
        f"amount={cs}{(final_state.get('final_amount_usd') or 0):,.2f}, "
        f"tokens={usage['total']}, cost=${usage['cost']:.6f}"
    )
    _store_to_memory(final_state)
    return {"paused": False, "state": _finalize(final_state)}


def resume_claim(claim_id: str, decision: dict) -> dict:
    """
    Resume a paused pipeline with an approver's decision.

    decision = {
        "decision": "approved" | "denied" | "escalated_hitl" | ...,
        "reviewer_id": str,
        "notes": str,
        "override_ai": bool,
        "settlement_override_usd": float | None,
    }
    """
    graph = get_compiled_graph()
    cfg = _thread_config(claim_id)
    logger.info(f"[{claim_id}] Resuming pipeline with decision={decision.get('decision')}")
    final_state = graph.invoke(Command(resume=decision), config=cfg)

    if _is_paused(final_state):
        # Can happen legitimately: e.g. fraud HITL approved, then evaluator
        # quality gate fails -> second hitl_checkpoint pause on same thread.
        logger.info(f"[{claim_id}] Pipeline paused again after resume (e.g. eval quality gate)")
        return {"paused": True, "state": _finalize(final_state)}

    logger.info(f"[{claim_id}] Pipeline completed after human approval")
    _store_to_memory(final_state)
    return {"paused": False, "state": _finalize(final_state)}

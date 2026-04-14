"""
LLM-as-Judge Evaluator

Uses a separate LLM call (judge) to evaluate the quality of claim decisions.

Evaluation dimensions:
  1. Accuracy    - Is the settlement amount correctly calculated?
  2. Completeness - Were all policy clauses and exclusions checked?
  3. Fairness    - Is this consistent with similar claims?
  4. Safety      - Were all guardrails and rules followed?
  5. Transparency - Is the reasoning clear, complete, and traceable?

If overall_score < min_score_to_release, the pipeline can re-route
for human review rather than auto-releasing the decision.

This pattern is used in production at scale to catch edge cases that
pass individual agent checks but fail holistic quality review.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_evaluation_config
from src.llm import get_judge_llm
from src.models.schemas import EvaluationOutput
from src.models.state import ClaimsState
from src.security.audit_log import log_agent_action
from src.utils import currency_symbol

logger = logging.getLogger(__name__)
AGENT_NAME = "evaluator"

JUDGE_SYSTEM_PROMPT = """You are an impartial insurance claims quality reviewer.
Your role is to evaluate whether an AI system made a good decision on an insurance claim.

You are evaluating the PROCESS and REASONING, not re-adjudicating the claim.
Score each dimension from 0.0 to 1.0:

ACCURACY (0-1): Was the math correct? Are amounts within policy limits?
COMPLETENESS (0-1): Were all relevant policy clauses checked? Any blind spots?
FAIRNESS (0-1): Would a human adjuster reach the same conclusion? Any bias?
SAFETY (0-1): Were fraud signals handled appropriately? No guardrail violations?
TRANSPARENCY (0-1): Can you follow the reasoning step by step? Is it explainable?

Be strict but fair. A score of 0.7+ means acceptable. Below 0.7 means re-review."""


def run_evaluator(state: ClaimsState) -> dict:
    """LangGraph node for LLM-as-judge evaluation."""
    cfg = get_evaluation_config()
    claim = state["claim"]
    claim_id = claim["claim_id"]
    start_time = time.time()

    # Skip evaluation based on sample rate (for batch processing efficiency)
    import random
    sample_rate = cfg.get("batch_eval_sample_rate", 1.0)

    # Always evaluate HITL claims, high-value claims, and human overrides
    from src.config import get_hitl_config
    high_value_threshold = get_hitl_config()["triggers"].get("min_amount", 10000)
    always_eval = (
        state.get("hitl_required", False)
        or float(claim.get("estimated_amount", 0)) >= high_value_threshold
        or state.get("human_override", False)
    )

    if not always_eval and random.random() > sample_rate:
        logger.info(f"[{claim_id}] Evaluation skipped (sample rate)")
        return {
            "evaluation_passed": True,
            "pipeline_trace": [{"agent": AGENT_NAME, "status": "skipped_sampling"}],
        }

    logger.info(f"[{claim_id}] Evaluation started")

    settlement = state.get("settlement_output")
    fraud = state.get("fraud_output")
    policy_check = state.get("policy_output")
    damage = state.get("damage_output")
    guardrails_violations = state.get("guardrails_violations", [])

    llm = get_judge_llm().with_structured_output(EvaluationOutput)
    _sym = currency_symbol()

    settlement_summary = "No settlement computed"
    if settlement:
        settlement_summary = f"""
Decision: {settlement.decision.value}
Settlement Amount: {_sym}{settlement.settlement_amount_usd:,.2f}
Gross Damage: {_sym}{settlement.gross_damage_usd:,.2f}
Deductible Applied: {_sym}{settlement.deductible_applied_usd:,.2f}
Depreciation Applied: {_sym}{settlement.depreciation_applied_usd:,.2f}
Calculation Steps: {chr(10).join(settlement.calculation_breakdown)}
Denial Reasons: {', '.join(settlement.denial_reasons) or 'N/A'}
Confidence: {settlement.confidence:.2f}
"""

    policy_summary = "No policy check"
    if policy_check:
        policy_summary = f"""
Coverage Status: {policy_check.coverage_status.value}
Covered Amount: {_sym}{policy_check.covered_amount_usd:,.2f}
Deductible: {_sym}{policy_check.deductible_usd:,.2f}
Exclusions Checked: {', '.join(policy_check.exclusions_triggered) or 'None triggered'}
Compliance Flags: {', '.join(policy_check.compliance_flags) or 'None'}
Confidence: {policy_check.confidence:.2f}
"""

    prompt = f"""
Evaluate the quality of this insurance claim decision:

=== CLAIM ===
ID: {claim_id}
Type: {claim.get('incident_type')}
Estimated Amount: {_sym}{float(claim.get('estimated_amount', 0)):,.2f}
Description: {claim.get('incident_description', 'N/A')}

=== DAMAGE ASSESSMENT ===
Assessed Amount: {_sym}{(damage.assessed_damage_usd if damage else 0):,.2f}
Confidence: {(damage.assessment_confidence if damage else 0):.2f}
Recommendation: {damage.repair_vs_replace if damage else 'N/A'}

=== FRAUD ASSESSMENT ===
Risk Level: {fraud.fraud_risk_level.value if fraud else 'N/A'}
Fraud Score: {(fraud.fraud_score if fraud else 0):.2f}
Primary Concerns: {', '.join(fraud.primary_concerns[:3]) if fraud else 'N/A'}

=== POLICY CHECK ===
{policy_summary}

=== SETTLEMENT ===
{settlement_summary}

=== GUARDRAILS ===
Violations: {', '.join(guardrails_violations) or 'None'}
Agent Calls: {state.get('agent_call_count', 0)}
Tokens Used: {state.get('total_tokens_used', 0):,}

Evaluate all 5 dimensions. Be strict. The minimum passing score is {cfg.get('min_score_to_release', 0.70):.2f}.
Flag any critical issues that require immediate attention.
"""

    try:
        output = llm.invoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
    except Exception as e:
        logger.error(f"[{claim_id}] Evaluator LLM failed: {e}")
        output = EvaluationOutput(
            overall_score=0.50,
            accuracy_score=0.50,
            completeness_score=0.50,
            fairness_score=0.50,
            safety_score=0.50,
            transparency_score=0.50,
            passed=False,
            feedback=f"Evaluation LLM failed: {str(e)}. Routing to human review.",
            flags=["evaluation_system_error"],
        )

    min_score = cfg.get("min_score_to_release", 0.70)
    passed = output.overall_score >= min_score

    duration_ms = int((time.time() - start_time) * 1000)

    log_agent_action(
        claim_id=claim_id,
        agent_name=AGENT_NAME,
        action="llm_judge_evaluation",
        output_summary={
            "overall_score": output.overall_score,
            "passed": passed,
            "flags": output.flags,
        },
        duration_ms=duration_ms,
    )

    if not passed:
        logger.warning(
            f"[{claim_id}] Evaluation FAILED: score={output.overall_score:.2f} < {min_score:.2f}. "
            f"Flags: {output.flags}"
        )
    else:
        logger.info(f"[{claim_id}] Evaluation PASSED: score={output.overall_score:.2f}")

    return {
        "evaluation_output": output,
        "evaluation_passed": passed,
        "pipeline_trace": [{
            "agent": AGENT_NAME,
            "overall_score": output.overall_score,
            "passed": passed,
            "duration_ms": duration_ms,
            "confidence": output.overall_score,
            "decision": "passed" if passed else "failed",
            "reasoning": output.feedback,
            "flags": output.flags,
            "findings": {
                "accuracy": output.accuracy_score,
                "completeness": output.completeness_score,
                "fairness": output.fairness_score,
                "safety": output.safety_score,
                "transparency": output.transparency_score,
            },
        }],
        "agent_call_count": state.get("agent_call_count", 0) + 1,
    }

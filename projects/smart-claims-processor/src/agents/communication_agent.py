"""
Communication Agent (LangGraph Node)

Generates the final claimant notification and internal adjuster notes.
This is the last step - it synthesizes all previous agent outputs into
clear, professional communication.

Output:
- Claimant-facing notification (email/letter ready)
- Internal notes for the claims file
- Next steps and appeal instructions if denied
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import get_structured_llm
from src.models.schemas import ClaimDecision, CommunicationOutput
from src.models.state import ClaimsState
from src.config import get_communication_config
from src.security.audit_log import log_agent_action, log_final_decision
from src.utils import currency_symbol as _currency_symbol


def _get_appeal_instructions() -> str:
    comm = get_communication_config()
    phone = comm.get("contact_phone", "our claims department")
    text = comm.get("appeal_window_text", "You have 30 days to file an appeal.")
    return f"Contact {phone}. {text}"

logger = logging.getLogger(__name__)
AGENT_NAME = "communication_agent"

SYSTEM_PROMPT = """You are a professional insurance claims communication specialist.
Write clear, empathetic, and legally accurate communications to insurance claimants.

Guidelines:
- Be human and empathetic, especially for denied or partial claims
- Never use jargon without explanation
- Be specific about amounts, timelines, and next steps
- For denials, clearly explain the reason and appeal process
- Do not promise outcomes not supported by the settlement calculation
- Internal notes should be factual, complete, and suitable for legal review"""


def run_communication_agent(state: ClaimsState) -> dict:
    """LangGraph node for generating claimant communication."""
    claim = state["claim"]
    claim_id = claim["claim_id"]
    settlement = state.get("settlement_output")
    fraud_output = state.get("fraud_output")
    evaluation = state.get("evaluation_output")
    hitl_required = state.get("hitl_required", False)
    human_decision = state.get("human_decision")
    start_time = time.time()

    logger.info(f"[{claim_id}] Communication agent started")

    # Final decision - prefer human override if present
    final_decision = state.get("final_decision") or ClaimDecision.PENDING_DOCUMENTS
    final_amount = state.get("final_amount_usd") or 0.0

    decision_str = final_decision.value if hasattr(final_decision, "value") else str(final_decision)

    llm = get_structured_llm(CommunicationOutput)

    hitl_context = ""
    if hitl_required and human_decision:
        hitl_context = f"""
        NOTE: This claim was escalated for human review.
        Human reviewer decision: {human_decision}
        Human override of AI recommendation: {state.get('human_override', False)}
        Reviewer notes: {state.get('human_notes', 'None')}
        """

    fraud_context = ""
    if fraud_output and fraud_output.fraud_score > 0.5:
        fraud_context = "NOTE: This claim had elevated fraud indicators. Do NOT mention fraud in claimant communication - this is for internal notes only."

    eval_context = ""
    if evaluation:
        eval_context = f"Decision quality score: {evaluation.overall_score:.2f}/1.0"

    # Build denial reasons from settlement output or intake flags
    intake = state.get("intake_output")
    if settlement and settlement.denial_reasons:
        denial_reasons_text = ", ".join(settlement.denial_reasons)
    elif intake and intake.validation_flags:
        denial_reasons_text = ", ".join(intake.validation_flags)
    else:
        denial_reasons_text = "N/A"

    intake_notes_text = intake.intake_notes if intake else "N/A"

    prompt = f"""
    Generate the final claimant communication for this claim:

    CLAIM: {claim_id}
    POLICY: {claim.get('policy_number')}
    CLAIMANT NAME: [CLAIMANT] (use this placeholder - name will be merged separately)
    INCIDENT TYPE: {claim.get('incident_type', 'unknown').replace('_', ' ').title()}
    INCIDENT DATE: {claim.get('incident_date')}

    FINAL DECISION: {decision_str.upper().replace('_', ' ')}
    SETTLEMENT AMOUNT: {_currency_symbol()}{final_amount:,.2f}

    CALCULATION SUMMARY:
    {chr(10).join(settlement.calculation_breakdown) if settlement else 'N/A'}

    DENIAL REASONS (if denied): {denial_reasons_text}
    INTAKE NOTES: {intake_notes_text}
    EXCLUSIONS: {', '.join(state.get('policy_output').exclusions_triggered) if state.get('policy_output') else 'N/A'}
    {hitl_context}
    {fraud_context}
    {eval_context}

    Write:
    1. A professional email notification to the claimant explaining the decision.
       IMPORTANT: If the claim is denied, you MUST include the SPECIFIC reasons
       for denial in the email (e.g. "deductible exceeds claim amount",
       "policy was inactive on incident date", "claim type not covered").
       Never send a generic "does not meet criteria" without explaining which
       criteria and why. The claimant deserves to know exactly why.
    2. Next steps they need to take
    3. Appeal instructions if denied or partially denied
    4. Internal adjuster notes (technical, for the claims file)

    Keep the claimant email warm but professional. Under 300 words.
    Internal notes should be complete and factual. Can be longer.
    """

    try:
        output = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
    except Exception as e:
        logger.error(f"[{claim_id}] Communication agent LLM failed: {e}")
        output = CommunicationOutput(
            subject=f"Insurance Claim {claim_id} - Decision Notice",
            message=_fallback_message(claim_id, decision_str, final_amount),
            internal_notes=f"Communication LLM error: {str(e)}. Fallback template used.",
            next_steps=["Contact your agent for details"],
            appeal_instructions=_get_appeal_instructions() if "denied" in decision_str else None,
        )

    duration_ms = int((time.time() - start_time) * 1000)

    # Log final decision (compliance record)
    log_final_decision(
        claim_id=claim_id,
        decision=decision_str,
        amount_usd=final_amount,
        total_tokens=state.get("total_tokens_used", 0),
        total_cost_usd=state.get("total_cost_usd", 0.0),
        evaluation_score=evaluation.overall_score if evaluation else None,
        human_reviewed=hitl_required,
    )

    log_agent_action(
        claim_id=claim_id,
        agent_name=AGENT_NAME,
        action="communication_generated",
        output_summary={"subject": output.subject, "decision": decision_str},
        duration_ms=duration_ms,
    )

    logger.info(f"[{claim_id}] Communication generated. Pipeline complete.")

    return {
        "communication_output": output,
        "pipeline_trace": [{
            "agent": AGENT_NAME,
            "decision": decision_str,
            "amount_usd": final_amount,
            "duration_ms": duration_ms,
            "confidence": 1.0,
            "reasoning": output.internal_notes,
            "flags": [],
            "findings": {
                "subject": output.subject,
                "next_steps": output.next_steps,
                "has_appeal_instructions": bool(output.appeal_instructions),
            },
        }],
    }


def _fallback_message(claim_id: str, decision: str, amount: float) -> str:
    sym = _currency_symbol()
    comm = get_communication_config()
    phone = comm.get("contact_phone", "our claims department")
    email = comm.get("contact_email", "claims@insurance.com")
    footer = comm.get("regulatory_footer", "")
    appeal_text = comm.get("appeal_window_text", "You have 30 days to file an appeal.")

    decision_text = {
        "approved": f"We are pleased to inform you that your claim has been approved for a settlement of {sym}{amount:,.2f}.",
        "denied": "We regret to inform you that your claim has been denied based on our policy review.",
        "approved_partial": f"Your claim has been partially approved. A settlement of {sym}{amount:,.2f} will be processed.",
        "escalated_human_review": "Your claim is under additional review. We will contact you within 3-5 business days.",
        "fraud_investigation": "Your claim requires additional verification. Please expect contact from our team.",
    }.get(decision, f"Your claim status has been updated. Reference: {claim_id}")

    denied_line = f"\n{appeal_text}" if "denied" in decision else ""

    return f"""Dear Claimant,

    Re: Claim Reference {claim_id}

    {decision_text}{denied_line}

    Please contact us at {phone} or {email} if you have questions.

    {footer}
    Sincerely,
    Claims Processing Team"""

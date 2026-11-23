"""Postmortem agent: produce a blameless writeup once the loop terminates.

Runs on EVERY terminal outcome, not just successes. A postmortem for a run that
escalated or hit a ceiling is more valuable than one for a run that worked, because
it is the record of what the agent could not do.

This agent also feeds the memory loop. `runbook_lesson` is a single self-contained
sentence that gets written into episodic memory and retrieved on future incidents,
so the prompt insists it stand alone: a lesson that says "the usual fix applied"
is useless to a responder six weeks later who has no idea what "usual" meant.
"""

from __future__ import annotations

import time

from ..llm import LLMClient
from ..models.schemas import PostmortemResult
from ..models.state import EventType, HaltReason, IncidentState
from ._prompting import action_history_block, incident_context, join_sections, system_prompt, triage_block

AGENT = "Postmortem"

_SYSTEM = system_prompt(
    "ROLE: incident postmortem author.\n"
    "TASK: write a blameless, factual writeup of the incident that just concluded.\n"
    "\n"
    "RULES:\n"
    "1. Blameless: describe systems and conditions, never people. 'The deploy pipeline had no "
    "canary stage', not 'the engineer skipped testing'.\n"
    "2. Stay inside the evidence. If the root cause was never confirmed, say it was not "
    "confirmed and state the best-supported hypothesis as a hypothesis.\n"
    "3. If the incident ESCALATED or hit a ceiling, that is the outcome to report. Do not write "
    "it up as a success. Explain what blocked the automated response - that is the most "
    "useful content in the document.\n"
    "4. Recommendations must be specific and actionable: 'add a pool-utilization alert at 80% "
    "for orders-service', not 'improve monitoring'.\n"
    "5. runbook_lesson: ONE self-contained sentence mapping symptom to fix, written for a "
    "responder who was not here and has no context - it will be stored and retrieved verbatim "
    "for future incidents. Include the distinguishing symptom, not just the fix.\n"
    "6. detection_gap: why did monitoring not catch this earlier? Say 'not identifiable from "
    "the available data' if that is the honest answer."
)


#: Bookkeeping events that would dominate the narrative without adding to it.
#: The complete log is still on `state.events` - this is the readable retelling,
#: not the audit trail, and those are different documents with different jobs.
_TIMELINE_NOISE = {EventType.TOKEN_USAGE, EventType.STEP_START}


def _timeline_detail(event) -> str:
    """One line saying what was actually decided at this step.

    Without this the timeline reads "Diagnostician: diagnostic" - the shape of the
    run with none of its content, which tells a reader nothing they could not have
    guessed from the agent list. The reasoning is already in the payload; it was
    simply being discarded.
    """
    p = event.payload or {}
    t = event.type

    def pct(v):
        return f"{round(float(v) * 100)}%" if v is not None else ""

    if t == EventType.STATUS:
        return str(p.get("message") or p.get("status") or "")
    if t == EventType.LOOP:
        return f"iteration {p.get('iteration')} began"
    if t == EventType.MEMORY_RECALL:
        return f"recalled {p.get('count', 0)} similar past incident(s)"
    if t == EventType.ROUTE:
        why = p.get("reason")
        return f"next: {p.get('route')}" + (f" - {why}" if why else "")
    if t == EventType.STEP_END:
        return str(p.get("name") or p.get("blast_radius") or "")
    if t == EventType.HYPOTHESIS:
        return f"{p.get('cause', '')} ({pct(p.get('confidence'))} confidence)"
    if t == EventType.DIAGNOSTIC:
        return f"{p.get('leading_cause', '')} ({pct(p.get('confidence'))} confidence)"
    if t == EventType.EVIDENCE_REQUEST:
        return f"asked for {', '.join(p.get('tools') or [])} - {p.get('reason', '')}"
    if t == EventType.EVIDENCE_GATHERED:
        return f"ran {', '.join(p.get('tools') or [])}"
    if t == EventType.PROPOSED_ACTION:
        return (
            f"{p.get('action_type')} on {p.get('target')} ({p.get('risk')} risk) - "
            f"{p.get('rationale', '')} Predicts: {p.get('expected_signal', '')}"
        )
    if t == EventType.CRITIQUE:
        verdict = "endorsed" if p.get("endorse") else "VETOED"
        return f"{verdict} - {p.get('strongest_objection', '')}"
    if t == EventType.GUARDRAIL:
        return f"{p.get('kind', 'rule')}: {p.get('message') or p.get('detail') or ''}"
    if t == EventType.AWAITING_APPROVAL:
        return f"paused for a human on {p.get('action_type')} ({p.get('risk')} risk)"
    if t == EventType.ACTION_EXECUTED:
        return f"{p.get('action_type')} on {p.get('target')} - {p.get('reason') or p.get('result') or ''}"
    if t == EventType.VERIFICATION:
        met = p.get("expected_signal_met")
        return (
            f"{'recovered' if p.get('recovered') else 'not recovered'}; "
            f"prediction {'held' if met else 'did not hold'}"
            + (f" - {p.get('signal_assessment')}" if p.get("signal_assessment") else "")
        )
    if t in (EventType.RESOLVED, EventType.ESCALATED, EventType.CANCELLED):
        return str(p.get("message") or p.get("reason") or "")
    if t == EventType.POSTMORTEM:
        return "postmortem written"
    if t == EventType.ERROR:
        return str(p.get("message") or "")
    return ""


def _facts(state: IncidentState) -> dict:
    """The parts of a postmortem that come from our own records, not from a model."""
    timeline = [
        {
            "ts": e.ts,
            "iteration": e.iteration,
            "agent": e.agent.value if e.agent else "system",
            "type": e.type.value,
            "event": f"{e.agent.value if e.agent else 'system'}: {e.type.value}",
            "detail": _timeline_detail(e),
        }
        for e in state.events
        if e.type not in _TIMELINE_NOISE
    ]
    actions_taken = [
        {
            "action_type": a.action_type,
            "target": a.target,
            "risk": a.risk.value,
            "status": a.status.value,
            "iteration": a.iteration,
            "expected_signal": a.expected_signal,
            "effective": (a.result or {}).get("effective"),
            "reason": (a.result or {}).get("reason"),
            # Audit: who let this action run (or stopped it), and when.
            "decided_by": a.decided_by,
            "decided_at": a.decided_at,
        }
        for a in state.actions
    ]
    evidence = [
        {"source": "diagnostics", "detail": d.get("leading_cause", ""), "confidence": d.get("confidence")}
        for d in state.diagnostics
    ]
    leading = state.diagnostics[-1].get("leading_cause", "unknown") if state.diagnostics else "unknown"
    return {"timeline": timeline, "actions_taken": actions_taken, "evidence": evidence, "leading": leading}


def _assemble(state: IncidentState, facts: dict, res: PostmortemResult) -> dict:
    return {
        "summary": res.summary,
        "root_cause": res.root_cause,
        "contributing_factors": res.contributing_factors,
        "timeline": facts["timeline"],
        "actions_taken": facts["actions_taken"],
        "evidence": facts["evidence"],
        "recommendations": res.recommendations,
        "runbook_lesson": res.runbook_lesson,
        "detection_gap": res.detection_gap,
        "halt_reason": state.halt_reason.value if state.halt_reason else None,
        "guardrail_flags": state.guardrail_flags,
        "started_by": state.created_by,
        "cancelled_by": state.cancelled_by,
        "generated_at": time.time(),
    }


def model_should_write(state: IncidentState) -> bool:
    """False when the run ended because the model failed every retry.

    Asking that model again for the writeup only repeats the failure: it delays the
    postmortem by another round of retries and feeds the circuit breaker. The record
    is assembled from state instead, straight away.
    """
    return state.halt_reason is not HaltReason.LLM_UNAVAILABLE


UNAVAILABLE_WHY = "the model was unavailable for this run, so it was not asked again."


def fallback(state: IncidentState, why: str) -> dict:
    """A postmortem built from the record alone, for when no model can write one.

    Invariant 6 says every terminal incident has a postmortem. The runs where the
    model was unreachable are exactly the ones an operator most needs a record of,
    so this one is assembled from state: what ran, how it ended, and why the
    narrative is missing. It deliberately has no runbook lesson, so nothing
    invented here is ever recalled later as experience.
    """
    facts = _facts(state)
    halt = state.halt_reason.value if state.halt_reason else "unknown"
    res = PostmortemResult(
        summary=f"The incident ended as {state.status.value} ({halt}). No model wrote this summary: {why}",
        root_cause=facts["leading"],
        contributing_factors=[],
        recommendations=["Review the timeline below; the narrative postmortem could not be generated."],
        runbook_lesson="",
        detection_gap="",
    )
    return _assemble(state, facts, res)


def run(state: IncidentState, llm: LLMClient) -> dict:
    facts = _facts(state)
    leading = facts["leading"]
    evidence = facts["evidence"]
    verifications = [
        {
            "recovered": v.get("recovered"),
            "expected_signal_met": v.get("expected_signal_met"),
            "assessment": v.get("signal_assessment"),
        }
        for v in state.verification_history
    ]

    user = join_sections(
        incident_context(state, include_budget=False),
        triage_block(state),
        join_sections(
            "HOW THE RUN ENDED:",
            f"- final status: {state.status.value}",
            f"- halt reason: {state.halt_reason.value if state.halt_reason else 'unknown'}",
            f"- loop iterations used: {state.iterations} of {state.max_iterations}",
            f"- evidence-gathering rounds: {state.evidence_rounds}",
            f"- cost: ${state.total_cost_usd:.4f}",
        ),
        f"BEST-SUPPORTED ROOT CAUSE FROM DIAGNOSIS: {leading}",
        f"DIAGNOSTIC HISTORY: {evidence}",
        action_history_block(state),
        f"VERIFICATION HISTORY: {verifications}",
        (
            f"GUARDRAILS THAT FIRED: {state.guardrail_flags}"
            if state.guardrail_flags
            else "GUARDRAILS THAT FIRED: none."
        ),
        "Write the postmortem.",
    )
    res: PostmortemResult = llm.structured(
        PostmortemResult,
        _SYSTEM,
        user,
        incident_id=state.incident_id,
        agent=AGENT,
        # The postmortem is written after the loop has already stopped, so it is
        # exempt from the pre-flight budget check: an incident that hit its cost
        # ceiling is exactly the one whose writeup we most want.
        budget_remaining_usd=None,
    )

    return _assemble(state, facts, res)

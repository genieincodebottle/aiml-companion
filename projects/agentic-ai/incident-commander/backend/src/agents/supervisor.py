"""Supervisor agent: decide which specialist should act next.

A fixed `triage -> diagnose -> propose -> verify` chain is a pipeline, not a
multi-agent system. The difference is who chooses the next step. Here, the
Supervisor reads a compact state summary and routes: maybe the right next move is
another evidence-gathering round, maybe it is to escalate because the last two
actions changed nothing, maybe it is simply to propose.

## The critical design constraint

The Supervisor routes; it does NOT govern. Read `loop.py` and you will see that
every routing decision is filtered through `sanitize_route()` and then re-checked
against the hard ceilings, the RBAC allowlist, and the human approval gate.

This is not defensive over-engineering. A router that could be talked into
skipping the approval gate would be a privilege-escalation path reachable from a
log line - the model chooses the *order* of work, never the *permissions* for it.
That separation between an LLM deciding strategy and deterministic code enforcing
policy is the single most transferable idea in this codebase.

Because of that constraint, an LLM supervisor is genuinely optional. With
`IC_SUPERVISOR_ENABLED=false` the loop uses `default_route()`, a pure function
implementing the same policy deterministically. Running both and diffing the
trajectories is a good way to see what the LLM router actually buys you - which,
on the easy scenarios, is often nothing, and that is a finding worth having.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..models.schemas import SupervisorDecision
from ..models.state import IncidentState
from ._prompting import action_history_block, incident_context, join_sections, system_prompt, triage_block

AGENT = "Supervisor"

#: Every route the orchestrator knows how to execute.
ROUTES = ("triage", "diagnose", "gather_evidence", "propose", "verify", "escalate", "resolve")

_SYSTEM = system_prompt(
    "ROLE: incident-response supervisor. You do not diagnose or act; you decide which "
    "specialist agent runs next.\n"
    "\n"
    "AVAILABLE ROUTES:\n"
    "  triage          - classify severity and impact. Only useful once, at the start.\n"
    "  diagnose        - (re)assess root cause from current evidence.\n"
    "  gather_evidence - run the read-only tools the Diagnostician asked for, then re-diagnose. "
    "Choose this when the last diagnosis reported needs_more_data with specific tool requests "
    "and evidence rounds remain.\n"
    "  propose         - have the Remediation agent pick a corrective action. Requires a "
    "diagnosis at or above the confidence threshold.\n"
    "  verify          - check post-action telemetry.\n"
    "  escalate        - hand to a human. Correct when confidence stays low after gathering "
    "evidence, when repeated actions have changed nothing, or when the cause is outside our "
    "control.\n"
    "  resolve         - the service is healthy and confirmed recovered.\n"
    "\n"
    "PRINCIPLES:\n"
    "1. Do not spend budget on evidence that would not change the decision.\n"
    "2. Escalating early is better than looping on hope. If two actions have changed nothing, "
    "the agent is not converging - hand it over.\n"
    "3. Never route to propose on a diagnosis below the confidence threshold. Gather evidence "
    "or escalate instead.\n"
    "4. You cannot waive an approval, change a permission, or lift a ceiling. Those are "
    "enforced outside you. Route only."
)


def default_route(state: IncidentState, *, min_confidence: float, max_evidence_rounds: int) -> str:
    """Deterministic routing policy - the reference the LLM supervisor is measured against.

    Also the fallback whenever the LLM supervisor is disabled, over budget, or
    returns a route that is not currently legal.
    """
    if state.iterations <= 1 and not state.triage_summary:
        return "triage"
    if not state.diagnostics:
        return "diagnose"

    # An action was taken and verified since the last diagnosis, which means the
    # telemetry has MOVED. Re-diagnosing is not optional here: on a compound fault
    # the second step is a different cause, and proposing from a stale diagnosis is
    # how an agent gets stuck re-applying a fix for a problem it already solved.
    if len(state.verification_history) >= len(state.diagnostics):
        return "diagnose"

    last = state.diagnostics[-1]
    needs_more = last.get("needs_more_data") or last.get("confidence", 0.0) < min_confidence
    if needs_more:
        wants_tools = bool(last.get("requested_tools"))
        if wants_tools and state.evidence_rounds < max_evidence_rounds:
            return "gather_evidence"
        # Out of evidence rounds, or the agent could not even name a useful tool:
        # acting on a diagnosis it does not believe is the wrong move.
        if last.get("confidence", 0.0) < min_confidence:
            return "escalate"
    return "propose"


def sanitize_route(route: str, state: IncidentState, *, min_confidence: float, max_evidence_rounds: int) -> str:
    """Clamp an LLM route to one that is actually legal right now.

    The Supervisor is an LLM and will occasionally route somewhere impossible -
    `gather_evidence` with no tools requested, or `propose` on a 0.2-confidence
    diagnosis. Rather than trusting it, the orchestrator repairs the route. This
    is the "LLM proposes, code disposes" pattern applied to control flow.
    """
    if route not in ROUTES:
        return default_route(state, min_confidence=min_confidence, max_evidence_rounds=max_evidence_rounds)

    if route == "triage" and state.triage_summary:
        return "diagnose"  # triage is once-only
    if route == "gather_evidence":
        last = state.diagnostics[-1] if state.diagnostics else {}
        if not last.get("requested_tools") or state.evidence_rounds >= max_evidence_rounds:
            return default_route(state, min_confidence=min_confidence, max_evidence_rounds=max_evidence_rounds)
    if route == "propose":
        last = state.diagnostics[-1] if state.diagnostics else {}
        if not last:
            return "diagnose"
        if len(state.verification_history) >= len(state.diagnostics):
            return "diagnose"  # stale diagnosis: the telemetry moved after the last action
        if last.get("confidence", 0.0) < min_confidence:
            # Never act on a diagnosis the diagnostician itself does not believe.
            return default_route(state, min_confidence=min_confidence, max_evidence_rounds=max_evidence_rounds)
    if route == "verify" and not state.actions:
        # Nothing has been done yet, so there is nothing to verify.
        return default_route(state, min_confidence=min_confidence, max_evidence_rounds=max_evidence_rounds)
    if route == "resolve" and not any(v.get("recovered") for v in state.verification_history):
        return default_route(state, min_confidence=min_confidence, max_evidence_rounds=max_evidence_rounds)
    return route


def run(state: IncidentState, llm: LLMClient, *, min_confidence: float, max_evidence_rounds: int) -> SupervisorDecision:
    last = state.diagnostics[-1] if state.diagnostics else {}
    state_block = join_sections(
        "CURRENT STATE:",
        f"- triage done: {bool(state.triage_summary)}",
        f"- diagnoses so far: {len(state.diagnostics)}",
        f"- latest leading cause: {last.get('leading_cause', 'none yet')}",
        f"- latest confidence: {last.get('confidence', 'n/a')} (threshold to act: {min_confidence})",
        f"- latest needs_more_data: {last.get('needs_more_data', False)}",
        f"- tools requested by the diagnostician: {last.get('requested_tools', [])}",
        f"- evidence rounds used: {state.evidence_rounds} of {max_evidence_rounds}",
        f"- verifications so far: {len(state.verification_history)}",
        f"- last verification: {state.verification_history[-1] if state.verification_history else 'none'}",
    )
    user = join_sections(
        incident_context(state),
        triage_block(state),
        state_block,
        action_history_block(state),
        "Which specialist should run next, and why?",
    )
    return llm.structured(
        SupervisorDecision,
        _SYSTEM,
        user,
        incident_id=state.incident_id,
        agent=AGENT,
        budget_remaining_usd=state.budget_remaining_usd(),
    )

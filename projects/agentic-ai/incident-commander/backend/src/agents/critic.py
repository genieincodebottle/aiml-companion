"""Critic agent: adversarially review a proposed action before anyone runs it.

This is the cheapest safety mechanism in the system and the one most often missing
from agent designs. The Remediation agent is optimizing for "produce a plausible
action"; nothing in that objective rewards noticing that the action is wrong. So a
second agent is given the opposite objective: find the reason this is a mistake.

Design notes that make the difference between a real critic and a rubber stamp:

- **It is not shown the Remediation agent's confidence.** Confidence is contagious;
  a critic told "the planner is 0.9 sure" will find reasons to agree.
- **`strongest_objection` is required even when endorsing.** Without that field a
  critic can discharge its duty with "looks good", which is worth nothing. Forcing
  the best counter-argument out of it is what makes the review real.
- **A veto is cheap; an error is not.** The prompt says so explicitly, because the
  default failure mode of an LLM reviewer is agreeableness.
- **A veto does not silently kill the incident.** It sends the loop back to
  re-diagnose with the objection recorded. The Critic advises; the orchestrator and
  the human still decide.

Enable/disable with `IC_CRITIC_ENABLED` - turning it off and watching what changes
is one of the more instructive experiments in this project.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..models.schemas import CriticVerdict, DiagnosisResult, RemediationProposal
from ..models.state import ActionRisk, IncidentState
from ._prompting import (
    ACTION_MENU,
    action_history_block,
    evidence_block,
    incident_context,
    join_sections,
    system_prompt,
    telemetry_block,
    triage_block,
)

AGENT = "Critic"

_SYSTEM = system_prompt(
    "ROLE: adversarial reviewer of a proposed production remediation. You are the last "
    "automated check before a human is asked to approve this.\n"
    "\n"
    "TASK: try to find the reason this action is WRONG. You are not here to agree.\n"
    "\n"
    "Judge the proposal against THIS system's action contract below, not against general "
    "intuition about what an action might do elsewhere. An action used for exactly the fault "
    "its contract names is not wrong for being that action; object to the evidence, the "
    "prediction or the risk instead.\n"
    "\n" + ACTION_MENU + "\n"
    "CHECK, IN ORDER:\n"
    "1. Does the evidence actually support the stated cause, or is the cause an assumption "
    "dressed up as a finding? Look for cited numbers, and check they appear in the telemetry.\n"
    "2. Is there a LESS INVASIVE action that would address the same cause? If so, name it in "
    "safer_alternative and do not endorse.\n"
    "3. Would this action make things WORSE if the diagnosis is wrong? Weight your objection by "
    "that downside. Restarting a healthy service is an outage you caused.\n"
    "4. Is expected_signal specific enough to falsify the claim? A prediction that cannot fail "
    "cannot verify anything - object to it.\n"
    "5. Has this action, or one just like it, already been tried without effect?\n"
    "6. Is the real cause outside our control (an expired third-party certificate, a vendor "
    "outage)? If so, every action except no_op is disruption without benefit - say that.\n"
    "\n"
    "OUTPUT RULES:\n"
    "- strongest_objection is REQUIRED even when you endorse. State the best argument against "
    "the action anyway. 'No objection' is not an acceptable answer; if the action is sound, "
    "name the assumption it depends on.\n"
    "- Endorse only if you genuinely cannot find a better next step. A wrong veto costs one "
    "loop iteration. A wrong endorsement costs a production outage. Veto when torn."
)


def run(
    state: IncidentState,
    diagnosis: DiagnosisResult,
    proposal: RemediationProposal,
    risk: ActionRisk,
    observations: dict,
    llm: LLMClient,
) -> CriticVerdict:
    proposal_block = join_sections(
        "PROPOSED ACTION UNDER REVIEW:",
        f"- action: {proposal.action_type.value}",
        f"- target: {proposal.target}",
        f"- risk class (assigned by the guardrail layer, not by the proposing agent): {risk.value}",
        f"- reversible: {proposal.reversible}",
        f"- rationale: {proposal.rationale}",
        f"- expected signal: {proposal.expected_signal}",
        f"- declared fallback: "
        f"{proposal.fallback_if_ineffective.value if proposal.fallback_if_ineffective else 'none declared'}",
        "",
        "THE DIAGNOSIS IT RESTS ON:",
        f"- leading cause: {diagnosis.leading_cause}",
        "- cited evidence:\n" + "\n".join(f"  * {e}" for e in diagnosis.evidence)
        if diagnosis.evidence
        else "- cited evidence: NONE PROVIDED (this is itself a finding)",
    )
    user = join_sections(
        incident_context(state),
        triage_block(state),
        proposal_block,
        action_history_block(state),
        telemetry_block(observations),
        # The evidence the Diagnostician asked for (config diff, pool stats, ...). Without it
        # the Critic judges a diagnosis built on evidence it cannot see, and in a live run it
        # vetoed the correct fix for Silent Config Drift three times as "hallucinated".
        evidence_block(state),
        "Find the strongest reason this action is wrong. Then decide whether to endorse it.",
    )
    return llm.structured(
        CriticVerdict,
        _SYSTEM,
        user,
        incident_id=state.incident_id,
        agent=AGENT,
        budget_remaining_usd=state.budget_remaining_usd(),
    )

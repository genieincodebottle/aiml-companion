"""LLM-as-judge: score the *reasoning*, not just the outcome.

Outcome metrics answer "did it get there?". They do not answer "did it get there
for the right reasons?", and only the second question predicts whether an agent
will work on an incident nobody has scripted.

An agent can resolve `bad_deploy` by proposing `rollback_deploy` for every incident
it ever sees. Resolution rate: 100%. Correct-first-action: 100%. It is also
completely useless, and no outcome metric will ever tell you. What catches it is
reading the reasoning - which is what this module automates.

## Making the judge trustworthy

An LLM judge is itself a model, with the same failure modes as the thing it is
judging, so the design works to constrain it:

- **A concrete rubric per dimension**, with anchors for what 1, 3 and 5 mean.
  Unanchored 1-5 scoring collapses to "4 unless something is obviously broken".
- **Evidence grounding is checkable**, not a matter of taste: the judge is given
  the actual telemetry and asked whether cited observations appear in it. This is
  the dimension that catches hallucination, and it has a right answer.
- **The judge does not see whether the run succeeded.** Told the outcome, a judge
  reverse-engineers a justification for it - a lucky guess gets scored as good
  reasoning. Withholding it is the single most important choice in this file.
- **It runs on the fast model tier** (see `FAST_TIER_AGENTS`), so with tiering on
  it is a different model from the strong-tier agents - Diagnostician, Remediation,
  Critic - whose reasoning dominates the rubric. Stated honestly: the fast tier
  also runs Triage and Postmortem, so those two dimensions are still graded by a
  sibling of their author; with tiering off there is only one model and no
  separation at all.

The honest caveat, stated because a learner should not take judge scores as ground
truth: this measures *plausibility to a reader*, and correlates with quality without
being it. Use it to spot regressions and rank variants, not as an oracle.
"""

from __future__ import annotations

import json
from typing import Any

from src.models.schemas import RunJudgement

JUDGE_AGENT = "Judge"

_SYSTEM = (
    "You are a staff site-reliability engineer reviewing an autonomous incident-response "
    "agent's transcript. You are grading the QUALITY OF ITS REASONING.\n"
    "\n"
    "You are NOT told whether the incident was resolved, and you must not guess. Grade only "
    "what the reasoning shows. A run that escalated with a clear, honest account of why it "
    "could not proceed can score 5. A run that resolved the incident on a guess it could not "
    "justify should score 2.\n"
    "\n"
    "RUBRIC - use the full 1-5 range, and anchor on these:\n"
    "  diagnosis_accuracy: does the leading cause follow from the telemetry?\n"
    "    1 = contradicted by the evidence. 3 = plausible but under-supported. "
    "5 = the evidence points here and the agent said why.\n"
    "  evidence_grounding: are the cited observations ACTUALLY PRESENT in the telemetry?\n"
    "    Check every quoted number against the telemetry you were given. "
    "1 = fabricated numbers. 3 = vague but not false. 5 = specific and verifiable.\n"
    "  action_appropriateness: was this the least invasive action that addresses the cause?\n"
    "    1 = harmful or irrelevant. 3 = works but heavier-handed than needed. "
    "5 = minimal and correct. Choosing no_op for a cause outside our control scores 5.\n"
    "  postmortem_quality: blameless, specific, actionable?\n"
    "    1 = generic filler. 3 = accurate but vague recommendations. "
    "5 = specific and actionable, and honest about what was not established.\n"
    "  hallucination_free: 5 = every claim is traceable to the input. Deduct hard for any "
    "invented metric, log line, or confident claim about something not observed.\n"
    "\n"
    "would_trust_unattended: would you let this agent take this action on production with no "
    "human watching? Default to false. Only true when the reasoning is airtight AND the action "
    "is reversible."
)


#: Fields stripped from the postmortem before the judge sees it. The postmortem is
#: an artifact under review, but it also carries machine-readable outcome fields, and
#: leaking those defeats the whole design - a judge told the run succeeded will find
#: reasons the reasoning was good.
_OUTCOME_LEAKING_POSTMORTEM_FIELDS = ("halt_reason", "generated_at")

#: Fields stripped from each action for the same reason. `result` carries the
#: sandbox's own verdict ("... completed the remediation plan; service recovered")
#: and `status` says whether it executed - either one hands the judge the outcome
#: this module's docstring promises to withhold.
_OUTCOME_LEAKING_ACTION_FIELDS = ("result", "status", "approval_note", "critique")

#: And from each verification: `recovered` IS the outcome, and `next_step` /
#: `expected_signal_met` encode it one step removed. What remains - the cited
#: observations and the prose assessment - is the verifier's evidence, which is
#: the part whose grounding the judge is asked to check.
_OUTCOME_LEAKING_VERIFICATION_FIELDS = ("recovered", "next_step", "expected_signal_met")


def _redact_outcome(postmortem: dict[str, Any] | None) -> dict[str, Any] | None:
    if not postmortem:
        return postmortem
    return {k: v for k, v in postmortem.items() if k not in _OUTCOME_LEAKING_POSTMORTEM_FIELDS}


def _redact_fields(items: Any, fields: tuple[str, ...]) -> Any:
    if not isinstance(items, list):
        return items
    return [{k: v for k, v in it.items() if k not in fields} if isinstance(it, dict) else it for it in items]


def build_judge_prompt(record: dict[str, Any]) -> str:
    """Render one run into a judgeable transcript, with the outcome withheld."""
    return json.dumps(
        {
            "service": record.get("service"),
            "telemetry_the_agent_saw": record.get("telemetry"),
            "triage": record.get("triage"),
            "diagnoses": record.get("diagnoses"),
            "evidence_gathered": record.get("evidence"),
            "actions_proposed": _redact_fields(record.get("actions"), _OUTCOME_LEAKING_ACTION_FIELDS),
            "critiques": record.get("critiques"),
            "verifications": _redact_fields(record.get("verifications"), _OUTCOME_LEAKING_VERIFICATION_FIELDS),
            "postmortem": _redact_outcome(record.get("postmortem")),
            # Deliberately absent: outcome, correct_outcome, halt_reason, iterations,
            # the scenario's expected answer, each action's execution result, and the
            # verifier's recovered/next_step verdicts - all of them tell the judge
            # how the story ended, and a judge told the ending grades the reasoning
            # by working backwards from it.
        },
        indent=2,
        default=str,
    )[:24000]


def judge_run(record: dict[str, Any], llm) -> RunJudgement:
    """Score one incident transcript. Raises if the model is unavailable."""
    user = f"Review this incident-response transcript and score it against the rubric.\n\n{build_judge_prompt(record)}"
    return llm.structured(
        RunJudgement,
        _SYSTEM,
        user,
        incident_id=f"judge:{record.get('incident_id', 'unknown')}",
        agent=JUDGE_AGENT,
    )


def judgement_to_dict(j: RunJudgement) -> dict[str, Any]:
    return {
        "diagnosis_accuracy": j.diagnosis_accuracy.score,
        "evidence_grounding": j.evidence_grounding.score,
        "action_appropriateness": j.action_appropriateness.score,
        "postmortem_quality": j.postmortem_quality.score,
        "hallucination_free": j.hallucination_free.score,
        "overall": j.overall,
        "would_trust_unattended": j.would_trust_unattended,
        "justifications": {
            "diagnosis_accuracy": j.diagnosis_accuracy.justification,
            "evidence_grounding": j.evidence_grounding.justification,
            "action_appropriateness": j.action_appropriateness.justification,
            "postmortem_quality": j.postmortem_quality.justification,
            "hallucination_free": j.hallucination_free.justification,
        },
    }

"""Pydantic schemas for structured LLM outputs - the inter-agent contracts.

Every agent is forced to return one of these, so the loop never has to parse free
text. The LLM is bound to the schema via `with_structured_output`, and the same
schema doubles as validation, so a malformed model response fails loudly instead
of silently corrupting loop state.

Two things to notice, because they are the difference between a demo and a system:

1. **Closed enums, not strings.** `RemediationProposal.action_type` is an
   `ActionType`, and `DiagnosisResult.requested_tools` is a list of `TelemetryTool`.
   The model physically cannot name an action or tool that does not exist.

2. **Falsifiable predictions.** `RemediationProposal.expected_signal` states what
   telemetry *should* change if the fix works, and `VerificationResult` is required
   to judge that prediction (`expected_signal_met`). A proposal that cannot be
   wrong cannot be verified, and an agent that cannot be wrong cannot learn.

`SCHEMA_VERSION` is bumped whenever a field changes meaning. Runbook memory
records the version it was written with, so a schema change never silently
poisons retrieval with entries the current agents would misread.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .state import ActionRisk, ActionType, Severity, TelemetryTool

#: Bump on any breaking change to the contracts below.
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------


class TriageResult(BaseModel):
    """Output of the Triage agent: classify and scope the incident."""

    severity: Severity = Field(description="Incident severity, SEV1 worst to SEV4 least")
    summary: str = Field(description="One or two sentence description of the impact")
    blast_radius: str = Field(description="Which users / services are affected")
    affected_service: str = Field(description="Primary service implicated")
    customer_facing: bool = Field(
        default=True,
        description="True if end users are affected right now (drives urgency, not severity)",
    )


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


class Hypothesis(BaseModel):
    cause: str = Field(description="A specific candidate root cause")
    confidence: float = Field(ge=0.0, le=1.0, description="0..1 confidence")
    evidence_needed: str = Field(description="What signal would confirm or refute this")
    #: Which tool would produce that signal. This is the bridge from "I am unsure"
    #: to "run this tool", i.e. from a passive agent to an acting one.
    tool_to_check: TelemetryTool | None = Field(
        default=None, description="The read-only tool that would settle this hypothesis"
    )


class DiagnosisResult(BaseModel):
    """Output of the Diagnostician: rank hypotheses against observed telemetry."""

    hypotheses: list[Hypothesis] = Field(description="Ranked candidate root causes, best first")
    leading_cause: str = Field(description="The single most likely root cause given evidence")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list, description="Concrete telemetry observations cited")
    needs_more_data: bool = Field(
        default=False,
        description="True if confidence is too low to act; the loop will run the requested tools and re-diagnose",
    )
    requested_tools: list[TelemetryTool] = Field(
        default_factory=list,
        description="Read-only tools to run before the next diagnosis. Only meaningful when needs_more_data is true.",
    )
    #: Did a recalled past incident inform this diagnosis? Lets the eval measure
    #: whether episodic memory is actually helping or just inflating the prompt.
    used_recalled_episode: bool = Field(
        default=False, description="True if a recalled past incident informed this diagnosis"
    )

    @field_validator("requested_tools")
    @classmethod
    def _dedupe_tools(cls, v: list[TelemetryTool]) -> list[TelemetryTool]:
        seen: list[TelemetryTool] = []
        for tool in v:
            if tool not in seen:
                seen.append(tool)
        return seen


# ---------------------------------------------------------------------------
# Remediation
# ---------------------------------------------------------------------------


class RemediationProposal(BaseModel):
    """Output of the Remediation agent: a single safe corrective action."""

    action_type: ActionType = Field(description="The corrective action, from the closed menu")
    target: str = Field(description="The service or resource the action touches")
    risk: ActionRisk = Field(description="The agent's own view of the action's blast radius")
    rationale: str = Field(description="Why this action addresses the leading cause")
    expected_signal: str = Field(
        description="The specific telemetry change that would prove this worked (e.g. 'error_rate below 0.01')"
    )
    #: What the agent would do next if this action fails. Forcing a fallback makes
    #: the agent reason about its own failure modes at proposal time.
    fallback_if_ineffective: ActionType | None = Field(
        default=None, description="The next action to try if this one changes nothing"
    )
    reversible: bool = Field(default=True, description="Can this action be undone if it makes things worse?")


# ---------------------------------------------------------------------------
# Critique (adversarial review of a proposal, before the human sees it)
# ---------------------------------------------------------------------------


class CriticVerdict(BaseModel):
    """Output of the Critic agent: challenge the proposal before it is executed.

    The Critic is deliberately adversarial. Its job is not to agree; it is to try
    to find the reason this action is wrong, and to say so *before* an operator is
    asked to approve it. A veto sends the loop back to re-diagnose.
    """

    endorse: bool = Field(description="True only if the action is the right next step given the evidence")
    concerns: list[str] = Field(default_factory=list, description="Specific, concrete objections")
    #: The single strongest reason this could be wrong. Required even when endorsing,
    #: so the Critic can never get away with a content-free "looks good".
    strongest_objection: str = Field(description="The best argument against this action, even if you endorse it")
    safer_alternative: ActionType | None = Field(
        default=None, description="A less invasive action that would also address the cause, if one exists"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this verdict")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """Output of the Verifier: did the system actually recover?"""

    recovered: bool = Field(description="True if telemetry shows the service is healthy again")
    confidence: float = Field(ge=0.0, le=1.0)
    observations: list[str] = Field(default_factory=list)
    #: Judged against the proposal's own prediction, not against a vibe. This is the
    #: falsification step: the action said X would change; did X change?
    expected_signal_met: bool = Field(
        default=False,
        description="True if the specific signal the remediation predicted actually changed as predicted",
    )
    signal_assessment: str = Field(
        default="",
        description="One sentence comparing the predicted signal against what the telemetry actually shows",
    )
    partial_progress: bool = Field(
        default=False,
        description="True if metrics improved but are not yet healthy (a compound fault needing another step)",
    )
    next_step: str = Field(description="resolve | loop | escalate")

    @field_validator("next_step")
    @classmethod
    def _valid_next_step(cls, v: str) -> str:
        allowed = {"resolve", "loop", "escalate"}
        normalized = (v or "").strip().lower()
        if normalized not in allowed:
            raise ValueError(f"next_step must be one of {sorted(allowed)}, got {v!r}")
        return normalized


# ---------------------------------------------------------------------------
# Supervisor (routing)
# ---------------------------------------------------------------------------


class SupervisorDecision(BaseModel):
    """Output of the Supervisor: which agent should act next, and why.

    This is what makes the system multi-agent rather than a fixed pipeline. The
    Supervisor sees the state summary and picks the next specialist. The
    orchestrator still enforces the hard ceilings and the human gate - a router
    that could route around the guardrails would be a security hole, not a feature.
    """

    next_agent: str = Field(description="One of: triage, diagnose, gather_evidence, propose, verify, escalate, resolve")
    reason: str = Field(description="One sentence: why this agent, now")
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)

    @field_validator("next_agent")
    @classmethod
    def _valid_agent(cls, v: str) -> str:
        allowed = {"triage", "diagnose", "gather_evidence", "propose", "verify", "escalate", "resolve"}
        normalized = (v or "").strip().lower()
        if normalized not in allowed:
            raise ValueError(f"next_agent must be one of {sorted(allowed)}, got {v!r}")
        return normalized


# ---------------------------------------------------------------------------
# Postmortem
# ---------------------------------------------------------------------------


class PostmortemResult(BaseModel):
    """Output of the Postmortem agent: a blameless writeup."""

    summary: str = Field(description="What happened, in two or three sentences")
    root_cause: str = Field(description="The confirmed (or best-supported) root cause")
    contributing_factors: list[str] = Field(
        default_factory=list, description="Conditions that made this incident possible or worse"
    )
    recommendations: list[str] = Field(default_factory=list, description="Concrete, actionable prevention items")
    #: A one-line, reusable lesson. This is the text that goes into runbook memory
    #: and is retrieved for future incidents, so it must stand alone.
    runbook_lesson: str = Field(
        default="",
        description="One self-contained sentence a future responder could act on: symptom -> fix",
    )
    detection_gap: str = Field(
        default="", description="Why this was not caught earlier (alerting/testing gap), if identifiable"
    )


# ---------------------------------------------------------------------------
# Evaluation judge (used by evaluation/, not by the running loop)
# ---------------------------------------------------------------------------


class JudgeScore(BaseModel):
    """LLM-as-judge rubric score for one dimension of one incident run."""

    score: int = Field(ge=1, le=5, description="1 = unacceptable, 3 = adequate, 5 = excellent")
    justification: str = Field(description="One or two sentences citing specifics from the run")


class RunJudgement(BaseModel):
    """LLM-as-judge assessment of a whole incident run.

    Outcome metrics (resolved / iterations) say whether the agent got there.
    These say whether it got there *for the right reasons* - which is the part
    that predicts whether it will work on an incident you have not seen.
    """

    diagnosis_accuracy: JudgeScore = Field(description="Did the leading cause match the evidence?")
    evidence_grounding: JudgeScore = Field(
        description="Were cited observations real and present in the telemetry, or invented?"
    )
    action_appropriateness: JudgeScore = Field(description="Was the action the least invasive effective fix?")
    postmortem_quality: JudgeScore = Field(description="Is the writeup blameless, specific, and actionable?")
    hallucination_free: JudgeScore = Field(description="5 = no unsupported claims anywhere in the run")
    overall: int = Field(ge=1, le=5, description="Overall score across the dimensions above")
    would_trust_unattended: bool = Field(
        description="Would you let this agent run this action without a human watching?"
    )

"""Triage agent: classify severity, summarize impact, scope the blast radius.

Runs once, on the first pass. Its job is deliberately narrow: decide how bad this
is and who it affects, and explicitly NOT propose a fix. Splitting "how bad is it"
from "what should we do" keeps severity honest - an agent that decides both at once
tends to inflate severity to justify the action it already wants to take.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..models.schemas import TriageResult
from ..models.state import IncidentState
from ._prompting import incident_context, join_sections, memory_block, system_prompt, telemetry_block

AGENT = "Triage"

_SYSTEM = system_prompt(
    "ROLE: on-call SRE triage.\n"
    "TASK: from the telemetry, classify severity, summarize customer impact in one or two "
    "sentences, and state the blast radius.\n"
    "SEVERITY RUBRIC (apply it literally; do not inflate):\n"
    "  SEV1 - total or near-total failure of a customer-facing path (error_rate above ~0.30, "
    "or a hard dependency fully down).\n"
    "  SEV2 - major degradation: significant errors or latency, customers clearly affected.\n"
    "  SEV3 - minor or partial: elevated latency with few or no errors.\n"
    "  SEV4 - cosmetic or internal only; no customer impact. Use this for a spurious alert "
    "where every real signal is nominal.\n"
    "IMPORTANT: if the metrics look healthy (error_rate near 0, latency near baseline), say so "
    "plainly and classify SEV4 with customer_facing=false. A page is not evidence of a fault.\n"
    "Do NOT propose remediation. That is another agent's job."
)


def run(state: IncidentState, observations: dict, llm: LLMClient) -> TriageResult:
    user = join_sections(
        incident_context(state, include_budget=False),
        memory_block(state),
        telemetry_block(observations),
        "Classify this incident against the rubric.",
    )
    return llm.structured(
        TriageResult,
        _SYSTEM,
        user,
        incident_id=state.incident_id,
        agent=AGENT,
        budget_remaining_usd=state.budget_remaining_usd(),
    )

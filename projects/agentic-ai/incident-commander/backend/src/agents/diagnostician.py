"""Diagnostician agent: form and rank hypotheses, and decide whether it can commit.

This is the reflective core of the system, and the agent with the most interesting
control flow, because it has three legitimate outputs rather than one:

1. **A confident leading cause** -> the loop proceeds to propose an action.
2. **`needs_more_data` plus `requested_tools`** -> the loop runs those read-only
   tools and calls this agent again with the new evidence. This is the difference
   between an agent and a function: it can decide what it needs to know next.
3. **Persistent low confidence** -> the loop escalates to a human rather than
   acting on a guess. `IC_MIN_ACTION_CONFIDENCE` is the threshold.

On every iteration it reconsiders the evidence including the outcome of prior
remediation attempts. That reflection is what lets it converge on a compound fault
instead of re-proposing a cause an earlier action already ruled out.
"""

from __future__ import annotations

from ..llm import LLMClient
from ..models.schemas import DiagnosisResult
from ..models.state import IncidentState
from ..tools.observability import describe_tools
from ._prompting import (
    action_history_block,
    evidence_block,
    incident_context,
    join_sections,
    memory_block,
    system_prompt,
    telemetry_block,
    triage_block,
)

AGENT = "Diagnostician"

_SYSTEM = system_prompt(
    "ROLE: SRE diagnostician.\n"
    "TASK: produce ranked root-cause hypotheses for the active incident, commit to a leading "
    "cause, and cite the specific telemetry that supports it.\n"
    "\n"
    "METHOD:\n"
    "1. List candidate causes, best first, each with a 0..1 confidence and the signal that "
    "would confirm or refute it.\n"
    "2. For each hypothesis, name the tool (tool_to_check) that would settle it.\n"
    "3. Quote concrete numbers in `evidence` - 'error_rate 0.18, up from 0.001 baseline', not "
    "'errors are elevated'. If you cannot quote it, you did not observe it.\n"
    "4. Discriminate between causes that share a symptom. High latency plus a saturated pool "
    "could be a pool that is too small OR a pool that is leaking; those need different fixes, "
    "so check before you commit.\n"
    "\n"
    "WHEN YOU ARE NOT SURE - this is the important part:\n"
    "Set needs_more_data=true and list the specific tools you want in requested_tools. Do NOT "
    "guess. The loop will run those tools and ask you again with the results. Only the tools "
    "listed below are available, and each round of evidence gathering costs budget, so request "
    "the ones that would actually change your answer - not everything.\n"
    "If you have already gathered evidence and are STILL below 0.6 confidence, say so honestly "
    "with needs_more_data=true and an empty requested_tools list. The loop will escalate to a "
    "human, which is the correct outcome. An escalation is a good result; a confident wrong "
    "diagnosis that triggers a production rollback is not.\n"
    "\n"
    "LEARN FROM FAILURE: you are shown what previous actions achieved. If an action addressed "
    "cause X and changed nothing, X is refuted - lower it or drop it. Do not re-propose it.\n"
    "\n"
    "AVAILABLE TOOLS (metrics, logs, traces and deploys are already in your input):\n"
    f"{describe_tools()}"
)


def run(state: IncidentState, observations: dict, llm: LLMClient) -> DiagnosisResult:
    user = join_sections(
        incident_context(state),
        triage_block(state),
        f"Evidence-gathering rounds used so far: {state.evidence_rounds}.",
        memory_block(state),
        action_history_block(state),
        telemetry_block(observations, label="DEFAULT TELEMETRY BUNDLE"),
        evidence_block(state),
        "Diagnose the root cause, or request the specific tools you need.",
    )
    return llm.structured(
        DiagnosisResult,
        _SYSTEM,
        user,
        incident_id=state.incident_id,
        agent=AGENT,
        budget_remaining_usd=state.budget_remaining_usd(),
    )

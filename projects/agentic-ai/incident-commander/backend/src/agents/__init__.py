"""The specialist agents that make up the Incident Commander.

Each agent is a pure function `run(state, ..., llm) -> Schema`. It reads state,
calls the model once through the `llm.structured` seam, and returns a validated
object. It never mutates state, never calls a write-side tool, and never decides
its own permissions - the orchestrator does all three. That uniformity is what
lets the same five functions be driven by either engine (`loop.py` or `graph.py`)
without modification.

| Agent          | Decides                                  | Runs        |
|----------------|------------------------------------------|-------------|
| `supervisor`   | which agent goes next                    | every pass  |
| `triage`       | severity, impact, blast radius           | once        |
| `diagnostician`| root cause, or which tools it needs next | every pass  |
| `remediation`  | the single next action + its prediction  | when confident |
| `critic`       | whether that action is actually right    | before the gate |
| `verifier`     | recovered / loop / escalate              | after acting |
| `postmortem`   | the blameless writeup + runbook lesson   | at the end  |
"""

from . import critic, diagnostician, postmortem, remediation, supervisor, triage, verifier
from ._prompting import PROMPT_VERSION

__all__ = [
    "triage",
    "diagnostician",
    "remediation",
    "critic",
    "verifier",
    "postmortem",
    "supervisor",
    "PROMPT_VERSION",
]

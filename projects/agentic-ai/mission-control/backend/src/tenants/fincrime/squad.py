"""The fincrime squad as a `Workload` - the ordered investigation the runtime drives.

The supervisor's plan is expressed as a linear list of steps: triage -> graph ->
screening -> narrative -> critic -> propose disposition -> commit. The commit step is
side-effecting and therefore HITL-gated by the engine; it runs inside a saga so a
partial commit failure compensates cleanly and the case escalates to a human instead
of being left half-filed.
"""

from __future__ import annotations

from typing import Any

from ...models.schemas import Alert
from ...runtime.saga import Saga, SagaError
from ...runtime.workflow import Step, WorkflowContext, Workload
from . import agents
from .tools import FincrimeTools

# Map a recommended action to the disposition tool the supervisor requests.
_ACTION_TOOL = {"file_sar": "case.file_sar", "escalate": "case.escalate", "close": "case.close"}


def _propose_disposition(ctx: WorkflowContext) -> None:
    """Supervisor decides the disposition and stages it for the HITL gate."""
    if ctx.state.get("force_escalate"):
        action = "escalate"
    else:
        action = ctx.state.get("narrative", {}).get("recommended_action", "escalate")
    ctx.state["proposed_action"] = action
    ctx.state["pending_action"] = {
        "tool": _ACTION_TOOL.get(action, "case.escalate"),
        "args": {"alert_id": ctx.state["alert"]["id"], "amount_usd": ctx.state["alert"]["amount_usd"]},
    }


def _commit_disposition(ctx: WorkflowContext) -> None:
    """Execute the approved disposition under a saga (record -> notify)."""
    action = ctx.state.get("proposed_action", "escalate")
    saga = Saga()
    case_record: dict[str, Any] = {}

    def _record() -> None:
        case_record["filed"] = action
        ctx.state["case_record"] = case_record

    def _unrecord() -> None:
        ctx.state.pop("case_record", None)
        ctx.state["compensated"] = True

    def _notify() -> None:
        if ctx.state.get("inject_commit_fault"):
            raise RuntimeError("downstream case-management system unavailable")
        ctx.state["notified"] = True

    def _unnotify() -> None:
        ctx.state.pop("notified", None)

    try:
        saga.run("record_case", _record, _unrecord)
        saga.run("notify_downstream", _notify, _unnotify)
        ctx.state["disposition"] = action
    except SagaError as exc:
        # Compensated: nothing was left half-committed. Hand to a human.
        from ...models.events import EventType

        ctx.events.append(EventType.SAGA_COMPENSATED, failed_step=exc.failed_step, compensated=saga.compensated)
        ctx.state["disposition"] = "escalate"
        ctx.state["compensated"] = True


class FincrimeSquad(Workload):
    tenant = "fincrime"
    committer = "fincrime.supervisor"

    def seed_state(self, *, alert: Alert, inject_commit_fault: bool = False, **_: Any) -> dict[str, Any]:
        return {
            "alert": alert.model_dump(),
            "_tools": FincrimeTools(),
            "inject_commit_fault": inject_commit_fault,
        }

    def rehydrate(self, state: dict[str, Any]) -> None:
        state["_tools"] = FincrimeTools()

    def steps(self) -> list[Step]:
        return [
            Step("triage", agents.triage),
            Step("graph_investigate", agents.graph_investigate),
            Step("screening", agents.screening),
            Step("narrative", agents.narrative),
            Step("critic", agents.critic),
            Step("propose_disposition", _propose_disposition),
            Step("commit_disposition", _commit_disposition, side_effecting=True),
        ]

"""The runtime engine: durable, checkpointed, human-gated execution.

The engine drives a workload's linear steps and owns the reliability behaviours that
make this a platform rather than a script:

  - Checkpointing: after every step it writes state + completed steps to disk.
  - Durable resume: a decision on a paused case works after a restart. The case is
    rebuilt from its checkpoint and event log; completed steps are never re-run or
    re-billed.
  - Human gate: before any side-effecting step it asks the policy engine. A
    REQUIRE_APPROVAL decision parks the workflow until a reviewer decides.
  - Maker-checker: the person who opened a case may not decide its disposition.
  - Saga: side-effecting commits run under a saga so a partial failure compensates.
  - Dead-letter queue: a workflow that blows a ceiling or dies is parked, never dropped.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

from ..config import Settings
from ..gateway import Gateway
from ..gateway.core import BudgetExceeded
from ..identity import IdentityBroker
from ..models.events import EventLog, EventType
from ..models.schemas import PolicyEffect, WorkflowStatus
from ..observability import Tracer
from ..policy import PolicyEngine
from ..registry import AgentRegistry, PromptRegistry
from ..storage import atomic_write
from .dlq import DeadLetterQueue
from .workflow import PolicyDenied, WorkflowContext, WorkflowResult, Workload

logger = logging.getLogger(__name__)

#: Bumped when the checkpoint layout changes; `_rehydrate` refuses unknown versions
#: rather than resuming a case from a file it cannot read correctly.
CHECKPOINT_VERSION = 2

TERMINAL = {
    WorkflowStatus.RESOLVED,
    WorkflowStatus.ESCALATED,
    WorkflowStatus.FAILED,
    WorkflowStatus.DEAD_LETTERED,
    WorkflowStatus.COMPENSATED,
}


class MakerCheckerViolation(PermissionError):
    """The analyst who opened a case tried to decide it."""


class AlreadyDecided(RuntimeError):
    """A decision arrived for a case a reviewer has already decided."""

    def __init__(self, previous: dict[str, Any]) -> None:
        super().__init__(f"already decided: {previous.get('decision')} by {previous.get('decided_by')}")
        self.previous = previous


class NotAwaitingApproval(RuntimeError):
    """A decision arrived for a case that is not parked at the gate (still running, or done)."""


class _Run:
    """In-memory handle for one live or paused workflow."""

    def __init__(self, workflow_id: str, workload: Workload, ctx: WorkflowContext) -> None:
        self.workflow_id = workflow_id
        self.workload = workload
        self.ctx = ctx
        self.completed: list[str] = []
        self.status = WorkflowStatus.PENDING
        self.pending_action: dict[str, Any] | None = None
        self.approval_granted = False
        self.lock = threading.Lock()  # one decision at a time per case


class Runtime:
    def __init__(
        self,
        settings: Settings,
        gateway: Gateway,
        policy: PolicyEngine,
        identity: IdentityBroker,
        agents: AgentRegistry,
        prompts: PromptRegistry,
        workloads: dict[str, Callable[[], Workload]],
    ) -> None:
        self.settings = settings
        self.gateway = gateway
        self.policy = policy
        self.identity = identity
        self.agents = agents
        self.prompts = prompts
        self.workloads = workloads  # tenant name -> factory, for rebuilding after a restart
        self.dlq = DeadLetterQueue(settings.data_dir)
        self._ckpt_dir = settings.data_dir / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, _Run] = {}
        self._runs_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------- #
    def open(
        self,
        workflow_id: str,
        workload: Workload,
        *,
        actor: str,
        prompt_label: str = "production",
        **seed_kwargs: Any,
    ) -> _Run:
        """Register a workflow and write its first checkpoint, without running a step.

        Split from `advance` so the API can hand the id back at once and run the steps
        on a worker thread while the console watches the event stream.
        """
        events = EventLog(workflow_id, self.settings.data_dir)
        state = workload.seed_state(**seed_kwargs)
        state["prompt_label"] = prompt_label
        state["audit"] = {"started_by": actor, "started_at": time.time()}
        run = _Run(workflow_id, workload, self._context(workflow_id, events, state))
        with self._runs_lock:
            self._runs[workflow_id] = run
        events.append(
            EventType.WORKFLOW_STARTED,
            tenant=workload.tenant,
            started_by=actor,
            **{k: str(v) for k, v in seed_kwargs.items()},
        )
        run.status = WorkflowStatus.RUNNING
        self._checkpoint(run)
        return run

    def advance(self, workflow_id: str) -> WorkflowResult:
        run = self._runs[workflow_id]
        try:
            return self._advance(run)
        except Exception as exc:  # noqa: BLE001 - nothing may leave a case silently stuck
            logger.exception("workflow %s crashed", workflow_id)
            try:
                return self._dead_letter(run, f"runtime crash: {exc}")
            except Exception:  # noqa: BLE001 - even the disk failed: say so in memory
                logger.exception("could not dead-letter %s", workflow_id)
                run.status = WorkflowStatus.FAILED
                run.ctx.state["dead_letter_reason"] = f"runtime crash, and the checkpoint could not be written: {exc}"
                return self._result(run)

    def start(self, workflow_id: str, workload: Workload, *, actor: str, **seed_kwargs: Any) -> WorkflowResult:
        """Open and run to the first pause or a terminal state (CLI, tests, eval)."""
        self.open(workflow_id, workload, actor=actor, **seed_kwargs)
        return self.advance(workflow_id)

    def resume_after_approval(
        self, workflow_id: str, decision: str, note: str | None = None, *, actor: str
    ) -> WorkflowResult:
        """Apply a reviewer's decision to a paused workflow and continue (or stop)."""
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be 'approve' or 'reject'")
        run = self._runs.get(workflow_id) or self._rehydrate(workflow_id)
        with run.lock:
            audit = run.ctx.state.setdefault("audit", {})
            if audit.get("decision"):
                raise AlreadyDecided(dict(audit))
            if run.status != WorkflowStatus.AWAITING_APPROVAL:
                raise NotAwaitingApproval(f"workflow {workflow_id} is {run.status.value}, not awaiting approval")
            if actor == audit.get("started_by"):
                run.ctx.events.append(EventType.MAKER_CHECKER_BLOCKED, actor=actor, decision=decision)
                raise MakerCheckerViolation(
                    f"{actor} opened this case, so a different reviewer must decide it (maker-checker)"
                )
            audit.update(decision=decision, decided_by=actor, decided_at=time.time(), note=note or "")
            if decision == "approve":
                run.approval_granted = True
                run.status = WorkflowStatus.RUNNING
                run.pending_action = None
                run.ctx.events.append(EventType.APPROVAL_GRANTED, decided_by=actor, note=note or "")
                return self._advance(run)
            # Rejection: the disposition is not committed; the case goes to a person.
            run.ctx.events.append(EventType.APPROVAL_REJECTED, decided_by=actor, note=note or "")
            run.status = WorkflowStatus.ESCALATED
            run.pending_action = None
            run.ctx.state["disposition"] = "escalate"
            self._checkpoint(run)
            return self._result(run)

    # -- the step loop ------------------------------------------------------ #
    def _advance(self, run: _Run) -> WorkflowResult:
        began = time.monotonic()  # time spent waiting for a human never counts
        for step in run.workload.steps():
            if step.name in run.completed:
                continue  # resumed: already done, do not re-run or re-bill
            if time.monotonic() - began > self.settings.max_wallclock_s:
                return self._dead_letter(run, f"wall-clock ceiling of {self.settings.max_wallclock_s:.0f}s exceeded")

            if step.side_effecting and not self._clear_to_commit(run):
                return self._result(run)  # paused at the gate, or dead-lettered by policy

            run.ctx.events.append(EventType.STEP_STARTED, step=step.name)
            try:
                with run.ctx.tracer.span(f"step:{step.name}", step=step.name):
                    step.fn(run.ctx)
            except BudgetExceeded as exc:
                return self._dead_letter(run, f"budget exceeded: {exc}")
            except PolicyDenied as exc:
                return self._dead_letter(run, f"policy denied: {exc}")
            except Exception as exc:  # noqa: BLE001 - one retry, then dead-letter
                run.ctx.events.append(EventType.RETRY, step=step.name, error=str(exc)[:300])
                try:
                    with run.ctx.tracer.span(f"step:{step.name}:retry", step=step.name):
                        step.fn(run.ctx)
                except BudgetExceeded as exc2:
                    return self._dead_letter(run, f"budget exceeded: {exc2}")
                except Exception as exc2:  # noqa: BLE001
                    return self._dead_letter(run, f"step '{step.name}' failed after retry: {exc2}")

            run.completed.append(step.name)
            run.ctx.events.append(EventType.STEP_COMPLETED, step=step.name)
            self._checkpoint(run)
            if len(run.completed) > self.settings.max_workflow_steps:
                return self._dead_letter(run, "exceeded max workflow steps")

        disposition = run.ctx.state.get("disposition")
        if run.ctx.state.get("compensated"):
            run.status = WorkflowStatus.COMPENSATED
        elif disposition == "escalate":
            run.status = WorkflowStatus.ESCALATED
        else:
            run.status = WorkflowStatus.RESOLVED
        run.ctx.events.append(EventType.WORKFLOW_COMPLETED, disposition=disposition or "")
        self._checkpoint(run)
        return self._result(run)

    def _clear_to_commit(self, run: _Run) -> bool:
        """Evaluate the policy/approval gate for a pending side-effecting action."""
        pending = run.ctx.state.get("pending_action") or {}
        tool = pending.get("tool", "")
        args = pending.get("args", {})
        card = self.agents.get(run.workload.committer)
        decision = self.policy.evaluate(card, tool, args)
        if decision.effect == PolicyEffect.DENY:
            self._dead_letter(run, f"policy denied commit: {decision.reason}")
            return False
        needs_human = decision.effect == PolicyEffect.REQUIRE_APPROVAL and not self.settings.auto_approve
        if needs_human:
            if not run.approval_granted:
                run.status = WorkflowStatus.AWAITING_APPROVAL
                run.pending_action = pending
                run.ctx.events.append(
                    EventType.APPROVAL_REQUESTED, tool=tool, rule=decision.rule, reason=decision.reason
                )
                self._checkpoint(run)
                return False
            # An approval is single-use: it clears exactly the action the reviewer saw.
            run.approval_granted = False
        elif decision.effect == PolicyEffect.REQUIRE_APPROVAL:
            # auto_approve stood in for a person. Say so on the record.
            run.ctx.state["audit"].update(decision="approve", decided_by="policy:auto_approve", decided_at=time.time())
        return True

    # -- terminal + persistence -------------------------------------------- #
    def _dead_letter(self, run: _Run, reason: str) -> WorkflowResult:
        run.status = WorkflowStatus.DEAD_LETTERED
        run.pending_action = None
        run.ctx.state["dead_letter_reason"] = reason
        run.ctx.events.append(EventType.DEAD_LETTERED, reason=reason)
        self.dlq.put(run.workflow_id, reason, _public(run.ctx.state))
        self._checkpoint(run)
        return self._result(run)

    def _checkpoint(self, run: _Run) -> None:
        run.ctx.events.append(EventType.CHECKPOINT_WRITTEN, completed=list(run.completed), status=run.status.value)
        record = {
            "version": CHECKPOINT_VERSION,
            "workflow_id": run.workflow_id,
            "tenant": run.workload.tenant,
            "status": run.status.value,
            "completed": run.completed,
            "pending_action": run.pending_action,
            "state": _public(run.ctx.state),
            "cost_usd": self.gateway.cost_of(run.workflow_id),
            "updated_at": time.time(),
        }
        # Write-then-rename: a crash mid-write leaves the previous checkpoint intact,
        # never a half-written file that cannot be resumed.
        atomic_write(self._ckpt_dir / f"{run.workflow_id}.json", json.dumps(record, default=str))

    def _rehydrate(self, workflow_id: str) -> _Run:
        """Rebuild a parked workflow from disk after a restart."""
        with self._runs_lock:
            if workflow_id in self._runs:
                return self._runs[workflow_id]
            ckpt = self.load_checkpoint(workflow_id)
            if ckpt is None:
                raise KeyError(f"unknown workflow {workflow_id}")
            if ckpt.get("version") != CHECKPOINT_VERSION:
                raise RuntimeError(f"checkpoint {workflow_id} has unsupported version {ckpt.get('version')}")
            factory = self.workloads.get(ckpt["tenant"])
            if factory is None:
                raise RuntimeError(f"no workload registered for tenant {ckpt['tenant']}")
            workload = factory()
            events = EventLog(workflow_id, self.settings.data_dir)
            events.replay()  # continue the existing log, keeping sequence numbers
            state = dict(ckpt["state"])
            workload.rehydrate(state)
            run = _Run(workflow_id, workload, self._context(workflow_id, events, state))
            run.completed = list(ckpt.get("completed", []))
            run.status = WorkflowStatus(ckpt["status"])
            run.pending_action = ckpt.get("pending_action")
            self.gateway.restore_cost(workflow_id, float(ckpt.get("cost_usd", 0.0)))
            self._runs[workflow_id] = run
        events.append(EventType.WORKFLOW_RESUMED, from_checkpoint=True, status=run.status.value)
        return run

    def _context(self, workflow_id: str, events: EventLog, state: dict[str, Any]) -> WorkflowContext:
        return WorkflowContext(
            workflow_id=workflow_id,
            gateway=self.gateway,
            policy=self.policy,
            identity=self.identity,
            agents=self.agents,
            prompts=self.prompts,
            events=events,
            tracer=Tracer(workflow_id, self.settings.data_dir),
            state=state,
        )

    def _result(self, run: _Run) -> WorkflowResult:
        return WorkflowResult(
            status=run.status.value,
            disposition=run.ctx.state.get("disposition"),
            state=run.ctx.state,
            pending_action=run.pending_action if run.status == WorkflowStatus.AWAITING_APPROVAL else None,
        )

    # -- views -------------------------------------------------------------- #
    def get_run(self, workflow_id: str) -> _Run | None:
        return self._runs.get(workflow_id)

    def cost_of(self, workflow_id: str) -> float:
        return self.gateway.cost_of(workflow_id)

    def load_checkpoint(self, workflow_id: str) -> dict[str, Any] | None:
        if not _safe_id(workflow_id):
            return None
        path = self._ckpt_dir / f"{workflow_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_checkpoints(self, limit: int = 50) -> list[dict[str, Any]]:
        files = sorted(self._ckpt_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for path in files[:limit]:
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):  # pragma: no cover - a torn file is skipped, not fatal
                continue
        return out


def _public(state: dict[str, Any]) -> dict[str, Any]:
    """State minus in-process objects ('_'-prefixed keys such as tool instances)."""
    return {k: v for k, v in state.items() if not k.startswith("_")}


def _safe_id(workflow_id: str) -> bool:
    """Workflow ids become file names: refuse anything that could leave the directory."""
    return bool(workflow_id) and all(c.isalnum() or c in "-_" for c in workflow_id) and len(workflow_id) <= 80

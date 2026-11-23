"""The Incident Commander loop engine - the heart of the system.

Two classes:

* `IncidentManager` owns every incident: starting one, the live event stream,
  and the human approval gate. It does not decide anything about an incident.
* `IncidentRun` is ONE incident, from detection to postmortem. Read
  `IncidentRun.run_loop()` first: it is the whole agent loop on one screen, and
  every step it calls is a short method below it.

One pass of the loop:

    check ceilings -> OBSERVE -> ROUTE -> [TRIAGE] -> DIAGNOSE (+ evidence)
      -> PROPOSE -> GUARDRAILS -> CRITIC -> HUMAN GATE -> ACT -> VERIFY

## Halt conditions (no runaway agents)

Six, all enforced here rather than requested in a prompt:

  * iteration ceiling    `IC_MAX_ITERATIONS`
  * cost ceiling         `IC_MAX_COST_USD`, checked at the top of every pass AND
                         pre-flight before every individual model call
  * wall-clock ceiling   `IC_MAX_WALLCLOCK_S`
  * human rejection      the operator rejects a proposed action
  * approval timeout     `IC_APPROVAL_TIMEOUT_S` - an operator who closes the tab
                         must not strand the incident forever
  * operator cancel      checked every pass, before the gate, and before acting

Plus two soft halts that escalate rather than act: persistent low diagnostic
confidence, and repeated Critic vetoes.

## The one rule that makes this safe

The LLM chooses *what to do next*. This module decides *what is allowed*. Routing,
diagnosis and action selection are model decisions; ceilings, RBAC, the repeat-
action block and the human gate are code. Nothing a model emits - and therefore
nothing an attacker can write into a log line - can widen its own permissions.

Every meaningful transition is emitted as an `IncidentEvent`, which is what the
React console renders live over SSE. State is one typed `IncidentState` object, so
the whole run is auditable and replayable.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from .agents import PROMPT_VERSION, critic, diagnostician, postmortem, remediation, supervisor, triage, verifier
from .config import get_settings
from .guardrails.manager import GuardrailManager
from .llm import LLMBudgetError, LLMConfigError, LLMUnavailableError, get_llm, token_tracker
from .memory.runbook import episode_from_state, get_memory
from .models.schemas import DiagnosisResult, RemediationProposal
from .models.state import (
    ActionStatus,
    AgentName,
    EventType,
    HaltReason,
    IncidentEvent,
    IncidentState,
    IncidentStatus,
    RemediationAction,
    Severity,
    TelemetryTool,
)
from .sandbox.scenarios import get_scenario
from .sandbox.stack import SandboxStack
from .tools.observability import observe_all, precondition_facts, run_tools
from .tools.remediation import execute_remediation

logger = logging.getLogger(__name__)

_TERMINAL = {
    IncidentStatus.RESOLVED,
    IncidentStatus.ESCALATED,
    IncidentStatus.CANCELLED,
    IncidentStatus.FAILED,
}

#: Sentinel pushed to a subscriber queue to close its SSE stream.
_STREAM_CLOSED = object()

#: Event types after which the whole incident row is rewritten. These are the
#: transitions that change something a list view or a dashboard would show.
_PERSIST_ON = {
    EventType.STATUS,
    EventType.LOOP,
    EventType.ACTION_EXECUTED,
    EventType.VERIFICATION,
    EventType.RESOLVED,
    EventType.ESCALATED,
    EventType.CANCELLED,
    EventType.POSTMORTEM,
    EventType.ERROR,
}


# =============================================================================
# IncidentManager: lifecycle, live stream, approval gate
# =============================================================================


class IncidentManager:
    """Owns all incidents, their sandboxes, subscribers, and approval futures."""

    def __init__(self, llm=None) -> None:
        # `llm` lets tests and the eval harness inject a client. In the running app
        # it is None and `get_llm()` picks Gemini or the offline agent lazily, on
        # the first incident (so importing the app never requires a key).
        self._llm_override = llm
        self.incidents: dict[str, IncidentState] = {}
        self.stacks: dict[str, SandboxStack] = {}
        self.guards: dict[str, GuardrailManager] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._approvals: dict[str, asyncio.Future] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._seq: dict[str, int] = {}
        #: action_id -> the decision already recorded, so a duplicate approval POST
        #: is idempotent rather than a 409 race.
        self._approval_results: dict[str, dict] = {}
        #: Optional persistence backend (src.db.repository), injected at startup
        #: when a database is configured. None means in-memory only, which is the
        #: default and a fully supported way to run this.
        self._repo = None

    def set_persistence(self, repo) -> None:
        """Attach a repository so incidents and events survive a restart."""
        self._repo = repo

    @property
    def settings(self):
        """Read settings live, so tests and config reloads are never stale."""
        return get_settings()

    def _resolve_llm(self):
        """The injected client (tests, eval) or the app's client (Gemini or offline).
        Raises LLMConfigError when Gemini is forced and no key is configured."""
        return self._llm_override if self._llm_override is not None else get_llm()

    # --- lifecycle ----------------------------------------------------------

    def trigger(
        self, scenario_id: str, service: str | None, auto_approve: bool | None, *, actor: str | None = None
    ) -> IncidentState:
        scenario = get_scenario(scenario_id)
        if scenario is None:
            raise ValueError(f"unknown scenario '{scenario_id}'")

        settings = self.settings
        svc = service or scenario.default_service
        state = IncidentState(
            scenario_id=scenario_id,
            service=svc,
            severity=Severity(scenario.severity),
            max_iterations=settings.max_iterations,
            max_cost_usd=settings.max_cost_usd,
            max_wallclock_s=settings.max_wallclock_s,
            auto_approve=settings.auto_approve if auto_approve is None else auto_approve,
            engine="graph" if settings.use_graph_engine else "loop",
            llm_mode="injected" if self._llm_override is not None else settings.llm_mode,
            created_by=actor,
        )
        iid = state.incident_id
        self.incidents[iid] = state
        self.stacks[iid] = SandboxStack(scenario=scenario, service=svc)
        self.guards[iid] = GuardrailManager(
            role=settings.agent_role,
            auto_approve=state.auto_approve,
            quarantine=settings.quarantine_injections,
            redact=settings.redact_sensitive,
        )
        self._subscribers[iid] = set()
        self._seq[iid] = 0
        self._tasks[iid] = asyncio.create_task(IncidentRun(self, state).run())
        self._evict_old_incidents()
        return state

    def get(self, incident_id: str) -> IncidentState | None:
        return self.incidents.get(incident_id)

    def list(self) -> list[dict]:
        return [s.to_summary() for s in sorted(self.incidents.values(), key=lambda x: x.started_at, reverse=True)]

    def request_cancel(self, incident_id: str, actor: str | None = None) -> bool:
        """Ask a running incident to stop at its next checkpoint.

        Cooperative rather than a hard task cancel: an incident that is mid-action
        should finish that action and write a postmortem, not vanish. If it is
        parked at the approval gate, the pending future is resolved so it unblocks
        immediately instead of waiting out the approval timeout.
        """
        state = self.incidents.get(incident_id)
        if state is None or state.status in _TERMINAL:
            return False
        state.cancel_requested = True
        state.cancelled_by = actor
        for action in state.actions:
            fut = self._approvals.get(action.action_id)
            if fut is not None and not fut.done():
                fut.set_result({"decision": "reject", "note": "incident cancelled by operator", "actor": actor})
        return True

    def _evict_old_incidents(self) -> None:
        """Bound in-memory retention: only terminal incidents, oldest first."""
        cap = self.settings.max_retained_incidents
        if cap <= 0 or len(self.incidents) <= cap:
            return
        terminal = sorted(
            (s for s in self.incidents.values() if s.status in _TERMINAL),
            key=lambda s: s.ended_at or s.started_at,
        )
        for state in terminal[: len(self.incidents) - cap]:
            iid = state.incident_id
            for registry in (self.incidents, self.stacks, self.guards, self._subscribers, self._tasks, self._seq):
                registry.pop(iid, None)
            token_tracker.forget(iid)
            for action in state.actions:
                self._approval_results.pop(action.action_id, None)
            logger.info("evicted incident %s (retention cap %d)", iid, cap)

    # --- pub / sub for SSE --------------------------------------------------

    async def emit(self, event: IncidentEvent) -> None:
        state = self.incidents.get(event.incident_id)
        if state is not None:
            self._seq[event.incident_id] = self._seq.get(event.incident_id, 0) + 1
            event.seq = self._seq[event.incident_id]
            state.events.append(event)
        for q in list(self._subscribers.get(event.incident_id, set())):
            await q.put(event)
        await self._persist(state, event)

    async def _persist(self, state: IncidentState | None, event: IncidentEvent) -> None:
        """Mirror the event (and, on transitions, the incident) into the database.

        Runs AFTER subscribers are notified: the live console must not wait on a
        database write, and a storage outage must not stall the loop.
        """
        if self._repo is None or state is None:
            return
        try:
            await self._repo.save_event(event.incident_id, event.model_dump(mode="json"))
            if event.type in _PERSIST_ON:
                await self._repo.save_incident(state.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001 - never let storage kill a run
            logger.warning("persistence skipped for %s: %s", event.incident_id, exc)

    async def subscribe(self, incident_id: str, after_seq: int = 0):
        """Async generator: replay history from `after_seq`, then stream live.

        `after_seq` is the SSE `Last-Event-ID`, so a reconnecting client resumes
        exactly where it dropped instead of re-rendering the whole timeline.
        """
        state = self.incidents.get(incident_id)
        if state is None:
            return
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(incident_id, set()).add(q)
        try:
            for ev in list(state.events):
                if ev.seq > after_seq:
                    yield ev
            # Finished runs send nothing more. `is_finalized`, not "has a
            # postmortem": a run that ended on an error must still close a late
            # viewer's stream instead of leaving it open forever.
            if state.is_finalized():
                return
            while True:
                ev = await q.get()
                if ev is None or ev is _STREAM_CLOSED:
                    return
                yield ev
        finally:
            self._subscribers.get(incident_id, set()).discard(q)

    # --- human-in-the-loop approval ----------------------------------------

    def resolve_approval(
        self, incident_id: str, action_id: str, decision: str, note: str = "", actor: str | None = None
    ) -> bool:
        state = self.incidents.get(incident_id)
        # The action must belong to the incident in the URL: futures are keyed by
        # action_id alone, and approvals are the security-sensitive endpoint.
        if state is None or all(a.action_id != action_id for a in state.actions):
            return False
        result = {"decision": decision, "note": note, "actor": actor}
        fut = self._approvals.get(action_id)
        if fut is not None and not fut.done():
            self._approval_results[action_id] = result
            fut.set_result(result)
            return True
        at_the_gate = state.status == IncidentStatus.AWAITING_APPROVAL and state.actions[-1].action_id == action_id
        if fut is None and action_id not in self._approval_results and at_the_gate:
            # Early decision: the gate became visible a beat before the run parked
            # on it (the graph engine flips status inside a graph step). Record it
            # and let `await_approval` collect it the moment it parks. Only for the
            # action actually at the gate: an approval sent while the Critic is
            # still reviewing it would be an approval of something nobody has seen.
            self._approval_results[action_id] = result
            return True
        return False

    def approval_already_recorded(self, action_id: str) -> dict | None:
        """The decision previously recorded for an action, if any (idempotency)."""
        return self._approval_results.get(action_id)

    async def await_approval(self, action_id: str, timeout_s: float) -> dict:
        """Block at the human gate, with a timeout.

        The timeout is the important part. A bare `await future` means an operator
        who closes their laptop pins the incident in memory forever. Timing out and
        escalating is the safe default and the honest one: no human approved.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        early = self._approval_results.get(action_id)
        if early is not None:
            fut.set_result(early)
        self._approvals[action_id] = fut
        try:
            if timeout_s and timeout_s > 0:
                return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout_s)
            return await fut
        except asyncio.TimeoutError:
            return {
                "decision": "timeout",
                "note": f"no operator response within {timeout_s:.0f}s",
                "actor": "nobody:timeout",
            }
        finally:
            self._approvals.pop(action_id, None)


# =============================================================================
# IncidentRun: one incident, start to finish
# =============================================================================


class IncidentRun:
    """Runs one incident. Created by `IncidentManager.trigger`, used once."""

    def __init__(self, manager: IncidentManager, state: IncidentState) -> None:
        self.manager = manager
        self.state = state
        self.stack = manager.stacks[state.incident_id]
        self.guard = manager.guards[state.incident_id]
        self.settings = manager.settings
        self.llm = None
        #: The diagnosis the next proposal is based on.
        self.diagnosis: DiagnosisResult | None = None
        self.critic_vetoes = 0
        #: Set when a proposal is vetoed or blocked. The next pass MUST re-diagnose
        #: (with the rejection in the action history) instead of re-proposing from
        #: the same diagnosis, which would only produce the same proposal again.
        self.rediagnose = False
        self._seen_findings = 0

    # --- plumbing -----------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.state.status in _TERMINAL

    async def emit(self, etype: EventType, agent: AgentName | None = None, **payload: Any) -> None:
        await self.manager.emit(
            IncidentEvent(
                type=etype,
                incident_id=self.state.incident_id,
                iteration=self.state.iterations,
                agent=agent,
                payload=payload,
            )
        )

    def sync_usage(self) -> None:
        """Refresh cost from the ledger without emitting.

        Called after every agent so the pre-flight budget check in `llm.structured`
        sees a current figure; refreshing only once per pass let one pass overshoot
        the ceiling before anyone noticed.
        """
        s = self.state
        s.total_tokens, s.total_cost_usd = token_tracker.incident_cost(s.incident_id)

    async def emit_usage(self) -> None:
        s = self.state
        self.sync_usage()
        s.agent_usage = token_tracker.incident_breakdown(s.incident_id)
        await self.emit(
            EventType.TOKEN_USAGE,
            tokens=s.total_tokens,
            cost_usd=round(s.total_cost_usd, 6),
            budget_usd=s.max_cost_usd,
            budget_used_pct=round(100 * s.total_cost_usd / s.max_cost_usd, 1) if s.max_cost_usd else 0.0,
            by_agent=s.agent_usage,
            iteration=s.iterations,
        )

    async def drain_findings(self) -> None:
        """Emit any guardrail findings (injection, redaction, blocks) not yet shown."""
        findings = self.guard.findings_as_dicts()
        for finding in findings[self._seen_findings :]:
            self.state.guardrail_flags.append(finding)
            await self.emit(
                EventType.GUARDRAIL,
                AgentName.ORCHESTRATOR,
                kind=finding["kind"],
                message=finding["detail"],
                patterns=finding.get("patterns", []),
                action_type=finding.get("action_type"),
            )
        self._seen_findings = len(findings)

    async def observe(self) -> dict:
        """Read-only telemetry, sanitized (quarantine + redaction) before any agent sees it."""
        observations = self.guard.sanitize(observe_all(self.stack)).observations
        await self.drain_findings()
        return observations

    async def halt(self, reason: HaltReason, status: IncidentStatus, message: str, agent=AgentName.ORCHESTRATOR):
        self.state.halt_reason = reason
        self.state.status = status
        etype = {IncidentStatus.RESOLVED: EventType.RESOLVED, IncidentStatus.CANCELLED: EventType.CANCELLED}.get(
            status, EventType.ESCALATED
        )
        await self.emit(etype, agent, reason=reason.value, message=message, iterations=self.state.iterations)

    async def stop_if_cancelled(self) -> bool:
        """Honour an operator cancel right now. Checked before every point of no return."""
        if not self.state.cancel_requested:
            return False
        await self.halt(HaltReason.OPERATOR_CANCELLED, IncidentStatus.CANCELLED, "incident cancelled by the operator")
        return True

    async def run_agent(self, fn, *args):
        """Run one (blocking) agent call off the event loop, then refresh the cost."""
        result = await asyncio.to_thread(fn, *args)
        self.sync_usage()
        return result

    # --- top level ----------------------------------------------------------

    async def run(self) -> None:
        s = self.state
        try:
            # Resolved inside the try so a missing-key failure is reported as an
            # error event (and the stream is closed) instead of crashing the task.
            self.llm = self.manager._resolve_llm()
            await self.emit(
                EventType.STATUS,
                AgentName.ORCHESTRATOR,
                status="detected",
                service=s.service,
                engine=s.engine,
                llm_mode=s.llm_mode,
                prompt_version=PROMPT_VERSION,
                role=self.guard.role,
                max_iterations=s.max_iterations,
                max_cost_usd=s.max_cost_usd,
            )
            await self.recall_memory()
            if self.settings.use_graph_engine:
                await self.run_graph()  # same agents and guardrails; see graph.py
            else:
                await self.run_loop()
            await self.finish()
        except LLMConfigError as exc:
            logger.error("incident aborted: %s", exc)
            s.status, s.halt_reason = IncidentStatus.FAILED, HaltReason.LLM_UNAVAILABLE
            await self.emit(EventType.ERROR, AgentName.ORCHESTRATOR, kind="llm_not_configured", message=str(exc))
            await self.finish(remember=False)
        except (LLMUnavailableError, LLMBudgetError) as exc:
            # The model is down, or the next call would breach the cost ceiling.
            # Both mean "stop and get a human", not "crash".
            logger.error("incident halted: %s", exc)
            s.status = IncidentStatus.ESCALATED
            s.halt_reason = HaltReason.COST_CEILING if isinstance(exc, LLMBudgetError) else HaltReason.LLM_UNAVAILABLE
            await self.emit(EventType.ESCALATED, AgentName.ORCHESTRATOR, reason=s.halt_reason.value, message=str(exc))
            await self.finish(remember=s.halt_reason is HaltReason.COST_CEILING)
        except Exception as exc:  # noqa: BLE001 - surface failures, never swallow
            logger.exception("incident loop crashed: %s", exc)
            s.status, s.halt_reason = IncidentStatus.FAILED, HaltReason.CRASH
            await self.emit(
                EventType.ERROR,
                AgentName.ORCHESTRATOR,
                kind="loop_crash",
                message=f"Loop crashed: {exc}",
                detail=str(exc),
            )
            # A crash is exactly the run a postmortem is for.
            with contextlib.suppress(Exception):
                await self.finish(remember=False)
        finally:
            if s.ended_at is None:
                s.ended_at = time.time()
            for q in list(self.manager._subscribers.get(s.incident_id, set())):
                await q.put(None)  # close every open SSE stream

    async def finish(self, *, remember: bool = True) -> None:
        """Postmortem (always, on every outcome), memory, final usage and status.

        `remember=False` for runs that ended because the model or the code failed:
        an outage teaches nothing about the fault, so it must not become an
        episode that is recalled later as experience.
        """
        s = self.state
        await self.write_postmortem()
        if remember:
            await self.record_memory()
        s.ended_at = time.time()
        await self.emit_usage()
        await self.emit(
            EventType.STATUS,
            AgentName.ORCHESTRATOR,
            status=s.status.value,
            halt_reason=s.halt_reason.value if s.halt_reason else None,
            duration_s=round(s.elapsed_s(), 2),
            final=True,
        )

    # --- the explicit loop --------------------------------------------------

    async def run_loop(self) -> None:
        """The agent loop. Each step returns False when this pass should end early."""
        while not self.finished:
            if await self.check_ceilings():
                return
            self.state.iterations += 1
            await self.emit(EventType.LOOP, AgentName.ORCHESTRATOR, iteration=self.state.iterations)

            observations = await self.observe()
            route = await self.route()
            if route == "escalate":
                await self.halt(
                    HaltReason.LOW_CONFIDENCE,
                    IncidentStatus.ESCALATED,
                    "the agent could not reach an actionable diagnosis; escalating to a human",
                    AgentName.SUPERVISOR,
                )
                return
            if route == "triage":
                await self.triage(observations)
                route = "diagnose"  # triage never ends a pass on its own
            if route == "gather_evidence" and self.diagnosis is not None:
                await self.gather_evidence(self.diagnosis)
                route = "diagnose"
            if route == "diagnose":
                observations = await self.diagnose(observations)
                if self.finished:
                    return
            if self.diagnosis is None:  # pragma: no cover - defensive
                continue

            action, proposal = await self.propose(observations)
            if not await self.passes_guardrails(action):
                continue
            if not await self.passes_critic(action, proposal, observations):
                continue
            if not await self.passes_human_gate(action):
                return
            if await self.stop_if_cancelled():  # last check before the world changes
                return
            await self.act(action)
            await self.verify(action)

    async def check_ceilings(self) -> bool:
        """Halt conditions, checked at the top of every pass."""
        self.sync_usage()
        tripped = self.state.exhausted_reason()
        if tripped is None:
            return False
        if tripped is HaltReason.OPERATOR_CANCELLED:
            await self.halt(tripped, IncidentStatus.CANCELLED, "incident cancelled by the operator")
        else:
            await self.halt(
                tripped, IncidentStatus.ESCALATED, f"halted on {tripped.value.replace('_', ' ')}; handing to a human"
            )
        return True

    def _policy_route(self) -> str:
        return supervisor.default_route(
            self.state,
            min_confidence=self.settings.min_action_confidence,
            max_evidence_rounds=self.settings.max_evidence_rounds,
        )

    async def route(self) -> str:
        """Pick the next specialist, then clamp the choice to something legal.

        The LLM Supervisor is advisory. `sanitize_route` is the authority, and it
        falls back to the deterministic policy whenever the model picks a route
        that is not currently valid.
        """
        if self.rediagnose:
            self.rediagnose = False
            await self.emit(
                EventType.ROUTE,
                AgentName.SUPERVISOR,
                route="diagnose",
                reason="the last proposal was rejected; re-diagnosing with the objection in history",
                llm=False,
            )
            return "diagnose"
        fallback = self._policy_route()
        if not self.settings.supervisor_enabled:
            await self.emit(
                EventType.ROUTE, AgentName.SUPERVISOR, route=fallback, reason="deterministic policy", llm=False
            )
            return fallback
        try:
            decision = await self.run_agent(
                lambda: supervisor.run(
                    self.state,
                    self.llm,
                    min_confidence=self.settings.min_action_confidence,
                    max_evidence_rounds=self.settings.max_evidence_rounds,
                )
            )
            chosen = supervisor.sanitize_route(
                decision.next_agent,
                self.state,
                min_confidence=self.settings.min_action_confidence,
                max_evidence_rounds=self.settings.max_evidence_rounds,
            )
            await self.emit(
                EventType.ROUTE,
                AgentName.SUPERVISOR,
                route=chosen,
                requested=decision.next_agent,
                overridden=chosen != decision.next_agent,
                reason=decision.reason,
                confidence=decision.confidence,
                llm=True,
            )
            return chosen
        except (LLMBudgetError, LLMUnavailableError) as exc:
            # Routing is the least important place to spend the last of the budget.
            await self.emit(
                EventType.ROUTE,
                AgentName.SUPERVISOR,
                route=fallback,
                reason=f"supervisor unavailable ({exc}); using deterministic policy",
                llm=False,
                degraded=True,
            )
            return fallback

    async def triage(self, observations: dict) -> None:
        s = self.state
        s.status = IncidentStatus.TRIAGING
        await self.emit(EventType.STEP_START, AgentName.TRIAGE)
        tri = await self.run_agent(triage.run, s, observations, self.llm)
        s.triage_summary, s.severity = tri.summary, tri.severity
        await self.emit(
            EventType.STEP_END,
            AgentName.TRIAGE,
            name=f"{tri.severity.value}: {tri.summary}",
            message=tri.blast_radius,
            severity=tri.severity.value,
            summary=tri.summary,
            blast_radius=tri.blast_radius,
            customer_facing=tri.customer_facing,
        )

    async def diagnose(self, observations: dict) -> dict:
        """Diagnose; if the Diagnostician asks for tools, run them and diagnose again.

        Returns the freshest observations. Escalates (and the pass ends) when
        confidence stays below the floor after the allowed evidence rounds.
        """
        s = self.state
        s.status = IncidentStatus.DIAGNOSING
        await self.emit(EventType.STEP_START, AgentName.DIAGNOSTICIAN)
        self.diagnosis = await self.run_agent(diagnostician.run, s, observations, self.llm)
        await self.record_diagnosis(self.diagnosis)

        # Bounded: `default_route` stops asking for evidence once
        # IC_MAX_EVIDENCE_ROUNDS is spent.
        route = self._policy_route()
        while route == "gather_evidence":
            await self.gather_evidence(self.diagnosis)
            observations = await self.observe()
            await self.emit(EventType.STEP_START, AgentName.DIAGNOSTICIAN, re_diagnosis=True)
            self.diagnosis = await self.run_agent(diagnostician.run, s, observations, self.llm)
            await self.record_diagnosis(self.diagnosis)
            route = self._policy_route()

        if route == "escalate":
            await self.halt(
                HaltReason.LOW_CONFIDENCE,
                IncidentStatus.ESCALATED,
                (
                    f"diagnostic confidence stayed below {self.settings.min_action_confidence} after "
                    f"{s.evidence_rounds} evidence round(s); refusing to act on a guess"
                ),
                AgentName.DIAGNOSTICIAN,
            )
        return observations

    async def record_diagnosis(self, diag: DiagnosisResult) -> None:
        self.state.diagnostics.append(diag.model_dump(mode="json"))
        for h in diag.hypotheses[:3]:
            self.state.hypotheses.append(h.model_dump(mode="json"))
            await self.emit(
                EventType.HYPOTHESIS,
                AgentName.DIAGNOSTICIAN,
                text=h.cause,
                cause=h.cause,
                confidence=h.confidence,
                evidence_needed=h.evidence_needed,
                tool_to_check=h.tool_to_check.value if h.tool_to_check else None,
            )
        await self.emit(
            EventType.DIAGNOSTIC,
            AgentName.DIAGNOSTICIAN,
            finding=diag.leading_cause,
            text=diag.leading_cause,
            source="; ".join(diag.evidence) if diag.evidence else "telemetry",
            leading_cause=diag.leading_cause,
            confidence=diag.confidence,
            evidence=diag.evidence,
            # The floor travels with the number: "0.55 against a 0.6 floor".
            confidence_floor=self.settings.min_action_confidence,
            needs_more_data=diag.needs_more_data,
            requested_tools=[t.value for t in diag.requested_tools],
            used_recalled_episode=diag.used_recalled_episode,
        )

    async def gather_evidence(self, diag: DiagnosisResult) -> None:
        """Run the read-only tools the Diagnostician asked for.

        Bounded by `IC_MAX_EVIDENCE_ROUNDS`. Results are sanitized like any other
        telemetry and land in `state.evidence`, where the next diagnosis sees them.
        """
        tools: list[TelemetryTool] = list(diag.requested_tools)
        if not tools:
            return
        s = self.state
        s.evidence_rounds += 1
        # Say WHY: a model will report needs_more_data at 0.9 confidence ("let me
        # confirm"), and claiming it was below the floor would be plainly false.
        floor = self.settings.min_action_confidence
        below_floor = diag.confidence < floor
        reason = (
            f"confidence {diag.confidence} is below the {floor} action threshold; gathering more evidence"
            if below_floor
            else f"the diagnostician asked to confirm its hypothesis (confidence {diag.confidence}) before acting"
        )
        await self.emit(
            EventType.EVIDENCE_REQUEST,
            AgentName.DIAGNOSTICIAN,
            tools=[t.value for t in tools],
            round=s.evidence_rounds,
            max_rounds=self.settings.max_evidence_rounds,
            below_confidence_floor=below_floor,
            confidence=diag.confidence,
            reason=reason,
        )
        s.status = IncidentStatus.GATHERING_EVIDENCE
        raw = await asyncio.to_thread(run_tools, self.stack, tools)
        cleaned = self.guard.sanitize(raw).observations
        for name, value in cleaned.items():
            s.evidence[f"iter{s.iterations}:{name}"] = value
        await self.emit(
            EventType.EVIDENCE_GATHERED,
            AgentName.DIAGNOSTICIAN,
            tools=[t.value for t in tools],
            round=s.evidence_rounds,
            results=cleaned,
        )
        await self.drain_findings()

    async def propose(self, observations: dict) -> tuple[RemediationAction, RemediationProposal]:
        s = self.state
        await self.emit(EventType.STEP_START, AgentName.REMEDIATION)
        prop = await self.run_agent(remediation.run, s, self.diagnosis, observations, self.llm)
        # Risk is assigned by the guardrail layer, NOT by the model. An agent that
        # could grade its own rollback "low risk" could route around the gate.
        action = RemediationAction(
            action_type=prop.action_type.value,
            target=prop.target or s.service,
            risk=self.guard.assess_risk(prop.action_type.value),
            rationale=prop.rationale,
            expected_signal=prop.expected_signal,
            iteration=s.iterations,
        )
        s.actions.append(action)
        await self.emit(
            EventType.PROPOSED_ACTION,
            AgentName.REMEDIATION,
            action_id=action.action_id,
            action_type=action.action_type,
            target=action.target,
            risk=action.risk.value,
            rationale=action.rationale,
            expected_signal=action.expected_signal,
            reversible=prop.reversible,
            fallback=prop.fallback_if_ineffective.value if prop.fallback_if_ineffective else None,
            self_assessed_risk=prop.risk.value,
        )
        return action, prop

    async def passes_guardrails(self, action: RemediationAction) -> bool:
        """RBAC allowlist, repeat-action block and the action's preconditions."""
        facts = await asyncio.to_thread(precondition_facts, self.stack)
        blocked = self.guard.check_action(action.action_type, self.state.failed_action_types(), facts)
        await self.drain_findings()
        if blocked is None:
            return True
        action.status = ActionStatus.BLOCKED
        action.approval_note = f"blocked: {blocked.detail}"
        if blocked.kind == "rbac_block":
            # The role may never run this: no amount of re-diagnosis will change that.
            await self.halt(
                HaltReason.GUARDRAIL_BLOCKED, IncidentStatus.ESCALATED, blocked.detail, AgentName.REMEDIATION
            )
        else:
            # A repeat of a failed action, or a precondition that does not hold:
            # recoverable. Re-diagnose; the iteration ceiling bounds the retries.
            self.rediagnose = True
        return False

    async def passes_critic(self, action: RemediationAction, prop: RemediationProposal, observations: dict) -> bool:
        """Adversarial review before a human is bothered. A veto means re-diagnose."""
        if not self.settings.critic_enabled:
            return True
        s = self.state
        s.status = IncidentStatus.CRITIQUING
        await self.emit(EventType.STEP_START, AgentName.CRITIC, action_id=action.action_id)
        verdict = await self.run_agent(critic.run, s, self.diagnosis, prop, action.risk, observations, self.llm)
        action.critique = verdict.model_dump(mode="json")
        await self.emit(
            EventType.CRITIQUE,
            AgentName.CRITIC,
            action_id=action.action_id,
            endorse=verdict.endorse,
            concerns=verdict.concerns,
            strongest_objection=verdict.strongest_objection,
            safer_alternative=verdict.safer_alternative.value if verdict.safer_alternative else None,
            confidence=verdict.confidence,
        )
        if verdict.endorse:
            return True
        self.critic_vetoes += 1
        action.status = ActionStatus.REJECTED
        action.approval_note = f"critic veto: {verdict.strongest_objection}"
        if self.critic_vetoes > self.settings.max_critic_vetoes:
            await self.halt(
                HaltReason.CRITIC_VETO,
                IncidentStatus.ESCALATED,
                (
                    f"the Critic vetoed {self.critic_vetoes} proposals; the planner and reviewer "
                    "cannot agree, so a human should decide"
                ),
                AgentName.CRITIC,
            )
        else:
            self.rediagnose = True
        return False

    async def passes_human_gate(self, action: RemediationAction) -> bool:
        """High risk always waits for a human; low/medium only when auto-approve is off."""
        if await self.stop_if_cancelled():
            return False
        if not self.guard.requires_approval(action):
            action.status = ActionStatus.APPROVED  # auto-approved (low/medium risk, opted in)
            action.decided_by, action.decided_at = "policy:auto_approve", time.time()
            await self.emit(
                EventType.STATUS,
                AgentName.ORCHESTRATOR,
                status="auto_approved",
                action_id=action.action_id,
                decided_by=action.decided_by,
            )
            return True

        s = self.state
        s.status = IncidentStatus.AWAITING_APPROVAL
        await self.emit_awaiting_approval(action)
        decision = await self.manager.await_approval(action.action_id, self.settings.approval_timeout_s)
        self.record_decision(action, decision)
        if decision["decision"] == "approve":
            action.status = ActionStatus.APPROVED
            return True

        action.status = ActionStatus.REJECTED
        if decision["decision"] == "timeout":
            await self.halt(
                HaltReason.APPROVAL_TIMEOUT,
                IncidentStatus.ESCALATED,
                "no operator responded to the approval request in time; no action was taken",
            )
        elif s.cancel_requested:
            await self.halt(
                HaltReason.OPERATOR_CANCELLED, IncidentStatus.CANCELLED, "incident cancelled by the operator"
            )
        else:
            await self.halt(
                HaltReason.HUMAN_REJECTED, IncidentStatus.ESCALATED, "operator rejected the proposed remediation"
            )
        return False

    def record_decision(self, action: RemediationAction, decision: dict) -> None:
        """Stamp the human gate's outcome on the action: what, who, and when."""
        action.approval_note = decision.get("note", "")
        action.decided_by = decision.get("actor") or "anonymous"
        action.decided_at = time.time()

    async def emit_awaiting_approval(self, action: RemediationAction) -> None:
        await self.emit(
            EventType.AWAITING_APPROVAL,
            AgentName.REMEDIATION,
            action_id=action.action_id,
            action_type=action.action_type,
            target=action.target,
            risk=action.risk.value,
            rationale=action.rationale,
            expected_signal=action.expected_signal,
            critique=action.critique,
            timeout_s=self.settings.approval_timeout_s,
            requires_approval=True,
        )
        # The run may now wait minutes for a person, after most of its model spend.
        # Without this, the console's cost meter read $0.0000 at the gate in a live run.
        await self.emit_usage()

    async def act(self, action: RemediationAction) -> None:
        """The only mutating call in the system, and only on an approved action."""
        self.state.status = IncidentStatus.REMEDIATING
        await self.emit(EventType.STEP_START, AgentName.REMEDIATION, action_id=action.action_id)
        result = await asyncio.to_thread(execute_remediation, self.stack, action)
        if result.get("effective"):
            label = "effective"
        elif result.get("progress"):
            label = "partial progress"
        elif result.get("harmful"):
            label = "harmful"
        else:
            label = "no effect"
        await self.emit(
            EventType.ACTION_EXECUTED,
            AgentName.REMEDIATION,
            action_id=action.action_id,
            action_type=action.action_type,
            target=action.target,
            risk=action.risk.value,
            result=f"{label}: {result.get('reason')}",
            effective=result.get("effective"),
            progress=result.get("progress"),
            harmful=result.get("harmful", False),
            reason=result.get("reason"),
        )

    async def verify(self, action: RemediationAction) -> None:
        """Fresh telemetry, judged against the action's own prediction."""
        s = self.state
        s.status = IncidentStatus.VERIFYING
        await self.emit(EventType.STEP_START, AgentName.VERIFIER)
        post_obs = await self.observe()
        verdict = await self.run_agent(verifier.run, s, post_obs, self.llm, action)
        s.verification_history.append(verdict.model_dump(mode="json"))
        detail = "recovered" if verdict.recovered else f"not recovered, next step: {verdict.next_step}"
        if verdict.observations:
            detail += f" ({'; '.join(verdict.observations)})"
        await self.emit(
            EventType.VERIFICATION,
            AgentName.VERIFIER,
            recovered=verdict.recovered,
            confidence=verdict.confidence,
            next_step=verdict.next_step,
            observations=verdict.observations,
            expected_signal_met=verdict.expected_signal_met,
            signal_assessment=verdict.signal_assessment,
            partial_progress=verdict.partial_progress,
            detail=detail,
        )
        await self.emit_usage()

        veto = None
        if verdict.recovered:
            veto = self.guard.recovery_not_confirmed(
                post_obs, self.settings.recovery_max_error_rate, self.settings.recovery_max_p99_ms
            )
        if veto is not None:
            # The model says "recovered", the numbers say otherwise: the numbers win.
            s.guardrail_flags.append({"kind": "recovery_not_confirmed", "detail": veto})
            await self.emit(
                EventType.GUARDRAIL,
                AgentName.ORCHESTRATOR,
                kind="recovery_not_confirmed",
                message=f"The Verifier reported recovery, but code refused it: {veto}. Looping again.",
            )
        elif verdict.recovered:
            # A recovery the prediction did not explain is a warning, not a clean
            # win: we do not know why it went away, so it may come back.
            if not verdict.expected_signal_met:
                await self.emit(
                    EventType.GUARDRAIL,
                    AgentName.VERIFIER,
                    kind="unexplained_recovery",
                    message=(
                        "The service recovered but the predicted signal did not change as predicted. "
                        "The diagnosis may be wrong even though the outcome was good."
                    ),
                )
            await self.halt(
                HaltReason.RECOVERED,
                IncidentStatus.RESOLVED,
                f"Service recovered after {s.iterations} loop iteration(s).",
            )
        elif verdict.next_step == "escalate":
            await self.halt(
                HaltReason.VERIFIER_ESCALATED,
                IncidentStatus.ESCALATED,
                "verifier could not confirm recovery and saw no productive next step",
            )
        # otherwise: loop again

    # --- the alternative engine: LangGraph StateGraph ------------------------

    async def run_graph(self) -> None:
        """Drive the LangGraph engine (`IC_ENGINE=graph`), servicing its approval interrupts.

        Same agents, same guardrail manager, executed by a graph instead of a
        `while`. The graph stops and CHECKPOINTS before the only mutating node, so
        the pending approval lives in the checkpointer, not only in this process.
        """
        from .graph import build_graph

        s = self.state
        collected: list[tuple[str, dict]] = []
        app = build_graph(self.llm, self.stack, self.guard, lambda kind, **p: collected.append((kind, p)))
        config = {"configurable": {"thread_id": s.incident_id}, "recursion_limit": 100}

        # Graph nodes report plain kind strings; map them onto the SAME event
        # vocabulary the explicit loop uses, so the console renders both engines.
        node_events: dict[str, tuple[EventType, AgentName]] = {
            "observe": (EventType.LOOP, AgentName.ORCHESTRATOR),
            "triage": (EventType.STEP_END, AgentName.TRIAGE),
            "diagnose": (EventType.DIAGNOSTIC, AgentName.DIAGNOSTICIAN),
            "gather_evidence": (EventType.EVIDENCE_GATHERED, AgentName.DIAGNOSTICIAN),
            "propose": (EventType.PROPOSED_ACTION, AgentName.REMEDIATION),
            "critique": (EventType.CRITIQUE, AgentName.CRITIC),
            "act": (EventType.ACTION_EXECUTED, AgentName.REMEDIATION),
            "verify": (EventType.VERIFICATION, AgentName.VERIFIER),
            "blocked": (EventType.GUARDRAIL, AgentName.ORCHESTRATOR),
        }

        async def flush_events() -> None:
            while collected:
                kind, payload = collected.pop(0)
                if kind == "awaiting_approval":
                    continue  # emitted below, with the full payload
                etype, agent = node_events.get(kind, (EventType.STATUS, AgentName.SUPERVISOR))
                await self.emit(etype, agent, node=kind, **payload)
            await self.drain_findings()

        async def push_cancel(current: IncidentState) -> None:
            # The graph runs on the checkpointer's COPY of the state, so a cancel
            # set on the manager's object is invisible to it until written back.
            if s.cancel_requested and not current.cancel_requested:
                current.cancel_requested = True
                await asyncio.to_thread(app.update_state, config, {"incident": current})

        await asyncio.to_thread(app.invoke, {"incident": s}, config)
        await flush_events()

        for _ in range(s.max_iterations * 3 + 5):
            snapshot = await asyncio.to_thread(app.get_state, config)
            if not snapshot.next:
                break
            current = snapshot.values["incident"]
            # Mirror the checkpoint onto the manager's state, so the API (which
            # serves the manager's object) shows the live status at the gate.
            _merge_graph_state(s, current)
            await push_cancel(current)
            if current.status == IncidentStatus.AWAITING_APPROVAL and current.actions and not s.cancel_requested:
                action = current.actions[-1]
                await self.emit_awaiting_approval(action)
                decision = await self.manager.await_approval(action.action_id, self.settings.approval_timeout_s)
                approved = decision["decision"] == "approve" and not s.cancel_requested
                self.record_decision(action, decision)
                if approved:
                    action.status = ActionStatus.APPROVED
                    current.status = IncidentStatus.REMEDIATING  # leave the gate at once
                else:
                    action.status = ActionStatus.REJECTED
                    if s.cancel_requested:
                        current.status, current.halt_reason = IncidentStatus.CANCELLED, HaltReason.OPERATOR_CANCELLED
                    elif decision["decision"] == "timeout":
                        current.status, current.halt_reason = IncidentStatus.ESCALATED, HaltReason.APPROVAL_TIMEOUT
                    else:
                        current.status, current.halt_reason = IncidentStatus.ESCALATED, HaltReason.HUMAN_REJECTED
                # `current` is a copy: the decision must be written back, not mutated.
                await asyncio.to_thread(app.update_state, config, {"incident": current, "approved": approved})
                _merge_graph_state(s, current)
                if not approved:
                    await self.emit_graph_halt()
                    return
            await asyncio.to_thread(app.invoke, None, config)
            await flush_events()
            await self.emit_usage()

        _merge_graph_state(s, (await asyncio.to_thread(app.get_state, config)).values["incident"])
        if not self.finished:
            s.status = IncidentStatus.ESCALATED
            s.halt_reason = s.halt_reason or HaltReason.ITERATION_CEILING
        await self.emit_graph_halt()

    async def emit_graph_halt(self) -> None:
        """The graph sets the terminal status inside a node; announce it like the loop does."""
        s = self.state
        reason = s.halt_reason or HaltReason.CRASH
        message = {
            HaltReason.RECOVERED: f"Service recovered after {s.iterations} loop iteration(s).",
            HaltReason.OPERATOR_CANCELLED: "incident cancelled by the operator",
            HaltReason.HUMAN_REJECTED: "operator rejected the proposed remediation",
            HaltReason.APPROVAL_TIMEOUT: "no operator responded to the approval request in time; no action was taken",
        }.get(reason, f"halted on {reason.value.replace('_', ' ')}")
        await self.halt(reason, s.status, message)

    # --- memory and postmortem ----------------------------------------------

    async def recall_memory(self) -> None:
        """Retrieve similar past incidents before the first diagnosis."""
        if not self.settings.memory_enabled:
            return
        s = self.state
        try:
            # The query is built from symptoms only: the service and its raw log
            # lines. Never the scenario id, which is the simulator's answer key.
            query = " ".join([s.service, *self.stack.logs(limit=6)])
            hits = get_memory().recall(query, top_k=self.settings.memory_top_k, exclude_incident=s.incident_id)
        except Exception as exc:  # noqa: BLE001 - memory is an enhancement, never a hard dependency
            logger.warning("runbook memory recall failed: %s", exc)
            return
        if not hits:
            return
        s.recalled_episodes = [{**h.as_dict(), "prompt_line": h.episode.as_prompt_line()} for h in hits]
        await self.emit(
            EventType.MEMORY_RECALL,
            AgentName.ORCHESTRATOR,
            count=len(hits),
            episodes=s.recalled_episodes,
            message=f"recalled {len(hits)} similar past incident(s) from runbook memory",
        )

    async def record_memory(self) -> None:
        """Write this incident back into episodic memory, outcome and all."""
        s = self.state
        if not self.settings.memory_enabled or s.status not in _TERMINAL:
            return
        try:
            get_memory().record(episode_from_state(s, s.postmortem))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not record runbook episode: %s", exc)

    async def write_postmortem(self) -> None:
        """Always write a postmortem - especially for the runs that failed.

        If no model can write it (not configured, down, or it errors), a factual
        one is assembled from the record instead, so invariant 6 holds on every
        outcome. A failed writeup never changes how the incident ended.
        """
        await self.emit(EventType.STEP_START, AgentName.POSTMORTEM)
        pm = None
        if self.llm is not None and not postmortem.model_should_write(self.state):
            pm = postmortem.fallback(self.state, postmortem.UNAVAILABLE_WHY)
        elif self.llm is not None:
            try:
                pm = await asyncio.to_thread(postmortem.run, self.state, self.llm)
            except Exception as exc:  # noqa: BLE001
                logger.warning("postmortem generation failed: %s", exc)
                await self.emit(
                    EventType.ERROR,
                    AgentName.POSTMORTEM,
                    kind="postmortem_failed",
                    message=f"Postmortem generation failed: {exc}",
                )
                pm = postmortem.fallback(self.state, f"the model call failed ({exc}).")
        else:
            pm = postmortem.fallback(self.state, "no model was available.")
        self.state.postmortem = pm
        await self.emit(EventType.POSTMORTEM, AgentName.POSTMORTEM, **pm)


#: Fields the graph engine owns and may write back onto the manager's state.
#: Explicit rather than a wholesale copy: the checkpointer round-trips a COPY of
#: the state, and blindly merging it back would erase the orchestrator-owned
#: event log with whatever the graph's copy last saw.
_GRAPH_OWNED_FIELDS = (
    "status",
    "severity",
    "halt_reason",
    "iterations",
    "evidence_rounds",
    "triage_summary",
    "hypotheses",
    "diagnostics",
    "actions",
    "verification_history",
    "evidence",
    "guardrail_flags",
)


def _merge_graph_state(state: IncidentState, graph_state: IncidentState) -> None:
    """Copy the graph's results onto the manager's state, preserving the event log."""
    if graph_state is state:  # pragma: no cover - only when no checkpointer is used
        return
    for name in _GRAPH_OWNED_FIELDS:
        setattr(state, name, getattr(graph_state, name))


_manager: IncidentManager | None = None


def get_manager() -> IncidentManager:
    global _manager
    if _manager is None:
        _manager = IncidentManager()
    return _manager


def reset_manager_cache() -> None:
    """Test hook: drop the memoized manager."""
    global _manager
    _manager = None

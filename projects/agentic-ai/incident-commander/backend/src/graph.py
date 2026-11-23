"""The same agent loop, expressed as a LangGraph `StateGraph`.

`loop.py` is the teaching reference: an explicit `while` loop where every halt
condition and the human `await` are visible on one screen. This module is the
production shape of the identical system, and it exists so a learner can read both
and see precisely what a graph framework does and does not give you.

## Reading the two side by side

| Concern              | `loop.py` (explicit)             | `graph.py` (StateGraph)               |
|----------------------|----------------------------------|---------------------------------------|
| Control flow         | `while` + `if`                   | nodes + conditional edges             |
| Who routes           | `_route()` calls the Supervisor  | `route_after_diagnose` edge function  |
| Halt conditions      | checked at the top of the loop   | a `guard` node that can jump to finish |
| Durability           | none - process restart loses it  | a checkpointer persists every step    |
| Resumability         | none                             | resume by thread_id after a crash     |
| Human-in-the-loop    | `await` on an asyncio Future     | `interrupt_before` + state update     |
| Debuggability        | read the function top to bottom  | inspect the checkpoint at each node   |

The honest summary: the graph buys you **durability, resumability, and
inspectability**. It does not buy you safety - the guardrails, RBAC and ceilings
are the same code called from the same places. A framework that made those
implicit would make the system harder to audit, not easier.

## What is genuinely better here

`interrupt_before=["act"]` is the real win. In `loop.py`, an incident parked at the
approval gate lives in one process's memory as a pending `asyncio.Future`; restart
the process and the incident is gone. Here the graph checkpoints *before* the
acting node, so with a persistent checkpointer the approval can arrive hours
later, from another process, after a deploy - and the run resumes from the
checkpoint. That is what "durable human-in-the-loop" means. As shipped the
checkpointer is the in-process `MemorySaver`: the seam is here, the durability
needs `AsyncPostgresSaver` (EXERCISES.md, exercise 15).

Select the engine with `IC_ENGINE=graph`. Both engines drive the *same* agent
functions and the same guardrail layer; `tests/test_graph_parity.py` asserts they
reach the same outcomes on every scenario, which is what makes this a genuine
alternative implementation rather than a second, subtly different system.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Literal, TypedDict

from .agents import critic, diagnostician, postmortem, remediation, supervisor, triage, verifier
from .config import get_settings
from .guardrails.manager import GuardrailManager
from .llm import token_tracker
from .models.state import (
    ActionStatus,
    HaltReason,
    IncidentState,
    IncidentStatus,
    RemediationAction,
)
from .sandbox.stack import SandboxStack
from .tools.observability import observe_all, precondition_facts, run_tools
from .tools.remediation import execute_remediation

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    """The channel dict LangGraph threads between nodes.

    `IncidentState` is carried whole rather than being shredded into channels. That
    is deliberate: one typed state object is what keeps the two engines equivalent,
    and what makes a checkpoint a complete, human-readable snapshot of the incident.
    """

    incident: IncidentState
    observations: dict[str, Any]
    diagnosis: Any
    proposal: Any
    route: str
    approved: bool
    approval_note: str


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------
# Nodes need the LLM, the sandbox and the guardrail manager, none of which belong
# in serialized graph state (they are not JSON, and a checkpoint containing an API
# client would be both useless and a security problem). They are closed over here
# instead, which is the standard way to inject runtime dependencies into a graph.


def build_nodes(llm, stack: SandboxStack, guard: GuardrailManager, emit: Callable | None = None) -> dict[str, Callable]:
    settings = get_settings()

    def _emit(kind: str, **payload) -> None:
        if emit is not None:
            emit(kind, **payload)

    def _sync_usage(incident: IncidentState) -> None:
        """Refresh the incident's cost figures from the process-wide ledger.

        The checkpointer round-trips a COPY of the state, so the loop engine's
        habit of updating `state.total_cost_usd` on the manager's object does not
        reach the graph's copy. Without this call, `exhausted_reason()` in the
        guard node reads a cost that is permanently 0.0 and the cost ceiling never
        trips - the graph engine would happily spend 10x the budget and then
        report `iteration_ceiling`. Called wherever spend may just have happened:
        before the halt check, and after every model-calling node.
        """
        tokens, cost = token_tracker.incident_cost(incident.incident_id)
        incident.total_tokens, incident.total_cost_usd = tokens, cost

    def observe(state: GraphState) -> GraphState:
        # Runs AFTER `guard`, exactly like the loop: check the ceilings, then count
        # the pass. Counting first halted one pass early on IC_MAX_ITERATIONS.
        incident = state["incident"]
        incident.iterations += 1
        raw = observe_all(stack)
        result = guard.sanitize(raw)
        for finding in guard.findings_as_dicts()[len(incident.guardrail_flags) :]:
            incident.guardrail_flags.append(finding)
        _emit("observe", iteration=incident.iterations)
        return {"incident": incident, "observations": result.observations}

    def guard_node(state: GraphState) -> GraphState:
        """Halt conditions. Identical policy to the top of `loop.py`'s while."""
        incident = state["incident"]
        _sync_usage(incident)
        tripped = incident.exhausted_reason()
        if tripped is not None:
            incident.halt_reason = tripped
            incident.status = (
                IncidentStatus.CANCELLED if tripped is HaltReason.OPERATOR_CANCELLED else IncidentStatus.ESCALATED
            )
        return {"incident": incident}

    def triage_node(state: GraphState) -> GraphState:
        incident = state["incident"]
        incident.status = IncidentStatus.TRIAGING
        result = triage.run(incident, state["observations"], llm)
        _sync_usage(incident)
        incident.triage_summary = result.summary
        incident.severity = result.severity
        _emit("triage", severity=result.severity.value, summary=result.summary)
        return {"incident": incident}

    def diagnose_node(state: GraphState) -> GraphState:
        incident = state["incident"]
        incident.status = IncidentStatus.DIAGNOSING
        diagnosis = diagnostician.run(incident, state["observations"], llm)
        _sync_usage(incident)
        incident.diagnostics.append(diagnosis.model_dump(mode="json"))
        _emit("diagnose", leading_cause=diagnosis.leading_cause, confidence=diagnosis.confidence)
        return {"incident": incident, "diagnosis": diagnosis}

    def gather_evidence_node(state: GraphState) -> GraphState:
        incident = state["incident"]
        diagnosis = state["diagnosis"]
        incident.status = IncidentStatus.GATHERING_EVIDENCE
        incident.evidence_rounds += 1
        raw = run_tools(stack, list(diagnosis.requested_tools))
        cleaned = guard.sanitize(raw).observations
        for name, value in cleaned.items():
            incident.evidence[f"iter{incident.iterations}:{name}"] = value
        _emit("gather_evidence", tools=[t.value for t in diagnosis.requested_tools])
        return {"incident": incident}

    def propose_node(state: GraphState) -> GraphState:
        incident = state["incident"]
        proposal = remediation.run(incident, state["diagnosis"], state["observations"], llm)
        _sync_usage(incident)
        action_type = proposal.action_type.value
        action = RemediationAction(
            action_type=action_type,
            target=proposal.target or incident.service,
            # Risk comes from the guardrail layer, never from the model - same rule
            # as the explicit loop, enforced in the same place.
            risk=guard.assess_risk(action_type),
            rationale=proposal.rationale,
            expected_signal=proposal.expected_signal,
            iteration=incident.iterations,
        )
        incident.actions.append(action)
        _emit("propose", action_type=action_type, risk=action.risk.value)
        return {"incident": incident, "proposal": proposal}

    def critique_node(state: GraphState) -> GraphState:
        incident = state["incident"]
        action = incident.actions[-1]
        if not settings.critic_enabled:
            return {"incident": incident}
        incident.status = IncidentStatus.CRITIQUING
        verdict = critic.run(incident, state["diagnosis"], state["proposal"], action.risk, state["observations"], llm)
        _sync_usage(incident)
        action.critique = verdict.model_dump(mode="json")
        if not verdict.endorse:
            action.status = ActionStatus.REJECTED
            action.approval_note = f"critic veto: {verdict.strongest_objection}"
            # Same bound as the explicit loop's `critic_vetoes` counter. The graph
            # has no loop-local variable to count in, but the vetoed actions are
            # all in the state, so the count is derived from the audit trail
            # itself. Without this, IC_MAX_CRITIC_VETOES is silently inert in this
            # engine: a stubborn Critic and a stubborn Planner argue until the
            # iteration ceiling and the halt reason misreports what happened.
            vetoes = sum(1 for a in incident.actions if a.critique and not a.critique.get("endorse", True))
            if vetoes > settings.max_critic_vetoes:
                incident.halt_reason = HaltReason.CRITIC_VETO
                incident.status = IncidentStatus.ESCALATED
        _emit("critique", endorse=verdict.endorse, objection=verdict.strongest_objection)
        return {"incident": incident}

    def gate_node(state: GraphState) -> GraphState:
        """The guardrail + approval decision point.

        The graph is configured with `interrupt_before=["act"]`, so when this node
        marks an action as needing a human the run STOPS here and is checkpointed.
        The approval arrives later as a state update, and the run resumes from the
        checkpoint - possibly in a different process, after a restart.
        """
        incident = state["incident"]
        action = incident.actions[-1]
        blocked = guard.check_action(action.action_type, incident.failed_action_types(), precondition_facts(stack))
        for finding in guard.findings_as_dicts()[len(incident.guardrail_flags) :]:
            incident.guardrail_flags.append(finding)
        if blocked is not None:
            action.status = ActionStatus.BLOCKED
            action.approval_note = f"blocked: {blocked.detail}"
            if blocked.kind == "rbac_block":
                # The role may never run this action: escalate.
                incident.halt_reason = HaltReason.GUARDRAIL_BLOCKED
                incident.status = IncidentStatus.ESCALATED
            # Otherwise (a repeated dud, or a precondition that does not hold) it is
            # recoverable, so
            # `route_after_gate` sends it round again to re-diagnose (same as the loop).
            _emit("blocked", reason=blocked.detail)
            return {"incident": incident, "approved": False}

        if guard.requires_approval(action):
            incident.status = IncidentStatus.AWAITING_APPROVAL
            _emit("awaiting_approval", action_id=action.action_id, risk=action.risk.value)
            # Explicitly False, never carried over. The channel keeps its last value
            # across supersteps, so echoing `state.get("approved")` here would let a
            # LOW-risk auto-approval from iteration 1 satisfy iteration 2's HIGH-risk
            # gate on any resume that skips servicing the interrupt - an approval no
            # human ever gave. The resuming caller supplies the real decision via
            # `update_state`.
            return {"incident": incident, "approved": False}

        action.status = ActionStatus.APPROVED
        action.decided_by, action.decided_at = "policy:auto_approve", time.time()
        return {"incident": incident, "approved": True}

    def act_node(state: GraphState) -> GraphState:
        incident = state["incident"]
        action = incident.actions[-1]
        if incident.cancel_requested:
            # Last check before the world changes, as in the loop.
            action.status = ActionStatus.REJECTED
            incident.status, incident.halt_reason = IncidentStatus.CANCELLED, HaltReason.OPERATOR_CANCELLED
            return {"incident": incident, "approved": False}
        # The approval decision arrives via `update_state` while the graph is
        # interrupted. Both the flag and the action status are checked: the flag is
        # the caller's intent, the status is the durable record of it, and acting
        # when they disagree would be exactly the bug this gate exists to prevent.
        if not state.get("approved") or action.status == ActionStatus.REJECTED:
            action.status = ActionStatus.REJECTED
            incident.halt_reason = incident.halt_reason or HaltReason.HUMAN_REJECTED
            incident.status = IncidentStatus.ESCALATED
            return {"incident": incident, "approved": False}
        action.status = ActionStatus.APPROVED
        incident.status = IncidentStatus.REMEDIATING
        result = execute_remediation(stack, action)
        _emit("act", action_type=action.action_type, effective=result.get("effective"))
        # The approval is consumed by the act it authorized. Leaving it True in the
        # channel would pre-approve whatever the NEXT iteration proposes.
        return {"incident": incident, "approved": False}

    def verify_node(state: GraphState) -> GraphState:
        incident = state["incident"]
        incident.status = IncidentStatus.VERIFYING
        post = guard.sanitize(observe_all(stack)).observations
        result = verifier.run(incident, post, llm, incident.actions[-1] if incident.actions else None)
        _sync_usage(incident)
        incident.verification_history.append(result.model_dump(mode="json"))
        veto = None
        if result.recovered:
            veto = guard.recovery_not_confirmed(post, settings.recovery_max_error_rate, settings.recovery_max_p99_ms)
        if veto is not None:
            # The model says "recovered", the numbers say otherwise: loop again.
            incident.guardrail_flags.append({"kind": "recovery_not_confirmed", "detail": veto})
            _emit("blocked", reason=f"recovery not confirmed: {veto}")
        elif result.recovered:
            incident.status = IncidentStatus.RESOLVED
            incident.halt_reason = HaltReason.RECOVERED
        elif result.next_step == "escalate":
            incident.status = IncidentStatus.ESCALATED
            incident.halt_reason = HaltReason.VERIFIER_ESCALATED
        _emit("verify", recovered=result.recovered, next_step=result.next_step)
        return {"incident": incident}

    def finish_node(state: GraphState) -> GraphState:
        """The end of the graph. Deliberately does nothing.

        The postmortem is written by whoever drives the graph (the orchestrator, or
        `run_incident` below), exactly once. Writing it here as well produced two
        postmortem model calls per graph run.
        """
        return {"incident": state["incident"]}

    return {
        "observe": observe,
        "guard": guard_node,
        "triage": triage_node,
        "diagnose": diagnose_node,
        "gather_evidence": gather_evidence_node,
        "propose": propose_node,
        "critique": critique_node,
        "gate": gate_node,
        "act": act_node,
        "verify": verify_node,
        "finish": finish_node,
    }


# ---------------------------------------------------------------------------
# Conditional edges - the routing logic, as pure functions
# ---------------------------------------------------------------------------
# Each returns the name of the next node. These are exactly the branch conditions
# from `loop.py`, lifted out of the `while` body. Being pure functions of state,
# they are directly unit-testable without running a graph at all - which is the
# main ergonomic win of the graph formulation.


_DONE = {IncidentStatus.ESCALATED, IncidentStatus.CANCELLED, IncidentStatus.RESOLVED}


def route_after_guard(state: GraphState) -> Literal["observe", "finish"]:
    return "finish" if state["incident"].status in _DONE else "observe"


def route_after_observe(state: GraphState) -> Literal["triage", "diagnose"]:
    return "diagnose" if state["incident"].triage_summary else "triage"


def route_after_diagnose(state: GraphState) -> Literal["gather_evidence", "propose", "finish"]:
    settings = get_settings()
    incident = state["incident"]
    diagnosis = state.get("diagnosis")
    if diagnosis is None:  # pragma: no cover - defensive
        return "finish"
    needs_more = diagnosis.needs_more_data or diagnosis.confidence < settings.min_action_confidence
    if needs_more:
        if diagnosis.requested_tools and incident.evidence_rounds < settings.max_evidence_rounds:
            return "gather_evidence"
        if diagnosis.confidence < settings.min_action_confidence:
            # Refuse to act on a diagnosis the diagnostician does not believe.
            incident.status = IncidentStatus.ESCALATED
            incident.halt_reason = HaltReason.LOW_CONFIDENCE
            return "finish"
    return "propose"


def route_after_critique(state: GraphState) -> Literal["gate", "guard", "finish"]:
    incident = state["incident"]
    if incident.status is IncidentStatus.ESCALATED:
        return "finish"  # veto ceiling reached: the planner and reviewer cannot agree
    action = incident.actions[-1] if incident.actions else None
    if action is not None and action.status == ActionStatus.REJECTED:
        return "guard"  # vetoed: go around (ceilings first) and re-diagnose
    return "gate"


def route_after_gate(state: GraphState) -> Literal["act", "guard", "finish"]:
    incident = state["incident"]
    if incident.status in _DONE:
        return "finish"
    if incident.actions and incident.actions[-1].status == ActionStatus.BLOCKED:
        return "guard"  # repeat-action block: go around and re-diagnose
    return "act"


def route_after_verify(state: GraphState) -> Literal["guard", "finish"]:
    return "finish" if state["incident"].status in _DONE else "guard"  # else: loop again


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def _checkpoint_types() -> list[tuple[str, str]]:
    """Every model and enum the incident state can contain, as (module, name)."""
    import enum

    from pydantic import BaseModel

    from .models import schemas, state

    allowed = []
    for module in (state, schemas):
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, (BaseModel, enum.Enum)) and obj.__module__ == module.__name__:
                allowed.append((obj.__module__, obj.__name__))
    return allowed


def build_graph(llm, stack: SandboxStack, guard: GuardrailManager, emit: Callable | None = None, checkpointer=None):
    """Compile the incident graph.

    `interrupt_before=["act"]` is the durable human-in-the-loop seam: the run halts
    and checkpoints immediately before the only mutating node, and resumes when the
    caller supplies an approval decision.
    """
    from langgraph.graph import END, START, StateGraph

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        # MemorySaver is per-process. Swap in `AsyncPostgresSaver` and the same graph
        # survives a restart with no other change - that substitutability is the
        # whole argument for the graph formulation.
        #
        # The serializer gets an explicit allowlist of OUR state types. Without it
        # LangGraph deserializes any type a checkpoint names (and warns that a future
        # version will refuse): a checkpoint store is input, not trusted code.
        checkpointer = MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_checkpoint_types()))

    nodes = build_nodes(llm, stack, guard, emit)
    graph = StateGraph(GraphState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", route_after_guard)
    graph.add_conditional_edges("observe", route_after_observe)
    graph.add_edge("triage", "diagnose")
    graph.add_conditional_edges("diagnose", route_after_diagnose)
    # After gathering evidence, go straight back to the diagnostician with it.
    graph.add_edge("gather_evidence", "diagnose")
    graph.add_edge("propose", "critique")
    graph.add_conditional_edges("critique", route_after_critique)
    graph.add_conditional_edges("gate", route_after_gate)
    graph.add_edge("act", "verify")
    graph.add_conditional_edges("verify", route_after_verify)
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["act"])


def run_incident(
    incident: IncidentState,
    stack: SandboxStack,
    guard: GuardrailManager,
    llm,
    approve: Callable[[RemediationAction], tuple[bool, str]],
    *,
    emit: Callable | None = None,
    checkpointer=None,
) -> IncidentState:
    """Drive the compiled graph to a terminal state, servicing approval interrupts.

    `approve` is called whenever the graph interrupts before the acting node, and
    returns `(approved, note)`. In the API this is backed by the operator; in the
    evaluation harness it is an approve-all policy; in tests it is whatever the case
    needs.

    Note the resume mechanism: the graph is invoked with `None` to continue from its
    checkpoint after the caller writes the decision in with `update_state`. That
    round trip is the whole point - the decision could equally have come from a
    different process an hour later.
    """
    app = build_graph(llm, stack, guard, emit, checkpointer)
    config = {"configurable": {"thread_id": incident.incident_id}, "recursion_limit": 100}

    state: dict[str, Any] = {"incident": incident}
    app.invoke(state, config)

    # Service interrupts until the graph reaches END.
    #
    # Note that `snapshot.values["incident"]` is a DESERIALIZED COPY, not the object
    # we passed in: the checkpointer round-trips state on every step. That is the
    # price of durability, and it means a decision has to be written back through
    # `update_state` rather than by mutating the original object. Getting this wrong
    # produces a graph that appears to accept an approval and then ignores it.
    current = incident
    for _ in range(incident.max_iterations * 4 + 8):
        snapshot = app.get_state(config)
        if not snapshot.next:
            break
        current = snapshot.values["incident"]
        action = current.actions[-1] if current.actions else None
        if action is not None and current.status == IncidentStatus.AWAITING_APPROVAL:
            approved, note = approve(action)
            action.approval_note = note
            action.decided_by, action.decided_at = "caller:run_incident", time.time()
            if not approved:
                action.status = ActionStatus.REJECTED
                current.status = IncidentStatus.ESCALATED
                current.halt_reason = HaltReason.HUMAN_REJECTED
            else:
                action.status = ActionStatus.APPROVED
            app.update_state(config, {"incident": current, "approved": approved})
            if not approved:
                # Nothing left to run: the only mutating node is the one we just
                # refused. Stop here rather than resuming into a node that would
                # only re-derive the same refusal.
                return _with_postmortem(current, llm)
        app.invoke(None, config)

    final = app.get_state(config).values["incident"]
    if final.halt_reason is None and final.status not in _DONE:
        # Ran out of graph steps without a terminal verdict: treat exactly like the
        # explicit loop's iteration ceiling rather than reporting a spurious success.
        final.status = IncidentStatus.ESCALATED
        final.halt_reason = HaltReason.ITERATION_CEILING
    return _with_postmortem(final, llm)


def _with_postmortem(incident: IncidentState, llm) -> IncidentState:
    """Write the postmortem once, after the graph ends (see `finish_node`)."""
    if not postmortem.model_should_write(incident):
        incident.postmortem = postmortem.fallback(incident, postmortem.UNAVAILABLE_WHY)
        return incident
    try:
        incident.postmortem = postmortem.run(incident, llm)
    except Exception as exc:  # noqa: BLE001 - a failed writeup must not flip the outcome
        logger.warning("postmortem generation failed: %s", exc)
        incident.postmortem = postmortem.fallback(incident, f"the model call failed ({exc}).")
    return incident


__all__ = [
    "GraphState",
    "build_graph",
    "run_incident",
    "build_nodes",
    "route_after_guard",
    "route_after_observe",
    "route_after_diagnose",
    "route_after_critique",
    "route_after_gate",
    "route_after_verify",
    "supervisor",  # re-exported so the graph engine and loop share one routing module
]

"""The incident state machine and the event envelope streamed to the UI.

`IncidentState` is the single source of truth that flows through every node of
the agent loop. Each agent reads it, appends to it, and hands it on. Keeping all
loop state in one typed object is what makes the loop bounded, auditable, and
serializable (which is what lets the LangGraph engine checkpoint and resume it).

Four enums are worth reading closely, because they are the system's contracts:

- `ActionType`    - the closed menu of things the agent may ever do. Because the
  LLM is bound to this enum, "hallucinate a new action" stops being a failure mode.
- `TelemetryTool` - the closed menu of read-only tools the Diagnostician may
  request when it decides it needs more evidence.
- `HaltReason`    - why a run ended. An enum, not free text, so the evaluation
  harness can measure *how* runs end and not just whether they succeeded.
- `EventType`     - the streaming contract with the React console.
"""

from __future__ import annotations

import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Severity(str, enum.Enum):
    SEV1 = "SEV1"  # full outage
    SEV2 = "SEV2"  # major degradation
    SEV3 = "SEV3"  # minor / partial
    SEV4 = "SEV4"  # cosmetic


class IncidentStatus(str, enum.Enum):
    DETECTED = "detected"
    TRIAGING = "triaging"
    DIAGNOSING = "diagnosing"
    GATHERING_EVIDENCE = "gathering_evidence"
    CRITIQUING = "critiquing"
    AWAITING_APPROVAL = "awaiting_approval"
    REMEDIATING = "remediating"
    VERIFYING = "verifying"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ActionType(str, enum.Enum):
    """The closed menu of remediation actions.

    Binding the Remediation agent's structured output to this enum means an
    out-of-menu action is a *decoding* error, not a runtime surprise. The guardrail
    layer then decides which of these the current role may actually run.
    """

    NO_OP = "no_op"
    CLEAR_CACHE = "clear_cache"
    RESTART_SERVICE = "restart_service"
    SCALE_OUT = "scale_out"
    ROLLBACK_DEPLOY = "rollback_deploy"
    FAILOVER = "failover"


class TelemetryTool(str, enum.Enum):
    """Read-only tools the Diagnostician can ask for by name.

    This enum is what turns the Diagnostician from a passive consumer of a fixed
    telemetry dump into an agent that decides which evidence it needs next.
    """

    METRICS = "get_metrics"
    LOGS = "get_logs"
    TRACES = "get_traces"
    DEPLOY_HISTORY = "get_deploy_history"
    POOL_STATS = "get_pool_stats"
    CONFIG_DIFF = "get_config_diff"
    DEPENDENCY_HEALTH = "get_dependency_health"
    SLO_BURN = "get_slo_burn"


class ActionRisk(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionStatus(str, enum.Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"  # refused by the guardrail layer; never reached a human
    EXECUTED = "executed"
    FAILED = "failed"


class AgentName(str, enum.Enum):
    TRIAGE = "Triage"
    DIAGNOSTICIAN = "Diagnostician"
    REMEDIATION = "Remediation"
    CRITIC = "Critic"
    VERIFIER = "Verifier"
    POSTMORTEM = "Postmortem"
    SUPERVISOR = "Supervisor"
    ORCHESTRATOR = "Orchestrator"


class EventType(str, enum.Enum):
    STATUS = "status"
    STEP_START = "step_start"
    STEP_END = "step_end"
    ROUTE = "route"  # supervisor picked the next agent
    MEMORY_RECALL = "memory_recall"  # similar past incidents retrieved from memory
    HYPOTHESIS = "hypothesis"
    DIAGNOSTIC = "diagnostic"
    EVIDENCE_REQUEST = "evidence_request"  # agent asked for specific tools
    EVIDENCE_GATHERED = "evidence_gathered"  # tools were run, results returned
    PROPOSED_ACTION = "proposed_action"
    CRITIQUE = "critique"  # Critic challenged the proposal
    GUARDRAIL = "guardrail"  # a guardrail fired (block, redaction, quarantine)
    AWAITING_APPROVAL = "awaiting_approval"
    ACTION_EXECUTED = "action_executed"
    VERIFICATION = "verification"
    LOOP = "loop"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"
    POSTMORTEM = "postmortem"
    TOKEN_USAGE = "token_usage"
    ERROR = "error"


class HaltReason(str, enum.Enum):
    """Why the loop stopped. Every terminal incident has exactly one."""

    RECOVERED = "recovered"
    ITERATION_CEILING = "iteration_ceiling"
    COST_CEILING = "cost_ceiling"
    WALLCLOCK_CEILING = "wallclock_ceiling"
    HUMAN_REJECTED = "human_rejected"
    APPROVAL_TIMEOUT = "approval_timeout"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    VERIFIER_ESCALATED = "verifier_escalated"
    LOW_CONFIDENCE = "low_confidence"
    CRITIC_VETO = "critic_veto"
    OPERATOR_CANCELLED = "operator_cancelled"
    LLM_UNAVAILABLE = "llm_unavailable"
    CRASH = "crash"


class RemediationAction(BaseModel):
    action_id: str = Field(default_factory=lambda: _new_id("act"))
    action_type: str  # one of ActionType values
    target: str  # service / resource the action touches
    risk: ActionRisk = ActionRisk.MEDIUM
    rationale: str = ""
    status: ActionStatus = ActionStatus.PROPOSED
    approval_note: str = ""
    #: What the proposing agent predicted would change if this worked. The Verifier
    #: is judged against this, which is what makes the proposal falsifiable.
    expected_signal: str = ""
    #: The Critic's verdict on this proposal, if the Critic ran.
    critique: dict[str, Any] | None = None
    iteration: int = 0
    result: dict[str, Any] | None = None
    #: WHO decided at the human gate: a username, "policy:auto_approve" for a
    #: low/medium action auto-approved by configuration, or "nobody:timeout".
    #: An action that can roll back production must say who allowed it.
    decided_by: str | None = None
    decided_at: float | None = None


class IncidentEvent(BaseModel):
    """The envelope streamed over SSE. Mirrors the frontend contract exactly.

    `seq` is a monotonic per-incident sequence number so an SSE client can resume
    from `Last-Event-ID` instead of replaying the whole history on every reconnect.
    """

    type: EventType
    ts: float = Field(default_factory=_now)
    incident_id: str
    seq: int = 0
    iteration: int = 0
    agent: AgentName | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class IncidentState(BaseModel):
    """Full evolving state of one incident as it moves through the loop."""

    incident_id: str = Field(default_factory=lambda: _new_id("inc"))
    scenario_id: str
    service: str
    severity: Severity = Severity.SEV3
    status: IncidentStatus = IncidentStatus.DETECTED
    #: Which engine ran this incident: "loop" (hand-rolled) or "graph" (LangGraph).
    engine: str = "loop"

    started_at: float = Field(default_factory=_now)
    ended_at: float | None = None

    # --- Loop-engineering counters and ceilings (the halt conditions) -------
    iterations: int = 0
    max_iterations: int = 6
    max_cost_usd: float = 0.50
    max_wallclock_s: float = 900.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    #: Extra read-only tool rounds spent this incident (bounded, see config).
    evidence_rounds: int = 0
    halt_reason: HaltReason | None = None

    # --- Working memory accumulated across loop iterations -----------------
    triage_summary: str = ""
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[RemediationAction] = Field(default_factory=list)
    verification_history: list[dict[str, Any]] = Field(default_factory=list)
    #: Tool results gathered on demand, keyed "iter<N>:<tool>".
    evidence: dict[str, Any] = Field(default_factory=dict)
    #: Similar past incidents recalled from runbook memory (episodic memory).
    recalled_episodes: list[dict[str, Any]] = Field(default_factory=list)
    #: Every guardrail that fired: injections quarantined, secrets redacted, actions blocked.
    guardrail_flags: list[dict[str, Any]] = Field(default_factory=list)
    #: Per-agent token/cost/latency, refreshed from the LLM ledger on each usage emit.
    agent_usage: dict[str, Any] = Field(default_factory=dict)
    postmortem: dict[str, Any] | None = None

    # Full ordered event log (also used to rebuild the timeline on reload).
    events: list[IncidentEvent] = Field(default_factory=list)

    auto_approve: bool = False
    #: Set by the operator via the cancel endpoint; checked at the top of every pass.
    cancel_requested: bool = False
    #: Audit trail for the people involved: who started it and who stopped it.
    created_by: str | None = None
    cancelled_by: str | None = None
    #: Which brain ran this incident: "gemini", "offline", or "injected" (tests, eval).
    #: Stamped on the record so an offline run is never mistaken for model reasoning.
    llm_mode: str = "gemini"

    # --- helpers ------------------------------------------------------------

    def pending_action(self) -> RemediationAction | None:
        """The most recently proposed-but-not-yet-resolved action, if any."""
        for action in reversed(self.actions):
            if action.status == ActionStatus.PROPOSED:
                return action
        return None

    def attempted_action_types(self) -> list[str]:
        """Actions already executed, so the loop can forbid repeating a dud.

        Enforced in code, not merely requested in a prompt - see
        `guardrails.manager.GuardrailManager.is_repeat_of_failed_action`.
        """
        return [a.action_type for a in self.actions if a.status == ActionStatus.EXECUTED]

    def failed_action_types(self) -> list[str]:
        """Executed actions that made no progress at all."""
        return [
            a.action_type
            for a in self.actions
            if a.status == ActionStatus.EXECUTED and a.result and not a.result.get("progress")
        ]

    def elapsed_s(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    def budget_remaining_usd(self) -> float:
        return max(0.0, self.max_cost_usd - self.total_cost_usd)

    def iteration_exhausted(self) -> bool:
        return self.iterations >= self.max_iterations

    def cost_exhausted(self) -> bool:
        return self.total_cost_usd >= self.max_cost_usd

    def wallclock_exhausted(self) -> bool:
        return self.elapsed_s() >= self.max_wallclock_s

    def budget_exhausted(self) -> bool:
        """True when ANY hard ceiling is reached."""
        return self.iteration_exhausted() or self.cost_exhausted() or self.wallclock_exhausted()

    def exhausted_reason(self) -> HaltReason | None:
        """Which ceiling tripped, if any. Checked at the top of every loop pass."""
        if self.cancel_requested:
            return HaltReason.OPERATOR_CANCELLED
        if self.iteration_exhausted():
            return HaltReason.ITERATION_CEILING
        if self.cost_exhausted():
            return HaltReason.COST_CEILING
        if self.wallclock_exhausted():
            return HaltReason.WALLCLOCK_CEILING
        return None

    def is_finalized(self) -> bool:
        """True once the run is fully wrapped up, not merely terminal.

        A terminal `status` is set the moment the loop decides to stop. The
        postmortem, the final usage figures and `ended_at` land a moment later.
        Anything that wants the COMPLETE record - the API, the evaluation harness,
        a UI that closes on completion - must wait for this, or it will
        occasionally read an incident that has stopped but not yet written itself
        up. Cheap race, annoying bug.
        """
        return self.ended_at is not None

    def to_summary(self) -> dict[str, Any]:
        """Compact view used by the list endpoint."""
        return {
            "incident_id": self.incident_id,
            "scenario_id": self.scenario_id,
            "service": self.service,
            "status": self.status.value,
            "severity": self.severity.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "iterations": self.iterations,
            "engine": self.engine,
            "halt_reason": self.halt_reason.value if self.halt_reason else None,
            "finalized": self.is_finalized(),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "resolved": self.status == IncidentStatus.RESOLVED,
        }

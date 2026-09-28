"""The migration state machine and the event envelope streamed to the UI.

A `JobState` owns a fleet of `RepoState` objects - one per repository the digital
employee is migrating. Each `RepoState` is driven by a Claude Agent SDK worker
running its own harnessed loop. Keeping all of it in typed objects is what makes a
long, multi-repo, multi-hour run auditable and replayable.
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


class RepoStatus(str, enum.Enum):
    QUEUED = "queued"
    MIGRATING = "migrating"          # SDK worker is editing + testing in a loop
    REVIEWING = "reviewing"          # reviewer agent is critiquing the diff
    AWAITING_APPROVAL = "awaiting_approval"  # human PR gate
    PR_OPEN = "pr_open"              # approved, "PR opened" (terminal-success)
    REJECTED = "rejected"            # human rejected the PR
    ESCALATED = "escalated"          # budget/verification failure -> needs a human
    FAILED = "failed"                # worker crashed


class JobStatus(str, enum.Enum):
    PLANNING = "planning"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class AgentName(str, enum.Enum):
    ORCHESTRATOR = "Orchestrator"     # the LangGraph fan-out / macro layer
    MIGRATOR = "Migrator"             # the Claude Agent SDK worker
    REVIEWER = "Reviewer"             # second-opinion agent on the diff
    GUARDRAIL = "Guardrail"           # the hook / policy plane


class EventType(str, enum.Enum):
    JOB_STARTED = "job_started"
    PLAN = "plan"
    REPO_STARTED = "repo_started"
    STEP = "step"                     # a worker reasoning/loop step
    TOOL_CALL = "tool_call"           # the worker invoked a tool (post-hook)
    EDIT_APPLIED = "edit_applied"     # a file was written in the worktree
    TESTS_RUN = "tests_run"           # the test-runner tool produced a verdict
    REVIEW = "review"                 # reviewer agent verdict on the diff
    AWAITING_APPROVAL = "awaiting_approval"
    PR_OPENED = "pr_opened"           # the mutating "merge/open PR" step
    REPO_DONE = "repo_done"           # a repo reached a terminal state
    GUARDRAIL_BLOCK = "guardrail_block"  # a hook blocked/redacted a tool call
    USAGE = "usage"                   # token / cost meter update
    JOB_SUMMARY = "job_summary"
    ERROR = "error"


class RepoState(BaseModel):
    """The evolving state of one repository's migration."""

    repo_id: str
    name: str
    status: RepoStatus = RepoStatus.QUEUED

    started_at: float | None = None
    ended_at: float | None = None

    # Loop-budget counters (the harness runs the loop; these are OUR ceilings).
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    # Work product.
    diff: str = ""                    # unified diff of the applied migration
    files_changed: list[str] = Field(default_factory=list)
    tests_passing: bool = False
    review_verdict: dict[str, Any] | None = None
    pr: dict[str, Any] | None = None  # {"number", "title", "url", "branch", "kind"}
    guardrail_blocks: list[dict[str, Any]] = Field(default_factory=list)
    escalation_reason: str = ""

    def to_summary(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "name": self.name,
            "status": self.status.value,
            "tests_passing": self.tests_passing,
            "files_changed": self.files_changed,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "cost_usd": round(self.cost_usd, 6),
        }


class MigrationEvent(BaseModel):
    """The envelope streamed over SSE. Mirrors the frontend contract exactly."""

    type: EventType
    ts: float = Field(default_factory=_now)
    job_id: str
    repo_id: str | None = None        # None for job-level events
    agent: AgentName | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class JobState(BaseModel):
    """A fleet migration job: one rule applied across many repositories."""

    job_id: str = Field(default_factory=lambda: _new_id("job"))
    rule_id: str
    title: str
    status: JobStatus = JobStatus.PLANNING
    mode: str = "stub"                # "live-sdk" | "stub" (which worker ran)

    started_at: float = Field(default_factory=_now)
    ended_at: float | None = None

    repos: list[RepoState] = Field(default_factory=list)
    events: list[MigrationEvent] = Field(default_factory=list)

    auto_approve: bool = False

    # Roll-up counters kept hot for the dashboard.
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def repo(self, repo_id: str) -> RepoState | None:
        return next((r for r in self.repos if r.repo_id == repo_id), None)

    def pending_repo(self) -> RepoState | None:
        return next((r for r in self.repos if r.status == RepoStatus.AWAITING_APPROVAL), None)

    def to_summary(self) -> dict[str, Any]:
        merged = sum(1 for r in self.repos if r.status == RepoStatus.PR_OPEN)
        return {
            "job_id": self.job_id,
            "run_id": self.job_id,  # stable id the UI uses for detail/stream/approve
            "rule_id": self.rule_id,
            "title": self.title,
            "status": self.status.value,
            "mode": self.mode,
            "started_at": self.started_at,
            "repos_total": len(self.repos),
            "repos_merged": merged,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }

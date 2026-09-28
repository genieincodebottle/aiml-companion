"""The live fleet orchestrator.

This is the interactive, streaming realization of the macro pipeline (the same shape
LangGraph expresses declaratively in `graph.py`). It owns jobs, per-repo worktrees,
SSE subscribers, and the human-approval futures, and it fans out across repositories
with bounded concurrency - exactly the "digital staffing" pattern: many autonomous
workers, each on the Claude Agent SDK harness, supervised by one control plane.

Per repository the flow is:

    MIGRATE (SDK/stub worker loop) -> REVIEW (reviewer agent) -> [HUMAN PR GATE] -> OPEN PR

The worker owns the inner agent loop (that is the harness's job); THIS layer owns
everything around it: concurrency, the approval gate, budget roll-ups, the event
stream, and turning a per-repo result into a fleet-level summary.
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..config import get_settings
from ..harness.base import Budget
from ..harness.reviewer import review as review_diff
from ..models.schemas import MigrationResult
from ..models.state import (
    AgentName,
    EventType,
    JobState,
    JobStatus,
    MigrationEvent,
    RepoState,
    RepoStatus,
)
from ..rulebook.rules import get_job, get_rule
from ..tools.memory import MemoryStore
from ..vcs.base import CheckoutHandle, RepoTarget
from .pipeline import migrate_repo_core

logger = logging.getLogger(__name__)

_TERMINAL_REPO = {RepoStatus.PR_OPEN, RepoStatus.REJECTED, RepoStatus.ESCALATED, RepoStatus.FAILED}


class MigrationManager:
    """Owns all jobs, their worktrees, subscribers, and approval futures."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.jobs: dict[str, JobState] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        # Keyed by (job_id, repo_id): two concurrent runs share repo ids, so a bare
        # repo_id key would let run B's gate steal (and orphan) run A's approval.
        self._approvals: dict[tuple[str, str], asyncio.Future] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._memory: dict[str, MemoryStore] = {}
        self._targets: dict[str, dict[str, RepoTarget]] = {}  # job_id -> {repo_id: target}

    # --- lifecycle ----------------------------------------------------------

    def trigger(self, job_id: str, auto_approve: bool | None = None) -> JobState:
        job_def = get_job(job_id)
        if job_def is None:
            raise ValueError(f"unknown job '{job_id}'")
        rule = get_rule(job_def.rule_id)
        if rule is None:
            raise ValueError(f"job '{job_id}' references unknown rule '{job_def.rule_id}'")

        state = JobState(
            rule_id=rule.id,
            title=job_def.title,
            mode=self.settings.execution_mode,
            auto_approve=self.settings.auto_approve if auto_approve is None else auto_approve,
            repos=[RepoState(repo_id=t.slug, name=t.name) for t in job_def.targets],
        )
        self.jobs[state.job_id] = state
        self._targets[state.job_id] = {t.slug: t for t in job_def.targets}
        self._subscribers[state.job_id] = set()
        self._memory[state.job_id] = MemoryStore(self.settings.data_dir, state.job_id)
        self._tasks[state.job_id] = asyncio.create_task(self._run(state, job_def))
        return state

    def get(self, job_id: str) -> JobState | None:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.to_summary() for j in sorted(self.jobs.values(), key=lambda x: x.started_at, reverse=True)]

    # --- pub / sub for SSE --------------------------------------------------

    async def _emit(self, event: MigrationEvent) -> None:
        job = self.jobs.get(event.job_id)
        if job is not None:
            job.events.append(event)
        for q in list(self._subscribers.get(event.job_id, set())):
            await q.put(event)

    async def subscribe(self, job_id: str):
        """Replay history, then stream live until the job is terminal."""
        job = self.jobs.get(job_id)
        if job is None:
            return
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, set()).add(q)
        try:
            for ev in list(job.events):
                yield ev
            if job.status in (JobStatus.DONE, JobStatus.FAILED):
                return
            while True:
                ev = await q.get()
                if ev is None:  # sentinel: stream closed
                    return
                yield ev
        finally:
            self._subscribers.get(job_id, set()).discard(q)

    # --- human-in-the-loop approval ----------------------------------------

    def resolve_approval(self, job_id: str, repo_id: str, decision: str, note: str = "") -> bool:
        fut = self._approvals.get((job_id, repo_id))
        if fut is None or fut.done():
            return False
        fut.set_result({"decision": decision, "note": note})
        return True

    async def _await_approval(self, job_id: str, repo_id: str) -> dict:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._approvals[(job_id, repo_id)] = fut
        try:
            return await fut
        finally:
            self._approvals.pop((job_id, repo_id), None)

    # --- the job (fleet fan-out) -------------------------------------------

    async def _run(self, job: JobState, job_def) -> None:
        try:
            job.status = JobStatus.RUNNING
            await self._emit(MigrationEvent(
                type=EventType.JOB_STARTED, job_id=job.job_id, agent=AgentName.ORCHESTRATOR,
                payload={"title": job.title, "mode": job.mode, "repos": [r.name for r in job.repos]},
            ))
            await self._emit(MigrationEvent(
                type=EventType.PLAN, job_id=job.job_id, agent=AgentName.ORCHESTRATOR,
                payload={
                    "rule_id": job.rule_id,
                    "repo_count": len(job.repos),
                    "concurrency": self.settings.fan_out_concurrency,
                    "auto_approve": job.auto_approve,
                },
            ))

            # Bounded-concurrency fan-out: many autonomous workers, one control plane.
            sem = asyncio.Semaphore(max(1, self.settings.fan_out_concurrency))
            await asyncio.gather(*(self._run_repo(job, job_def, repo, sem) for repo in job.repos))

            job.status = JobStatus.DONE
            job.ended_at = time.time()
            await self._emit(MigrationEvent(
                type=EventType.JOB_SUMMARY, job_id=job.job_id, agent=AgentName.ORCHESTRATOR,
                payload=self._summary_payload(job),
            ))
        except Exception as exc:  # noqa: BLE001 - never swallow; surface it
            logger.exception("job %s crashed: %s", job.job_id, exc)
            job.status = JobStatus.FAILED
            job.ended_at = time.time()
            await self._emit(MigrationEvent(
                type=EventType.ERROR, job_id=job.job_id, agent=AgentName.ORCHESTRATOR,
                payload={"message": f"Job crashed: {exc}"},
            ))
        finally:
            for q in list(self._subscribers.get(job.job_id, set())):
                await q.put(None)

    async def _run_repo(self, job: JobState, job_def, repo: RepoState, sem: asyncio.Semaphore) -> None:
        rule = get_rule(job.rule_id)
        try:
            handle: CheckoutHandle | None = None
            # ---- compute phase (migrate + review) under the concurrency cap ----
            async with sem:
                result, budget, handle = await self._migrate_one(job, repo, rule)
                repo.steps, repo.tool_calls = budget.steps, budget.tool_calls
                repo.tokens, repo.cost_usd = budget.tokens, budget.cost_usd
                repo.diff, repo.files_changed = result.diff, result.files_changed
                repo.tests_passing = result.tests_passing
                job.total_tokens += budget.tokens
                job.total_cost_usd += budget.cost_usd
                await self._emit(MigrationEvent(
                    type=EventType.USAGE, job_id=job.job_id, repo_id=repo.repo_id,
                    payload={"tokens": budget.tokens, "cost_usd": round(budget.cost_usd, 6)},
                ))

                if result.escalate:
                    repo.status = RepoStatus.ESCALATED
                    repo.escalation_reason = result.escalation_reason
                    await self._repo_done(job, repo, reason=result.escalation_reason)
                    return

                # ---- reviewer agent (second opinion + tamper gate) ----
                repo.status = RepoStatus.REVIEWING
                verdict = review_diff(handle.worktree, result)
                repo.review_verdict = verdict.model_dump()
                await self._emit(MigrationEvent(
                    type=EventType.REVIEW, job_id=job.job_id, repo_id=repo.repo_id, agent=AgentName.REVIEWER,
                    payload=verdict.model_dump(),
                ))
                if not verdict.approve:
                    repo.status = RepoStatus.ESCALATED
                    repo.escalation_reason = "; ".join(verdict.reasons) or "reviewer rejected the change"
                    await self._repo_done(job, repo, reason=repo.escalation_reason)
                    return

            # ---- human PR-approval gate (outside the compute semaphore) ----
            if not job.auto_approve:
                repo.status = RepoStatus.AWAITING_APPROVAL
                await self._emit(MigrationEvent(
                    type=EventType.AWAITING_APPROVAL, job_id=job.job_id, repo_id=repo.repo_id,
                    agent=AgentName.ORCHESTRATOR,
                    payload={
                        "repo": repo.name,
                        "files_changed": repo.files_changed,
                        "diff": repo.diff,
                        "summary": result.summary,
                    },
                ))
                decision = await self._await_approval(job.job_id, repo.repo_id)
                if decision["decision"] != "approve":
                    repo.status = RepoStatus.REJECTED
                    await self._repo_done(job, repo, reason=decision.get("note") or "operator rejected the PR")
                    return

            # ---- open the PR (the mutating, post-approval step) ----
            pr = await self._open_pr(job, repo, result, handle)
            repo.pr = pr
            repo.status = RepoStatus.PR_OPEN
            await self._emit(MigrationEvent(
                type=EventType.PR_OPENED, job_id=job.job_id, repo_id=repo.repo_id, agent=AgentName.ORCHESTRATOR,
                payload=pr,
            ))
            await self._repo_done(job, repo, reason=f"PR opened: {pr.get('url', '')}")

        except Exception as exc:  # noqa: BLE001
            logger.exception("repo %s crashed: %s", repo.repo_id, exc)
            repo.status = RepoStatus.FAILED
            repo.escalation_reason = str(exc)
            await self._emit(MigrationEvent(
                type=EventType.ERROR, job_id=job.job_id, repo_id=repo.repo_id,
                payload={"message": f"Repo migration crashed: {exc}"},
            ))
            await self._repo_done(job, repo, reason=str(exc))

    async def _migrate_one(
        self, job: JobState, repo: RepoState, rule
    ) -> tuple[MigrationResult, Budget, CheckoutHandle]:
        repo.status = RepoStatus.MIGRATING
        repo.started_at = time.time()
        await self._emit(MigrationEvent(
            type=EventType.REPO_STARTED, job_id=job.job_id, repo_id=repo.repo_id, agent=AgentName.ORCHESTRATOR,
            payload={"repo": repo.name, "rule": rule.name},
        ))
        target = self._targets[job.job_id][repo.repo_id]
        # Delegate to the shared per-repo core so the streaming path and the LangGraph
        # batch path can never diverge. We pass our streaming `_emit` as the sink.
        result, budget, handle, blocks = await migrate_repo_core(
            self.settings, job.job_id, target, rule, self._memory[job.job_id], emit=self._emit,
        )
        repo.guardrail_blocks = blocks
        return result, budget, handle

    # --- helpers ------------------------------------------------------------

    async def _open_pr(
        self, job: JobState, repo: RepoState, result: MigrationResult, handle: CheckoutHandle
    ) -> dict:
        rule = get_rule(job.rule_id)
        title = f"[{repo.name}] {rule.name}"
        body = (
            f"Automated migration by the Migration Engineer ({job.mode} mode).\n\n"
            f"Rule: {rule.name}\n"
            f"Files changed: {', '.join(repo.files_changed) or 'none'}\n"
            f"Tests: {'passing' if repo.tests_passing else 'FAILING'}\n\n"
            f"{result.summary}\n\n```diff\n{repo.diff}\n```"
        )
        # git push + PR creation is blocking network/subprocess work -> run in a thread.
        pr = await asyncio.to_thread(handle.provider.publish, handle, title, body, repo.files_changed)
        return {
            "number": pr.number, "title": pr.title, "url": pr.url,
            "branch": pr.branch, "kind": pr.kind,
        }

    async def _repo_done(self, job: JobState, repo: RepoState, reason: str) -> None:
        repo.ended_at = time.time()
        await self._emit(MigrationEvent(
            type=EventType.REPO_DONE, job_id=job.job_id, repo_id=repo.repo_id, agent=AgentName.ORCHESTRATOR,
            payload={"status": repo.status.value, "reason": reason, **repo.to_summary()},
        ))

    def _summary_payload(self, job: JobState) -> dict:
        by_status: dict[str, int] = {}
        for r in job.repos:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        merged = sum(1 for r in job.repos if r.status == RepoStatus.PR_OPEN)
        return {
            "repos_total": len(job.repos),
            "repos_merged": merged,
            "by_status": by_status,
            "total_tokens": job.total_tokens,
            "total_cost_usd": round(job.total_cost_usd, 6),
            "mode": job.mode,
        }


_manager: MigrationManager | None = None


def get_manager() -> MigrationManager:
    global _manager
    if _manager is None:
        _manager = MigrationManager()
    return _manager

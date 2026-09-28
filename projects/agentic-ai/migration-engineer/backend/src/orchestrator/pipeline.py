"""The shared per-repo migration core.

Both orchestration layers call this so they can never drift apart:

  * `engine.py`  - the live, streaming, interactive control plane (SSE + human gate);
  * `graph.py`   - the declarative LangGraph batch pipeline (durable / non-interactive).

It selects a real VCS provider, clones the repo into a real git worktree on a fresh
feature branch, wires the budget, guardrail hooks, tracer, tools (with the repo's real
test command), and the chosen worker (SDK or stub), runs the worker, and returns the
result, the budget consumed, and the live checkout handle (which the caller uses to open
the pull request). It does NOT decide approval - that policy belongs to the caller.
"""

from __future__ import annotations

import asyncio

from ..config import Settings
from ..guardrails.hooks import HookManager
from ..harness.base import Budget, EmitFn, WorkerContext
from ..harness.worker import build_worker
from ..models.schemas import MigrationResult
from ..observability import Tracer
from ..rulebook.rules import MigrationRule
from ..tools.memory import MemoryStore
from ..tools.registry import ToolInvoker, build_tool_specs
from ..vcs.base import CheckoutHandle, RepoTarget
from ..vcs.factory import get_provider


async def _noop_emit(_event) -> None:  # used by the non-streaming batch path
    return None


async def migrate_repo_core(
    settings: Settings,
    job_id: str,
    target: RepoTarget,
    rule: MigrationRule,
    memory: MemoryStore,
    emit: EmitFn | None = None,
) -> tuple[MigrationResult, Budget, CheckoutHandle, list[dict]]:
    """Run one repository's migration on a real git worktree (no approval/review here)."""
    emit = emit or _noop_emit

    # Real VCS: clone the repo onto a fresh feature branch (git is blocking -> thread).
    provider = get_provider(settings, target)
    feature_branch = f"migration/{rule.id}/{job_id[-6:]}"
    handle = await asyncio.to_thread(provider.checkout, target, job_id, feature_branch)

    budget = Budget(
        max_steps=settings.max_agent_steps,
        max_tool_calls=settings.max_tool_calls,
        max_cost_usd=settings.max_cost_usd,
    )
    hooks = HookManager(worktree=handle.worktree, enforce=not settings.disable_guardrails)
    tracer = Tracer(job_id, settings.data_dir)
    ctx = WorkerContext(
        job_id=job_id, repo_id=target.slug, repo_name=target.name,
        worktree=handle.worktree, rule=rule, budget=budget, emit=emit,
        tools=None, hooks=hooks,
    )
    ctx.tools = ToolInvoker(
        ctx,
        build_tool_specs(memory, test_command=target.test_command),
        tracer,
        simulate_usage=not settings.use_live_worker,
    )

    worker = build_worker(settings)
    result = await worker.run(ctx)
    return result, budget, handle, list(hooks.blocks)

"""The deterministic offline worker.

This is the worker that runs when there is no ANTHROPIC_API_KEY / SDK installed (or
when ME_FORCE_STUB=1, as in CI). It is NOT a mock: it drives the exact same tools,
guardrail hooks, budget, and event stream as the live Claude Agent SDK worker. The
only difference is *how it decides what to edit* - it applies the rule's scripted
`edit_steps` instead of reasoning, and it runs a genuine edit -> test -> edit loop,
so the offline demo shows real iteration (e.g. the auth-service needing a second pass
to add the `timezone` import after the tests fail with NameError).

Because it exercises the real contract, everything downstream - guardrails, review,
the approval gate, evals, traces - behaves identically to the live system.
"""

from __future__ import annotations

import re

from ..models.schemas import MigrationResult
from ..models.state import AgentName
from ..rulebook.rules import EditStep
from .base import BudgetExceeded, WorkerContext
from .diffing import compute_diff


class StubMigrator:
    """A scripted worker that migrates one repository deterministically."""

    async def run(self, ctx: WorkerContext) -> MigrationResult:
        rule = ctx.rule
        try:
            return await self._migrate(ctx, rule)
        except BudgetExceeded as exc:
            await ctx.step(AgentName.MIGRATOR, f"Budget ceiling hit: {exc}. Escalating to a human.")
            return MigrationResult(
                tests_passing=False,
                files_changed=list(ctx.tools_files_changed),
                escalate=True,
                escalation_reason=str(exc),
                steps_taken=ctx.budget.steps,
            )

    async def _migrate(self, ctx: WorkerContext, rule) -> MigrationResult:
        await ctx.step(AgentName.MIGRATOR, f"Starting '{rule.name}' on {ctx.repo_name}")

        # 1) Recall what earlier repos in this job already learned (cross-repo memory).
        recalled = await ctx.tools.call("recall_memory", query=rule.name, top_k=3)
        if recalled.get("count"):
            await ctx.step(AgentName.MIGRATOR, f"Recalled {recalled['count']} gotcha(s) from earlier repos")

        # 2) Pull the migration guidance from the rulebook (RAG).
        await ctx.tools.call("search_migration_rules", query=f"{rule.name} {' '.join(rule.tags)}")

        # 3) Locate the files that need changing.
        hits = await ctx.tools.call("grep", pattern=rule.detect)
        targets = sorted({m["path"] for m in hits.get("matches", [])})
        if not targets:
            await ctx.step(AgentName.MIGRATOR, "No occurrences found; nothing to migrate here.")
            return MigrationResult(
                tests_passing=True, files_changed=[], summary="no occurrences; nothing to migrate",
                steps_taken=ctx.budget.steps,
            )
        await ctx.step(AgentName.MIGRATOR, f"Found {len(targets)} file(s) to change: {', '.join(targets)}")

        # 4) Apply the ordered edit steps, running tests between each - the loop.
        passing = False
        used_import_fix = False
        for i, step in enumerate(rule.edit_steps, start=1):
            await ctx.step(AgentName.MIGRATOR, f"Edit step {i}/{len(rule.edit_steps)}: {step.description}")
            changed_any = False
            for path in targets:
                content = (await ctx.tools.call("read_file", path=path)).get("content", "")
                new = _apply_edit_step(step, content)
                if new != content:
                    await ctx.tools.call("write_file", path=path, content=new)
                    changed_any = True
            if step.kind == "ensure_import" and changed_any:
                used_import_fix = True

            verdict = await ctx.tools.call("run_tests")
            if verdict.get("passed"):
                passing = True
                await ctx.step(AgentName.MIGRATOR, "Tests are green; migration complete.")
                break
            await ctx.step(
                AgentName.MIGRATOR,
                f"Tests still red after step {i}; iterating. ({_last_line(verdict.get('output', ''))})",
            )

        # 5) Persist a reusable gotcha for the rest of the fleet.
        if used_import_fix and passing:
            await ctx.tools.call(
                "record_memory",
                note=(
                    f"For '{rule.name}', replacing the call is not enough: you must also add the "
                    f"`timezone` import or the tests fail with NameError."
                ),
                repo=ctx.repo_name,
                tags=" ".join(rule.tags),
            )

        diff = compute_diff(ctx.worktree)
        summary = (
            f"Migrated {len(ctx.tools_files_changed)} file(s) for '{rule.name}'"
            + ("; tests green" if passing else "; tests still failing")
        )
        return MigrationResult(
            tests_passing=passing,
            files_changed=list(ctx.tools_files_changed),
            diff=diff,
            summary=summary,
            steps_taken=ctx.budget.steps,
            escalate=not passing,
            escalation_reason="" if passing else "tests did not pass within the edit-step budget",
        )


# --- edit-step application -------------------------------------------------


def _apply_edit_step(step: EditStep, content: str) -> str:
    if step.kind == "regex_replace":
        return re.sub(step.find, step.replace, content)
    if step.kind == "ensure_import":
        return _ensure_symbol_import(content, step.module, step.symbol)
    return content


def _ensure_symbol_import(content: str, module: str, symbol: str) -> str:
    """Ensure `from {module} import ... {symbol} ...` is present."""
    line_rx = re.compile(rf"^from {re.escape(module)} import (?P<syms>.+)$", re.M)
    m = line_rx.search(content)
    if m:
        syms = [s.strip() for s in m.group("syms").split(",")]
        if symbol in syms:
            return content  # already imported - idempotent
        syms.append(symbol)
        new_line = f"from {module} import {', '.join(syms)}"
        return content[: m.start()] + new_line + content[m.end():]
    # No existing import line: add one after the module docstring / at the top.
    return f"from {module} import {symbol}\n" + content


def _last_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

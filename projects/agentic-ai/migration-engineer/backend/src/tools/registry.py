"""The tool registry and the `ToolInvoker`.

`build_tool_specs()` produces the single list of tools the worker can use. That list
is consumed two ways:

  * the stub worker calls `ToolInvoker.call(name, **kwargs)` directly;
  * the live worker turns the same specs into an in-process Claude Agent SDK MCP
    server (see sdk_agent.py), whose handlers ALSO call `ToolInvoker.call`.

So no matter which worker runs, every tool call passes through one place that applies
the guardrail hooks, charges the budget, records a trace span, and emits an event.
That single choke point is what makes the system observable and safe by construction.
"""

from __future__ import annotations

import asyncio

from ..harness.base import ToolSpec, WorkerContext
from ..models.state import AgentName, EventType
from ..observability import Tracer
from . import repo, testrunner
from .memory import MemoryStore
from .rules import search_migration_rules

# Blended simulated price (USD / 1M tokens) used to give the cost ceiling something to
# meter in stub mode. In live mode the worker adds the SDK's real reported usage too.
_SIM_PRICE_PER_1M = 5.0
_MAX_EVENT_TEXT = 600


def build_tool_specs(memory: MemoryStore, test_command: tuple[str, ...] | None = None) -> list[ToolSpec]:
    def _run_tests(worktree, **_kwargs):
        return testrunner.run_tests(worktree, command=test_command)

    return [
        ToolSpec(
            name="search_migration_rules",
            description="Retrieve migration guidance from the rulebook for a described change.",
            handler=search_migration_rules,
            input_schema={"query": "string", "top_k": "integer?"},
        ),
        ToolSpec(
            name="list_files",
            description="List files in the repository (optionally by glob).",
            handler=repo.list_files,
            input_schema={"glob": "string?"},
        ),
        ToolSpec(
            name="read_file",
            description="Read a file's contents (path relative to the repo root).",
            handler=repo.read_file,
            input_schema={"path": "string"},
        ),
        ToolSpec(
            name="grep",
            description="Search the repository for a regex pattern.",
            handler=repo.grep,
            input_schema={"pattern": "string", "glob": "string?"},
        ),
        ToolSpec(
            name="write_file",
            description="Write file contents (path relative to the repo root). Mutating.",
            handler=repo.write_file,
            mutating=True,
            input_schema={"path": "string", "content": "string"},
        ),
        ToolSpec(
            name="run_tests",
            description="Run the repository's tests and return pass/fail with output.",
            handler=_run_tests,
            input_schema={},
        ),
        ToolSpec(
            name="record_memory",
            description="Record a reusable gotcha for later repos in this job.",
            handler=memory.record,
            input_schema={"note": "string", "repo": "string?", "tags": "string?"},
        ),
        ToolSpec(
            name="recall_memory",
            description="Recall gotchas recorded by earlier repos in this job.",
            handler=memory.recall,
            input_schema={"query": "string", "top_k": "integer?"},
        ),
    ]


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_EVENT_TEXT else text[:_MAX_EVENT_TEXT] + " ...[truncated]"


class ToolInvoker:
    """The one place every tool call flows through, in either execution mode."""

    def __init__(
        self, ctx: WorkerContext, specs: list[ToolSpec], tracer: Tracer, simulate_usage: bool = True
    ) -> None:
        self.ctx = ctx
        self.tracer = tracer
        self._specs = {s.name: s for s in specs}
        # In live-sdk mode the SDK reports REAL token usage, so we don't fabricate any
        # here (that would double-count); in stub mode we simulate a little so the cost
        # ceiling has something to meter.
        self.simulate_usage = simulate_usage

    @property
    def tool_names(self) -> list[str]:
        return list(self._specs)

    async def call(self, name: str, **kwargs) -> dict:
        spec = self._specs.get(name)
        if spec is None:
            return {"error": f"unknown tool '{name}'"}

        # 1) GUARDRAIL HOOK (PreToolUse) - vet + redact BEFORE anything runs.
        decision = self.ctx.hooks.pre_tool_use(name, spec.mutating, kwargs)
        if not decision.allow:
            await self.ctx.event(
                EventType.GUARDRAIL_BLOCK,
                AgentName.GUARDRAIL,
                tool=name,
                reason=decision.reason,
                target=str(kwargs.get("path", "")),
            )
            self.ctx.budget.charge_tool(tokens=1, cost=0.0)
            return {"blocked": True, "reason": decision.reason}

        # The redacted copy is for LOGGING/EVENTS only. Executing with it would write
        # "***REDACTED***" into real source files; the worktree's own content is not a
        # secret to itself, so the handler gets the original input.
        log_input = decision.tool_input or kwargs

        # 2) EXECUTE inside a trace span. Handlers are sync (file IO / subprocess test
        # runs), so run them in a thread to keep the event loop responsive during the
        # concurrent fleet fan-out and live SSE streaming. A handler error becomes an
        # observation the worker can react to, never a crash of the whole repo run.
        with self.tracer.span(f"tool:{name}", tool=name, mutating=spec.mutating):
            try:
                result = await asyncio.to_thread(spec.handler, self.ctx.worktree, **kwargs)
            except Exception as exc:  # noqa: BLE001
                result = {"error": f"{type(exc).__name__}: {exc}"}

        # 3) REDACT outputs, then meter the budget.
        result = self._redact_result(result)
        if self.simulate_usage:
            tokens = max(1, (len(str(log_input)) + len(str(result))) // 4)
            cost = tokens / 1_000_000 * _SIM_PRICE_PER_1M
        else:
            tokens, cost = 0, 0.0

        # 4) EMIT the appropriate event(s).
        await self.ctx.event(
            EventType.TOOL_CALL,
            AgentName.MIGRATOR,
            tool=name,
            input=_truncate(str(log_input)),
            redactions=decision.redactions,
            summary=self._summarize(name, result),
        )
        if name == "write_file" and not result.get("blocked") and not result.get("error"):
            path = kwargs.get("path", "")
            if path not in self.ctx.tools_files_changed:
                self.ctx.tools_files_changed.append(path)
            await self.ctx.event(EventType.EDIT_APPLIED, AgentName.MIGRATOR, path=path, bytes=result.get("bytes", 0))
        if name == "run_tests":
            await self.ctx.event(
                EventType.TESTS_RUN,
                AgentName.MIGRATOR,
                passed=bool(result.get("passed")),
                output=_truncate(str(result.get("output", ""))),
            )

        self.ctx.budget.charge_tool(tokens=tokens, cost=cost)
        return result

    # --- helpers ------------------------------------------------------------

    def _redact_result(self, result: dict) -> dict:
        out = {}
        for k, v in result.items():
            if isinstance(v, str):
                out[k], _ = self.ctx.hooks.redact(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _summarize(name: str, result: dict) -> str:
        if result.get("blocked"):
            return f"blocked: {result.get('reason')}"
        if result.get("error"):
            return f"error: {result.get('error')}"
        if name == "run_tests":
            return "tests passed" if result.get("passed") else "tests failed"
        if name == "grep":
            return f"{result.get('count', 0)} match(es)"
        if name == "list_files":
            return f"{result.get('count', 0)} file(s)"
        if name in ("search_migration_rules", "recall_memory"):
            return f"{result.get('count', 0)} result(s)"
        if name == "read_file":
            return f"read {result.get('path', '')}"
        if name == "write_file":
            return f"wrote {result.get('path', '')} ({result.get('bytes', 0)} bytes)"
        if name == "record_memory":
            return "memory recorded"
        return "ok"

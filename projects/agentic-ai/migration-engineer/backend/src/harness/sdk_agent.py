"""The live worker: a real Claude Agent SDK agent that migrates one repository.

This is the centrepiece of the "inherit the harness" argument. We do NOT write an
agent loop here. We hand the Claude Agent SDK:

  * an in-process MCP server whose tools are OUR tools (repo / tests / rules / memory),
    each routed back through the same `ToolInvoker` so guardrails, budget, and events
    apply exactly as they do offline;
  * a PreToolUse hook (a second safety net at the SDK layer);
  * the migration guidance as the task; and a tight `max_turns` / permission mode.

...and the SDK runs the observe -> act -> verify loop. Everything this file does is
*around* that loop. That division of labour is the whole point.

Requires `uv sync --extra sdk` and ANTHROPIC_API_KEY. When neither is present
the orchestrator never selects this worker (see harness/worker.py); the deterministic
stub runs instead, so the project is fully runnable with zero credentials.
"""

from __future__ import annotations

import json

from ..models.schemas import MigrationResult
from ..models.state import AgentName, EventType
from .base import BudgetExceeded, WorkerContext
from .diffing import compute_diff

_PY_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool}

_SYSTEM = (
    "You are an autonomous migration engineer. You modernize one repository by making "
    "the smallest correct change, guided by the migration rule you are given.\n"
    "Rules of engagement:\n"
    "  * Use ONLY the provided MCP tools (search_migration_rules, list_files, read_file, "
    "grep, write_file, run_tests, record_memory, recall_memory).\n"
    "  * NEVER edit the repository's tests. Make the real code correct instead.\n"
    "  * After each edit, call run_tests and read the output. Keep iterating until the "
    "tests pass. If you cannot make them pass, say you are escalating and stop.\n"
    "  * Keep the diff minimal and behaviour-preserving beyond the intended change."
)


class SdkMigrator:
    """Runs one repo's migration on the real Claude Agent SDK loop."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def run(self, ctx: WorkerContext) -> MigrationResult:
        # Imported here so the module only loads when the SDK is actually installed.
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            HookMatcher,
            ResultMessage,
            TextBlock,
            create_sdk_mcp_server,
            query,
            tool,
        )

        # --- expose OUR tools as an in-process SDK MCP server ---------------
        sdk_tools = []
        for spec in ctx.tools._specs.values():  # noqa: SLF001 - registry is ours
            schema = {k.rstrip("?"): _PY_TYPES.get(v.rstrip("?"), str) for k, v in spec.input_schema.items()}
            sdk_tools.append(self._make_sdk_tool(tool, ctx, spec.name, spec.description, schema))
        server = create_sdk_mcp_server(name="migration", version="0.1.0", tools=sdk_tools)
        allowed = [f"mcp__migration__{spec.name}" for spec in ctx.tools._specs.values()]  # noqa: SLF001

        # --- a PreToolUse hook at the SDK layer (defense in depth) ----------
        async def pre_tool_use(input_data, tool_use_id, context):  # noqa: ANN001
            # An audit breadcrumb at the SDK layer; hard enforcement lives in ToolInvoker.
            name = input_data.get("tool_name", "")
            await ctx.event(EventType.STEP, AgentName.GUARDRAIL, text=f"PreToolUse hook vetted {name}")
            return {}

        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=_SYSTEM,
            mcp_servers={"migration": server},
            allowed_tools=allowed,
            disallowed_tools=["Bash", "Write", "Edit", "WebSearch", "WebFetch"],  # only our MCP tools
            permission_mode="acceptEdits",
            max_turns=ctx.budget.max_steps,
            cwd=str(ctx.worktree),
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[pre_tool_use])]},
        )

        prompt = (
            f"Repository: {ctx.repo_name}\n"
            f"Migration rule '{ctx.rule.name}':\n{ctx.rule.guidance}\n\n"
            "First call search_migration_rules and recall_memory, then locate and fix every "
            "occurrence, running the tests until they pass. When done, reply with a one-line summary."
        )

        escalated, reason = False, ""
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            await ctx.step(AgentName.MIGRATOR, block.text.strip()[:300])
                elif isinstance(message, ResultMessage):
                    # Add the SDK's REAL reported usage to our budget meter.
                    ctx.budget.tokens += int((message.usage or {}).get("output_tokens", 0)) + int(
                        (message.usage or {}).get("input_tokens", 0)
                    )
                    ctx.budget.cost_usd += float(message.total_cost_usd or 0.0)
        except BudgetExceeded as exc:
            escalated, reason = True, str(exc)
            await ctx.step(AgentName.MIGRATOR, f"Budget ceiling hit: {exc}. Escalating.")

        # --- authoritative verdict: run the tests ourselves, then diff ------
        try:
            verdict = await ctx.tools.call("run_tests")
        except BudgetExceeded:  # ceiling already hit mid-run; the verdict stays "not passing"
            verdict = {}
        passing = bool(verdict.get("passed")) and not escalated
        diff = compute_diff(ctx.worktree)
        return MigrationResult(
            tests_passing=passing,
            files_changed=list(ctx.tools_files_changed),
            diff=diff,
            summary=f"SDK migration of {ctx.repo_name}: {'tests green' if passing else 'not resolved'}",
            steps_taken=ctx.budget.steps,
            escalate=not passing,
            escalation_reason=reason or ("" if passing else "tests did not pass"),
        )

    @staticmethod
    def _make_sdk_tool(tool_decorator, ctx: WorkerContext, name: str, description: str, schema: dict):
        """Build one SDK tool whose handler routes back through our ToolInvoker."""

        @tool_decorator(name, description, schema)
        async def _handler(args):
            result = await ctx.tools.call(name, **args)
            return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

        return _handler

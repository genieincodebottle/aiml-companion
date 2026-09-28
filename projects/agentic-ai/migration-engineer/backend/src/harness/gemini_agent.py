"""The Gemini live worker: a function-calling loop over the same harness contract.

The primary live worker is the Claude Agent SDK (`sdk_agent.py`), where the loop is
inherited. When only a Gemini key is configured we still want a *real* LLM driving the
migration, so this worker runs Gemini 2.5 with native function calling over the exact
same `ToolInvoker` - which means the guardrail hooks, budget ceilings, redaction, and
event stream all apply unchanged. The seam in `harness/worker.py` picks it exactly like
it picks the stub: the orchestrator never knows which worker it got.

Requires `pip install google-genai` and GEMINI_API_KEY (or GOOGLE_API_KEY).
"""

from __future__ import annotations

import json

from ..models.schemas import MigrationResult
from ..models.state import AgentName, EventType
from .base import BudgetExceeded, WorkerContext
from .diffing import compute_diff

# Gemini function-calling schema types for our compact tool schema notation.
_GENAI_TYPES = {"string": "STRING", "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN"}

# gemini-2.5-flash list price (USD / 1M tokens) so the cost ceiling meters real usage.
_PRICE_IN_PER_1M = 0.30
_PRICE_OUT_PER_1M = 2.50

_SYSTEM = (
    "You are an autonomous migration engineer. You modernize one repository by making "
    "the smallest correct change, guided by the migration rule you are given.\n"
    "Rules of engagement:\n"
    "  * Use ONLY the provided tools (search_migration_rules, list_files, read_file, "
    "grep, write_file, run_tests, record_memory, recall_memory).\n"
    "  * NEVER edit the repository's tests. Make the real code correct instead.\n"
    "  * After each edit, call run_tests and read the output. Keep iterating until the "
    "tests pass. If you cannot make them pass, say you are escalating and stop.\n"
    "  * Keep the diff minimal and behaviour-preserving beyond the intended change.\n"
    "  * When the tests pass, reply with a one-line summary and stop calling tools."
)


def build_function_declarations(specs: dict) -> list[dict]:
    """Convert our compact ToolSpec schemas to Gemini function declarations.

    Our notation: {"query": "string", "top_k": "integer?"} - a trailing '?' marks the
    parameter optional. Pure dicts so this is unit-testable without google-genai.
    """
    decls = []
    for spec in specs.values():
        properties = {}
        required = []
        for key, typ in spec.input_schema.items():
            name = key.rstrip("?")
            optional = key.endswith("?") or typ.endswith("?")
            properties[name] = {"type": _GENAI_TYPES.get(typ.rstrip("?"), "STRING")}
            if not optional:
                required.append(name)
        params = {"type": "OBJECT", "properties": properties}
        if required:
            params["required"] = required
        decl = {"name": spec.name, "description": spec.description}
        if properties:
            decl["parameters"] = params
        decls.append(decl)
    return decls


class GeminiMigrator:
    """Runs one repo's migration on a Gemini function-calling loop."""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    async def run(self, ctx: WorkerContext) -> MigrationResult:
        # Imported here so the module only loads when google-genai is installed.
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        declarations = build_function_declarations(ctx.tools._specs)  # noqa: SLF001 - registry is ours
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            tools=[types.Tool(function_declarations=declarations)],
            temperature=0.0,
        )

        prompt = (
            f"Repository: {ctx.repo_name}\n"
            f"Migration rule '{ctx.rule.name}':\n{ctx.rule.guidance}\n\n"
            "First call search_migration_rules and recall_memory, then locate and fix every "
            "occurrence, running the tests until they pass. When done, reply with a one-line summary."
        )
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

        escalated, reason = False, ""
        try:
            # The model loop: each turn Gemini either calls tools (which we execute
            # through the SAME ToolInvoker as every other mode) or finishes with text.
            while True:
                response = await client.aio.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                self._meter_usage(ctx, response)

                candidate = (response.candidates or [None])[0]
                if candidate is None or candidate.content is None:
                    escalated, reason = True, "Gemini returned no candidate (safety block or empty response)"
                    await ctx.step(AgentName.MIGRATOR, reason)
                    break
                contents.append(candidate.content)

                calls = [p.function_call for p in (candidate.content.parts or []) if p.function_call]
                texts = [p.text.strip() for p in (candidate.content.parts or []) if p.text and p.text.strip()]
                step_text = texts[0] if texts else f"Calling tool(s): {', '.join(c.name for c in calls)}"
                await ctx.step(AgentName.MIGRATOR, step_text[:300])

                if not calls:
                    break  # the model is done reasoning

                response_parts = []
                for call in calls:
                    result = await ctx.tools.call(call.name, **dict(call.args or {}))
                    response_parts.append(
                        types.Part.from_function_response(
                            name=call.name, response={"result": json.loads(json.dumps(result, default=str))}
                        )
                    )
                contents.append(types.Content(role="user", parts=response_parts))
        except BudgetExceeded as exc:
            escalated, reason = True, str(exc)
            # The budget is spent, so emit directly instead of charging another step.
            await ctx.event(EventType.STEP, AgentName.MIGRATOR, text=f"Budget ceiling hit: {exc}. Escalating.")
        except Exception as exc:  # network / API errors -> escalate, never crash the fleet
            escalated, reason = True, f"Gemini API error: {exc}"

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
            summary=f"Gemini migration of {ctx.repo_name}: {'tests green' if passing else 'not resolved'}",
            steps_taken=ctx.budget.steps,
            escalate=not passing,
            escalation_reason=reason or ("" if passing else "tests did not pass"),
        )

    @staticmethod
    def _meter_usage(ctx: WorkerContext, response) -> None:
        """Add Gemini's real reported token usage + list-price cost to the budget."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        tokens_in = int(usage.prompt_token_count or 0)
        tokens_out = int(usage.candidates_token_count or 0) + int(getattr(usage, "thoughts_token_count", 0) or 0)
        ctx.budget.tokens += tokens_in + tokens_out
        ctx.budget.cost_usd += tokens_in / 1e6 * _PRICE_IN_PER_1M + tokens_out / 1e6 * _PRICE_OUT_PER_1M

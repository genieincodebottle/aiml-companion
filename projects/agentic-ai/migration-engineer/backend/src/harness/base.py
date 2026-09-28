"""The harness contract shared by every worker, in EITHER execution mode.

This is the single most important design decision in the project, and the one an
interviewer is really probing when they ask "where is the harness?":

    We do not re-implement the agent loop. The loop is the Claude Agent SDK's job.
    What WE own - and what actually determines production reliability - is the stuff
    *around* the loop: the tool contract, the guardrail hooks, the budget ceilings,
    and the event stream. Those are defined here, ONCE, so that:

      * the real Claude Agent SDK worker (`sdk_agent.py`) and
      * the deterministic offline worker (`stub_agent.py`)

    are genuine drop-in replacements for each other. Same tools, same hooks, same
    budget, same events. The stub is not a fake - it exercises the identical
    contract, which is why the offline demo faithfully represents the live system.

A `WorkerContext` is everything a worker needs. A worker is any object with an
async `run(ctx) -> MigrationResult`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models.state import AgentName, EventType, MigrationEvent
from ..rulebook.rules import MigrationRule

EmitFn = Callable[[MigrationEvent], Awaitable[None]]


class BudgetExceeded(RuntimeError):
    """Raised by the budget when a ceiling is hit. Halts the worker cleanly.

    Unbounded consumption is the #1 way naive agents burn money and never
    terminate; the harness makes the ceilings explicit and enforced.
    """


@dataclass
class Budget:
    """Hard ceilings enforced on every worker, regardless of execution mode."""

    max_steps: int
    max_tool_calls: int
    max_cost_usd: float
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    def charge_step(self) -> None:
        self.steps += 1
        if self.steps > self.max_steps:
            raise BudgetExceeded(f"step ceiling reached ({self.max_steps})")

    def charge_tool(self, tokens: int = 0, cost: float = 0.0) -> None:
        self.tool_calls += 1
        self.tokens += tokens
        self.cost_usd += cost
        if self.tool_calls > self.max_tool_calls:
            raise BudgetExceeded(f"tool-call ceiling reached ({self.max_tool_calls})")
        if self.cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(f"cost ceiling reached (${self.max_cost_usd:.2f})")


@dataclass
class WorkerContext:
    """Everything a Migrator worker needs to migrate one repository."""

    job_id: str
    repo_id: str
    repo_name: str
    worktree: Path                    # isolated copy of the repo the worker edits
    rule: MigrationRule               # the migration to perform (guidance only)
    budget: Budget
    emit: EmitFn                      # stream an event to SSE subscribers
    # Injected collaborators (both modes share these exact objects).
    tools: Any                        # ToolInvoker (src/tools/registry.py)
    hooks: Any                        # HookManager (src/guardrails/hooks.py)
    # Files the worker has written this run (accumulated by the invoker for the diff).
    tools_files_changed: list[str] = field(default_factory=list)

    async def step(self, agent: AgentName, text: str, **payload: Any) -> None:
        """Record one reasoning/loop step and stream it. Charges the step budget."""
        self.budget.charge_step()
        await self.emit(
            MigrationEvent(
                type=EventType.STEP,
                job_id=self.job_id,
                repo_id=self.repo_id,
                agent=agent,
                payload={"text": text, "step": self.budget.steps, **payload},
            )
        )

    async def event(self, etype: EventType, agent: AgentName | None = None, **payload: Any) -> None:
        await self.emit(
            MigrationEvent(
                type=etype,
                job_id=self.job_id,
                repo_id=self.repo_id,
                agent=agent,
                payload=payload,
            )
        )


@dataclass
class ToolSpec:
    """Metadata + handler for one tool exposed to the worker.

    The same ToolSpec list is turned into (a) direct callables for the stub worker
    and (b) an in-process Claude Agent SDK MCP server for the live worker.
    """

    name: str
    description: str
    # handler(worktree, **kwargs) -> dict. Pure w.r.t. everything except the worktree.
    handler: Callable[..., dict]
    mutating: bool = False            # does it write to the worktree?
    input_schema: dict = field(default_factory=dict)

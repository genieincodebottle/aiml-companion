"""Structured output contracts for the agents.

In `live-sdk` mode these are the shapes the Claude Agent SDK worker and the reviewer
return; in `stub` mode the deterministic worker fills the same shapes. Either way the
orchestrator only ever sees validated objects, never raw text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MigrationResult(BaseModel):
    """What a Migrator worker reports after finishing one repository."""

    tests_passing: bool
    files_changed: list[str] = Field(default_factory=list)
    diff: str = ""
    summary: str = ""                 # one/two line description of what changed
    steps_taken: int = 0
    escalate: bool = False            # worker gave up (budget/no safe fix)
    escalation_reason: str = ""


class ReviewVerdict(BaseModel):
    """The reviewer agent's second opinion on a proposed migration diff."""

    approve: bool
    confidence: float = 0.5
    reasons: list[str] = Field(default_factory=list)
    # Anti-reward-hacking signal: did the diff weaken or delete the tests?
    tampered_with_tests: bool = False


class PullRequest(BaseModel):
    """The artifact produced at the (human-gated) end of a repo migration."""

    number: int
    title: str
    body: str

"""Comparing two Hermes sessions that did the same task.

The claim this project tests: once Hermes has written a skill for a task, doing
that task again should cost less.

Measuring that naively goes wrong, and the reason is worth understanding before
reading any number this module produces.

A second run is cheaper for two independent reasons:

1.  **The skill made it shorter.** Fewer tool calls, fewer trips round the
    agent loop, because the procedure was written down instead of rediscovered.
2.  **The prompt prefix was cached.** Hermes orders the prompt stable-first
    exactly so providers can cache it. Cached input tokens are billed at a
    fraction of normal input tokens, or not at all.

Only the first one is the skill loop working. A total-token delta mixes them
together and will happily show a large "improvement" on a run where the skill
was never loaded at all.

So this module reports:

    tool_calls        the headline. Caching cannot touch it. If the skill
                      shortened the procedure, this drops.
    api_calls         iterations of the agent loop, same reasoning.
    uncached_input    input_tokens minus cache_read_tokens, floored at zero.
    output_tokens     never cached, always real work.

A caveat on uncached_input: whether ``input_tokens`` already excludes cache
reads is provider-dependent. Anthropic reports them as separate fields;
some OpenAI-compatible gateways fold cache reads into the input count. This
module cannot tell which one you are on, so it prints the raw fields alongside
the derived number and leaves the judgement visible instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .state_db import Session


def uncached_input(session: Session) -> int:
    """Input tokens the provider actually had to read, best effort.

    Floored at zero because on a gateway that reports cache reads *outside*
    input_tokens, the subtraction goes negative and a negative token count is
    a worse lie than an undercount.
    """
    return max(session.input_tokens - session.cache_read_tokens, 0)


def billable_work(session: Session) -> int:
    """Tokens that were neither cached nor free. The closest single number
    to 'how much did this actually cost me', without needing prices."""
    return uncached_input(session) + session.output_tokens


@dataclass(frozen=True)
class Delta:
    """One metric, before and after, with the change."""

    name: str
    baseline: float
    candidate: float
    lower_is_better: bool = True

    @property
    def change(self) -> float:
        return self.candidate - self.baseline

    @property
    def pct_change(self) -> float | None:
        """None rather than a division-by-zero or a fake 100%.

        A baseline of zero happens for real: a run that used no cached tokens,
        or a task that made no tool calls. Reporting 'inf% better' there would
        be the single most misleading thing this tool could print.
        """
        if self.baseline == 0:
            return None
        return (self.change / self.baseline) * 100.0

    @property
    def improved(self) -> bool:
        if self.change == 0:
            return False
        return (self.change < 0) if self.lower_is_better else (self.change > 0)


@dataclass(frozen=True)
class Comparison:
    """The full before/after for one task run twice."""

    baseline: Session
    candidate: Session
    deltas: list[Delta]
    baseline_tools: dict[str, int]
    candidate_tools: dict[str, int]
    warnings: list[str]

    @property
    def verdict(self) -> str:
        """What the numbers support, stated no more strongly than that.

        tool_calls is the deciding metric because caching cannot affect it.
        Token movement alone is not evidence the skill did anything.
        """
        tool_delta = next((d for d in self.deltas if d.name == "tool_calls"), None)
        work_delta = next((d for d in self.deltas if d.name == "billable_work"), None)

        if tool_delta is None:
            return "inconclusive: no tool-call data"

        if tool_delta.change < 0:
            return "skill shortened the procedure"
        if tool_delta.change == 0:
            if work_delta is not None and work_delta.change < 0:
                return (
                    "same number of tool calls, fewer tokens: likely caching, "
                    "not the skill"
                )
            return "no change"
        return "second run was longer: read the skill, it probably did not match"

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_session": asdict(self.baseline),
            "candidate_session": asdict(self.candidate),
            "deltas": [
                {
                    "name": d.name,
                    "baseline": d.baseline,
                    "candidate": d.candidate,
                    "change": d.change,
                    "pct_change": d.pct_change,
                    "improved": d.improved,
                }
                for d in self.deltas
            ],
            "baseline_tools": self.baseline_tools,
            "candidate_tools": self.candidate_tools,
            "warnings": self.warnings,
            "verdict": self.verdict,
        }


def compare(
    baseline: Session,
    candidate: Session,
    *,
    baseline_tools: dict[str, int] | None = None,
    candidate_tools: dict[str, int] | None = None,
    baseline_recount: int | None = None,
    candidate_recount: int | None = None,
) -> Comparison:
    """Build the before/after for two sessions that ran the same task."""

    deltas = [
        Delta("tool_calls", baseline.tool_call_count, candidate.tool_call_count),
        Delta("api_calls", baseline.api_call_count, candidate.api_call_count),
        Delta("billable_work", billable_work(baseline), billable_work(candidate)),
        Delta("uncached_input", uncached_input(baseline), uncached_input(candidate)),
        Delta("output_tokens", baseline.output_tokens, candidate.output_tokens),
        Delta("cache_read_tokens", baseline.cache_read_tokens,
              candidate.cache_read_tokens, lower_is_better=False),
        Delta("total_tokens", baseline.total_tokens, candidate.total_tokens),
    ]

    warnings = list(_warnings(
        baseline, candidate, baseline_recount, candidate_recount
    ))

    return Comparison(
        baseline=baseline,
        candidate=candidate,
        deltas=deltas,
        baseline_tools=baseline_tools or {},
        candidate_tools=candidate_tools or {},
        warnings=warnings,
    )


def _warnings(
    baseline: Session,
    candidate: Session,
    baseline_recount: int | None,
    candidate_recount: int | None,
):
    """Everything that would make the comparison mean less than it looks."""

    if baseline.id == candidate.id:
        yield (
            "Both ids are the same session, so every delta is zero by "
            "construction. Run `sessions` and pick two different ids."
        )

    if baseline.model and candidate.model and baseline.model != candidate.model:
        yield (
            f"Different models: baseline ran {baseline.model}, candidate ran "
            f"{candidate.model}. The delta is not attributable to the skill."
        )

    if baseline.source and candidate.source and baseline.source != candidate.source:
        yield (
            f"Different entry points ({baseline.source} vs {candidate.source}). "
            "Gateway and CLI sessions can carry different toolsets."
        )

    if candidate.parent_session_id is not None:
        yield (
            "The candidate session has a parent_session_id, so it is a "
            "continuation or a post-compression child rather than a fresh "
            "session. A fresh session is the only fair comparison: a "
            "continuation inherits context the baseline never had."
        )

    for label, sess, recount in (
        ("baseline", baseline, baseline_recount),
        ("candidate", candidate, candidate_recount),
    ):
        if recount is None:
            continue
        if recount != sess.tool_call_count:
            yield (
                f"{label}: sessions.tool_call_count is {sess.tool_call_count} "
                f"but the messages table has {recount} tool rows. The session "
                "probably ended abnormally. Treat its numbers as approximate."
            )

    if baseline.tool_call_count == 0 and candidate.tool_call_count == 0:
        yield (
            "Neither session made a tool call, so there was no procedure to "
            "shorten. Pick a task that actually touches the filesystem or "
            "the network."
        )

    if baseline.cache_read_tokens == 0 and candidate.cache_read_tokens > 0:
        yield (
            "Only the candidate got cache reads. Some of its token saving is "
            "prompt caching, not the skill. tool_calls is the metric to read."
        )

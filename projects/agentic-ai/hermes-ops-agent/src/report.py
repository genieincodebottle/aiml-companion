"""Rendering a Comparison for a terminal and for a file.

Two rules here:

- Warnings print above the numbers, not below. A caveat under a table gets
  skipped; the whole point of the warning is that it changes how you read
  the table.
- No colour codes. This output goes into CI logs and into README paste as
  often as it goes to a terminal.
"""

from __future__ import annotations

import json
from typing import Iterable

from .metrics import Comparison
from .state_db import Session


def _fmt_int(n: float) -> str:
    return f"{int(n):,}"


def _fmt_pct(p: float | None) -> str:
    if p is None:
        return "    n/a"
    return f"{p:+6.1f}%"


def _arrow(improved: bool, change: float) -> str:
    if change == 0:
        return "="
    return "better" if improved else "worse"


def render_text(cmp_: Comparison) -> str:
    lines: list[str] = []
    b, c = cmp_.baseline, cmp_.candidate

    lines.append("=" * 72)
    lines.append("Hermes skill loop: did the second run cost less?")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  baseline   {b.id}   {b.model or 'unknown model'}   ({b.source or '?'})")
    lines.append(f"  candidate  {c.id}   {c.model or 'unknown model'}   ({c.source or '?'})")
    lines.append("")

    if cmp_.warnings:
        lines.append("-" * 72)
        lines.append("READ THIS FIRST")
        lines.append("-" * 72)
        for w in cmp_.warnings:
            for i, chunk in enumerate(_wrap(w, 68)):
                lines.append(("  * " if i == 0 else "    ") + chunk)
        lines.append("")

    lines.append("-" * 72)
    lines.append(f"{'metric':<20}{'baseline':>12}{'candidate':>12}{'change':>12}{'':>8}")
    lines.append("-" * 72)
    for d in cmp_.deltas:
        lines.append(
            f"{d.name:<20}"
            f"{_fmt_int(d.baseline):>12}"
            f"{_fmt_int(d.candidate):>12}"
            f"{_fmt_int(d.change):>12}"
            f"  {_fmt_pct(d.pct_change)}  {_arrow(d.improved, d.change)}"
        )
    lines.append("-" * 72)
    lines.append("")

    if cmp_.baseline_tools or cmp_.candidate_tools:
        lines.append("Tools used (this is where a skill shows up)")
        names = sorted(set(cmp_.baseline_tools) | set(cmp_.candidate_tools))
        width = max((len(n) for n in names), default=10)
        for n in names:
            bn = cmp_.baseline_tools.get(n, 0)
            cn = cmp_.candidate_tools.get(n, 0)
            note = ""
            if bn and not cn:
                note = "   <- gone in the second run"
            elif cn and not bn:
                note = "   <- new in the second run"
            lines.append(f"  {n:<{width}}  {bn:>4} -> {cn:<4}{note}")
        lines.append("")

    lines.append(f"VERDICT: {cmp_.verdict}")
    lines.append("")
    lines.append(
        "tool_calls is the metric to trust. Prompt caching can move every "
        "token number\nwithout the skill doing anything; it cannot remove a "
        "tool call."
    )
    return "\n".join(lines)


def render_json(cmp_: Comparison) -> str:
    return json.dumps(cmp_.to_dict(), indent=2, sort_keys=False)


def render_session_table(sessions: Iterable[Session]) -> str:
    """Plain listing, used by `main.py sessions` to help pick two ids."""
    rows = list(sessions)
    if not rows:
        return "No sessions found."

    head = (
        f"{'session id':<26}{'source':<10}{'tools':>6}{'api':>6}"
        f"{'in':>10}{'out':>9}{'cached':>9}  model"
    )
    out = [head, "-" * len(head)]
    for s in rows:
        out.append(
            f"{s.id[:24]:<26}{(s.source or '?'):<10}"
            f"{s.tool_call_count:>6}{s.api_call_count:>6}"
            f"{s.input_tokens:>10,}{s.output_tokens:>9,}"
            f"{s.cache_read_tokens:>9,}  {s.model or '?'}"
        )
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines

"""One place that decides which model the agents talk to.

Why this exists
---------------
Five agents each constructed `ChatGoogleGenerativeAI` directly, so there was no
single point at which the project could be pointed somewhere else. Without a
`GOOGLE_API_KEY` every one of them raised inside its own `try`, the graph ran to
completion with nothing in it, and `python -m src.agents` printed
"No report generated" and **exited 0**. A reader could run the pipeline, see
success, and have watched nothing happen.

`get_llm()` returns the real client when a key is present and a deterministic
stand-in when it is not.

What the stand-in is, and what it is not
----------------------------------------
It is NOT a model. It is a rule engine that answers the prompts these five
agents actually send, in the shapes they parse: three want
`with_structured_output(Schema, include_raw=True)` and two want a plain
`.invoke()` returning a message. It exists so the orchestration is watchable --
the parallel fan-out, the quality gate, the revision loop, the budget guardrail
-- without a key or a network.

Loud failure is the point
-------------------------
Every agent in this project catches its own exceptions and returns a degraded
state, so a wrong reply produces a run that completes and looks plausible. So
the stand-in builds real Pydantic objects through the real schema and lets a
mismatch raise, and it refuses a prompt it does not recognise rather than
answering it with something that will parse.
"""

from __future__ import annotations

import os
import re
from typing import Any

from .config import get_model_name


class OfflineFixtureError(RuntimeError):
    """An offline reply could not be built, or did not match its schema.

    Deliberately fatal. The alternative is a pipeline that completes with every
    agent silently degraded, which is worse than a crash because it looks like
    success.
    """


# The shipped .env.example fills every key with a placeholder, and graph.py
# calls load_dotenv(). A placeholder is a non-empty string, so a plain
# `os.getenv(...)` truthiness test says "there is a key here" and the offline
# fallback never fires. That sent every search to the real Tavily client, which
# raised, was caught, and returned [] -- so the pipeline ran to completion with
# zero sources and still printed a report.
_PLACEHOLDER_MARKERS = ("your-", "your_", "xxx", "changeme", "replace-me", "here")


def has_real_key(name: str) -> bool:
    """True only when the environment holds something that could be a key."""
    value = (os.getenv(name) or "").strip().strip('"').strip("'")
    if len(value) < 12:
        return False
    lowered = value.lower()
    return not any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def is_offline() -> bool:
    """True when the agents should use the stand-in.

    Explicit opt-in wins, so you can run offline while holding a valid key.
    """
    forced = os.getenv("RESEARCH_OFFLINE", "").strip().lower()
    if forced in ("1", "true", "yes"):
        return True
    if forced in ("0", "false", "no"):
        return False
    return not (has_real_key("GOOGLE_API_KEY") or has_real_key("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# Message objects shaped like the ones LangChain returns
# ---------------------------------------------------------------------------


class _OfflineMessage:
    """Carries the two attributes this codebase reads off a response.

    `usage_metadata` matters: `token_count()` returns 0 when it is missing and
    warns that the budget guardrail is blind. Reporting a plausible token count
    here keeps the guardrail demonstrable in offline runs.
    """

    def __init__(self, content: str, tokens: int) -> None:
        self.content = content
        self.text = content
        self.usage_metadata = {
            "input_tokens": int(tokens * 0.7),
            "output_tokens": int(tokens * 0.3),
            "total_tokens": tokens,
        }
        self.response_metadata: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"<OfflineMessage {self.content[:50]!r}>"


def _tokens_for(prompt: str, output: str) -> int:
    """A stable, prompt-dependent token estimate.

    Roughly four characters per token. It is an estimate and the docstring says
    so; the point is that the number MOVES with the work done, so the budget
    guardrail behaves like it does in a real run instead of being a constant.
    """
    return max(1, (len(prompt) + len(output)) // 4)


# ---------------------------------------------------------------------------
# Fixtures, one per schema the agents ask for
# ---------------------------------------------------------------------------


def _query_from(prompt: str) -> str:
    """Recover the topic under research, or raise.

    A stand-in that quietly researched an empty string would produce a report
    about nothing and look entirely plausible doing it.
    """
    for pattern in (
        r"Query:\s*(.+?)(?:\n|$)",
        r"research query into",  # planner phrasing, query follows
        r"Topic:\s*(.+?)(?:\n|$)",
    ):
        found = re.search(pattern, prompt)
        if found and found.groups():
            text = found.group(1).strip()
            if text:
                return text
    # Fall back to the longest quoted or line-leading fragment before giving up.
    lines = [l.strip() for l in prompt.splitlines() if l.strip()]
    if lines:
        return max(lines, key=len)[:200]
    raise OfflineFixtureError(
        "Offline stand-in could not find the research query in this prompt:\n"
        + prompt[:200]
    )


def _planner(schema, prompt: str):
    query = _query_from(prompt)
    limit = re.search(r"1-(\d+) focused sub-topics", prompt)
    max_topics = int(limit.group(1)) if limit else 3

    # Three angles a real planner would plausibly separate. Prefixed rather than
    # grammatically embedded: the query is usually already a question, and
    # "What is the current state of What are the latest trends in X?" is the
    # kind of sentence that makes a demo look broken.
    topic = query.rstrip("?").strip()
    angles = [
        f"Current state: {topic}",
        f"Supporting evidence: {topic}",
        f"Criticisms and limitations: {topic}",
    ][:max_topics]

    return schema(
        sub_topics=angles,
        research_plan=(
            "Split the question into current state, supporting evidence and "
            "known limitations, research each independently, then reconcile."
        ),
    )


def _analyst(schema, prompt: str):
    # The claims deliberately DISAGREE. A conflicts list that is always empty
    # means the cross-referencing half of this pipeline never runs, and the
    # reader never sees the thing the analyst exists to do.
    claim_cls = schema.model_fields["claims"].annotation.__args__[0]
    claims = [
        claim_cls(
            claim="Adoption has grown quickly over the last two years.",
            evidence="Two of the retrieved sources report year-on-year growth.",
            confidence="high",
            source_idx=0,
        ),
        claim_cls(
            claim="Measured reliability in production remains well below demo results.",
            evidence="A practitioner survey reports a large gap between benchmark and deployed performance.",
            confidence="medium",
            source_idx=1,
        ),
        claim_cls(
            claim="Costs fall steadily as models get cheaper.",
            evidence="Vendor pricing pages show repeated reductions.",
            confidence="medium",
            source_idx=2,
        ),
        claim_cls(
            claim="Total cost of ownership has risen for most teams.",
            evidence="The same survey reports higher spend despite lower unit prices.",
            confidence="medium",
            source_idx=1,
        ),
        claim_cls(
            claim="No independent benchmark covers this area yet.",
            evidence="Only a vendor blog post makes this claim; no third party confirms it.",
            confidence="low",
            source_idx=3,
        ),
    ]
    return schema(
        claims=claims,
        conflicts=[
            "Claim 3 says unit costs are falling while claim 4 says total spend is "
            "rising. Both can be true, and the disagreement is itself the finding.",
        ],
    )


# The exact strings the writer's two prompts carry. Keyed off the real prompts
# rather than a guess: the first version of this matched "revision" and
# "reviewer feedback", neither of which appears (the prompt says "revising" and
# "Reviewer issues to fix"), so the writer never revised, the reviewer never saw
# a revision, and the demo published a report the reviewer had scored 5 out of 10.
_REVISION_PROMPT_MARKER = "Reviewer issues to fix"
_REVISED_DRAFT_MARKER = "Revised after review"


def _reviewer(schema, prompt: str):
    # Fail the FIRST draft. A reviewer that always passes means the revision
    # loop, which is a whole node and a conditional edge, never executes in a
    # demo run and the reader never sees the graph go backwards.
    #
    # The reviewer's prompt embeds the draft, so the writer's own marker is the
    # signal for whether this draft has already been through a revision.
    is_revision = _REVISED_DRAFT_MARKER in prompt
    if is_revision:
        return schema(
            score=8,
            issues=[],
            suggestions=["Tighten the conclusion."],
            passed=True,
        )
    return schema(
        score=5,
        issues=[
            "The low-confidence claim is presented with the same weight as the rest.",
            "The cost contradiction is stated but never resolved.",
        ],
        suggestions=[
            "Mark the unverified claim explicitly as unconfirmed.",
            "Explain that unit cost and total spend can move in opposite directions.",
        ],
        passed=False,
    )


_SCHEMA_FIXTURES = {
    "PlannerOutput": _planner,
    "AnalystOutput": _analyst,
    "ReviewOutput": _reviewer,
}


# ---------------------------------------------------------------------------
# Free-text fixtures for the two agents that do not use structured output
# ---------------------------------------------------------------------------


def _synthesis(prompt: str) -> str:
    return (
        "The evidence splits into three groups.\n\n"
        "Growth is well supported: multiple independent sources report the same "
        "direction, and none contradicts it.\n\n"
        "Cost is contested. Unit prices are falling and total spend is rising, "
        "which are not in conflict once you separate price per call from number "
        "of calls. Any report that quotes only one of them is misleading.\n\n"
        "One claim rests on a single vendor source with no independent "
        "confirmation, and should be carried as unverified rather than dropped "
        "or asserted."
    )


def _report(prompt: str) -> str:
    revising = _REVISION_PROMPT_MARKER in prompt
    marker = (
        f"\n\n> {_REVISED_DRAFT_MARKER}: the unverified claim is now labelled, and "
        "the cost contradiction is explained rather than left standing.\n"
        if revising else ""
    )
    return (
        "# Research Report\n"
        f"{marker}\n"
        "## Summary\n\n"
        "Adoption is growing and the growth is well evidenced. Reliability in "
        "production lags demonstration results by a wide margin. Costs move in "
        "two directions at once, and reports that quote only one of them mislead.\n\n"
        "## Findings\n\n"
        "1. Adoption has grown quickly over the last two years (high confidence).\n"
        "2. Production reliability remains below benchmark results (medium confidence).\n"
        "3. Unit costs are falling while total spend is rising (medium confidence).\n\n"
        "## Uncertainties\n\n"
        "One claim is supported only by a vendor blog post with no independent "
        "confirmation. It is recorded here rather than asserted or discarded.\n\n"
        "## Conclusion\n\n"
        "The direction of travel is clear; the economics are not. Anyone "
        "planning spend should model calls per user rather than price per call.\n"
    )


_TEXT_ROUTES = (
    ("synthes", _synthesis),
    ("Known conflicts", _synthesis),
    ("report", _report),
    ("draft", _report),
)


# ---------------------------------------------------------------------------
# The stand-in itself
# ---------------------------------------------------------------------------


class _StructuredOffline:
    """What `with_structured_output(Schema, include_raw=True)` returns."""

    def __init__(self, schema, include_raw: bool) -> None:
        self.schema = schema
        self.include_raw = include_raw

    def invoke(self, prompt: str):
        name = self.schema.__name__
        builder = _SCHEMA_FIXTURES.get(name)
        if builder is None:
            raise OfflineFixtureError(
                f"No offline fixture for schema {name!r}. An agent was given a new "
                "structured output type without a matching offline reply, so the "
                "pipeline would otherwise have degraded silently."
            )
        try:
            parsed = builder(self.schema, prompt)
        except OfflineFixtureError:
            raise
        except Exception as exc:
            # Building through the real schema means a fixture that no longer
            # matches its model fails HERE, loudly, rather than surfacing as a
            # degraded agent three nodes later.
            raise OfflineFixtureError(
                f"Offline fixture for {name} does not satisfy the schema: {exc}"
            ) from exc

        raw = _OfflineMessage(parsed.model_dump_json(), _tokens_for(prompt, str(parsed)))
        if not self.include_raw:
            return parsed
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


class OfflineLLM:
    """Drop-in replacement for `ChatGoogleGenerativeAI` in this project."""

    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        self.model = model or "offline-stand-in"
        self.temperature = temperature
        self.prompts: list[str] = []

    def with_structured_output(self, schema, include_raw: bool = False):
        return _StructuredOffline(schema, include_raw)

    def invoke(self, prompt: Any) -> _OfflineMessage:
        text = prompt if isinstance(prompt, str) else str(prompt)
        self.prompts.append(text)
        for marker, handler in _TEXT_ROUTES:
            if marker.lower() in text.lower():
                body = handler(text)
                return _OfflineMessage(body, _tokens_for(text, body))
        raise OfflineFixtureError(
            "Offline stand-in has no handler for this prompt. An LLM call was "
            "added without a matching offline reply, so the run would otherwise "
            "have degraded silently. First 300 characters:\n" + text[:300]
        )



# langchain-google-genai maintains _FIXED_SAMPLING_AND_NO_PREFILL_MODELS, and
# every model in it discards temperature, top_k and top_p, warning once per
# request. Passing a parameter the model throws away is worse than not passing
# it: it invites you to explain the system's determinism in terms of a value
# that never reached the API.
_FIXED_SAMPLING_MODELS = frozenset({"gemini-3.5-flash-lite", "gemini-3.6-flash"})


def honours_temperature(model: str) -> bool:
    """False for models with fixed sampling defaults."""
    return (model or "").lower().rsplit("/", 1)[-1] not in _FIXED_SAMPLING_MODELS


def get_llm(temperature: float = 0.0):
    """The model the agents should use.

    Every agent calls this instead of constructing a client directly, so there
    is one place to point the project somewhere else.
    """
    if is_offline():
        return OfflineLLM(temperature=temperature)
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = get_model_name()
    if honours_temperature(model):
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    # Fixed-sampling model: sending temperature would be dropped and warned on
    # every single request.
    return ChatGoogleGenerativeAI(model=model)

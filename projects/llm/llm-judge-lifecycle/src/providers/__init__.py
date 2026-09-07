"""The only place in this project that talks to a model vendor.

Every call in every phase goes through :class:`Provider`. Four adapters
implement it - gemini (default), openai_compatible, anthropic, stub - and no
other module imports a vendor SDK. ``tests/test_layering.py`` fails the build
if one does.

Why bother, when a single-provider project would be shorter?

Because the central claim of this project cannot be tested inside one model.
A judge and the generator it grades have different interests, and if they are
the same model you cannot separate "this output is good" from "this output is
written the way I write" - the self-preference bias documented in Panickssery
et al. (2024). Controlling for it means pointing two roles at two vendors, and
that means a seam. The seam is here.

The secondary payoff is mundane and arrives sooner: vendor APIs churn. When a
model id retires or a parameter is renamed, exactly one file under this package
changes and the benchmark, the RART loop, the serving loop and the monitor are
untouched.

Adding a provider
-----------------
Write a module exposing ``build(role_config) -> Provider``, register it in
``_ADAPTERS`` below, and run ``pytest tests/test_providers.py``. The contract
test runs the same seven assertions against every registered adapter, so a new
one is correct-by-construction or it fails immediately.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..config import RoleConfig

# name -> module under this package exposing build(RoleConfig) -> Provider
_ADAPTERS = {
    "gemini": ".gemini",
    "openai_compatible": ".openai_compatible",
    "anthropic": ".anthropic",
    "stub": ".stub",
}


@dataclass
class Completion:
    """One model response, normalised across vendors.

    ``finish_reason`` is normalised to a small vocabulary and carries far more
    weight than its size suggests - see :meth:`Completion.was_truncated`.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    #: "stop" | "length" | "filter" | "other"
    finish_reason: str = "stop"
    model: str = ""
    provider: str = ""
    raw: Any = field(default=None, repr=False)

    @property
    def was_truncated(self) -> bool:
        return self.finish_reason == "length"


class TruncatedCompletion(RuntimeError):
    """The model stopped because it hit the output cap, not because it finished.

    This is raised rather than returned quietly, and the reason is worth stating
    plainly because it has bitten this codebase's sibling project hard enough to
    corrupt a published benchmark.

    A truncated response is a valid PREFIX. Truncated JSON fails to parse with
    "Unterminated string", which reads exactly like the model formatted its
    output wrongly - so you spend an hour fixing a prompt that was never broken.
    Truncated PROSE is worse, because it does not fail at all: a judge's reason
    cut off after one sentence still parses, still looks like a reason, and
    still gets appended to the generator's revision prompt. The pipeline runs
    green and the measurements are quietly wrong.

    On thinking-capable models this is not an edge case. Reasoning tokens are
    drawn from the SAME ``max_output_tokens`` budget as the visible answer, so a
    judgement needing 200 tokens of JSON can spend 1,900 tokens deliberating and
    emit a fragment. Raising the ceiling does not fix it; it buys more thinking.
    Turning thinking off for the structured call does. See ``gemini.py``.
    """


@runtime_checkable
class Provider(Protocol):
    """The whole surface. Deliberately four methods, not forty.

    A wide abstraction over model vendors ends up expressing the union of their
    features and the intersection of their guarantees. This one expresses the
    intersection of both, which is all four phases actually need.
    """

    name: str
    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        """Free text, or JSON when ``json_schema`` is given.

        When a schema is supplied the adapter must use the vendor's native
        structured-output mode where one exists. "Reply in JSON please" fails on
        roughly one call in ten; a schema-constrained decode does not fail at
        all, and at the volumes a tuning loop runs at that difference is the gap
        between a working pipeline and a flaky one.
        """
        ...

    def estimated_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Order-of-magnitude only. An estimate for the UI, never a bill."""
        ...


class Usage:
    """Per-process token and cost tally, kept per role.

    Split by role because the interesting cost question in this project is not
    "what did the run cost" but "what did the JUDGE cost, given it runs on every
    single artefact, plus once more per retry". Serving cost scales with the
    retry budget, and that number is what makes K a business decision rather
    than a hyperparameter.
    """

    def __init__(self) -> None:
        self.by_role: dict[str, dict[str, float]] = {}

    def add(self, role: str, comp: Completion, usd: float) -> None:
        slot = self.by_role.setdefault(
            role, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
        )
        slot["calls"] += 1
        slot["input_tokens"] += comp.input_tokens
        slot["output_tokens"] += comp.output_tokens
        slot["usd"] += usd

    @property
    def total_usd(self) -> float:
        return sum(s["usd"] for s in self.by_role.values())

    @property
    def total_calls(self) -> int:
        return int(sum(s["calls"] for s in self.by_role.values()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "by_role": {
                r: {**s, "usd": round(s["usd"], 6)} for r, s in self.by_role.items()
            },
            "total_calls": self.total_calls,
            "total_usd": round(self.total_usd, 6),
        }


def get_provider(role_config: RoleConfig) -> Provider:
    """Build the adapter named by a role's config."""
    module_name = _ADAPTERS.get(role_config.provider)
    if module_name is None:
        raise ValueError(
            f"unknown provider {role_config.provider!r}. "
            f"Registered: {sorted(_ADAPTERS)}. "
            "Add one by writing src/providers/<name>.py with build(role_config) "
            "and registering it in _ADAPTERS."
        )
    module = importlib.import_module(module_name, package=__package__)
    return module.build(role_config)


def registered_providers() -> list[str]:
    return sorted(_ADAPTERS)


def is_retryable(exc: Exception) -> bool:
    """Retry transient vendor failures only.

    Two rules, both learned the expensive way.

    First, a 400 for a retired model id must fail immediately and loudly.
    Retrying it four times makes the learner wait fifteen seconds for the same
    wrong answer, and the real message - "that model no longer exists" - is
    buried under retry noise.

    Second, and less obvious: a retry policy written against HTTP status codes
    misses every failure that happens BELOW the status code. A connection
    dropped mid-stream raises a transport error carrying no status at all. Those
    are exactly the failures a long batch job hits, and a tuning run that dies
    on call 40 of 118 because a socket closed has thrown away 39 completed calls
    for no reason. Hence the transport-level markers.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in (
            # HTTP-level
            "429", "resource_exhausted", "rate limit", "overloaded",
            "503", "unavailable", "500", "internal", "deadline", "timeout",
            # Transport-level: the request never landed, nothing was wrong with it
            "remoteprotocolerror", "server disconnected", "connectionerror",
            "connecterror", "connection reset", "connection aborted",
            "readerror", "writeerror", "protocolerror", "incomplete read",
            "ssl", "eof occurred",
        )
    )


__all__ = [
    "Completion",
    "Provider",
    "TruncatedCompletion",
    "Usage",
    "get_provider",
    "is_retryable",
    "registered_providers",
]

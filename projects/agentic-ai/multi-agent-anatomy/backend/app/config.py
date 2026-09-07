"""Single source of truth for model names, per-token prices and budgets.

Everything in this file goes stale. Model ids get renamed, prices get cut, and
context windows grow. That is exactly why it all lives in one file instead of
being sprinkled through the agents: when the vendor changes something, you edit
here and nowhere else.

Prices are USD per 1 million tokens. Update them against the provider's pricing
page before you quote any number from this project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env from the project root and from backend/, in that order, before
# anything below reads an environment variable. Without this the whole file
# silently falls back to defaults and live mode never turns on.
try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve()
    for _candidate in (_here.parents[2] / ".env", _here.parents[1] / ".env"):
        if _candidate.exists():
            load_dotenv(_candidate, override=False)
except ImportError:  # replay mode does not need python-dotenv
    pass

PRICES_LAST_CHECKED = "2026-08-04"


@dataclass(frozen=True)
class ModelPrice:
    """Per-1M-token prices for one model.

    cached_input is the price of an input token that hit the provider's prompt
    cache. It is the whole reason stage 3 and stage 6 are affordable: the
    orchestrator sends its long stable prefix twice per request.
    """

    name: str
    input_per_m: float
    cached_input_per_m: float
    output_per_m: float

    def cost(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        """Cost in USD. cached_tokens is a subset of input_tokens, not an extra."""
        fresh = max(input_tokens - cached_tokens, 0)
        return (
            fresh * self.input_per_m
            + cached_tokens * self.cached_input_per_m
            + output_tokens * self.output_per_m
        ) / 1_000_000


# The orchestrator is the only place a top-tier model earns its price. It plans
# once and merges once, and both calls decide what the user actually reads.
#
# Verified against models.list() on 2026-08-04. There is no gemini-3.6-pro; the
# newest available pro tier is the 3.1 preview. This is exactly the staleness
# this file exists to contain.
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "gemini-3.1-pro-preview")

# Sub-agents do bounded, schema-checked lookups. A cheap model is correct here,
# not a compromise. Four of the five agents run on it.
WORKER_MODEL = os.getenv("WORKER_MODEL", "gemini-3.6-flash")

# Stage 2 classification decides whether a request needs the full fan-out at
# all. It has to cost close to nothing or it defeats its own purpose.
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "gemini-3.5-flash-lite")

# If the top-tier id above has been renamed by the provider, fall back to this
# rather than failing the request. The trace records that it happened.
FALLBACK_MODEL = WORKER_MODEL

# Verified against https://ai.google.dev/gemini-api/docs/pricing on the date in
# PRICES_LAST_CHECKED above. Standard tier, not batch.
#
# Gemini 3.1 Pro is tiered on prompt size: the rates below are the
# "prompts <= 200k tokens" ones. Above 200k it is 4.00 / 0.40 / 18.00. This
# project's whole token budget is 60k, so it never crosses that line, but a
# system that let context grow unbounded would double its input price without
# any code change. That is worth knowing before you let a supervisor accumulate
# worker output.
PRICES: dict[str, ModelPrice] = {
    "gemini-3.1-pro-preview": ModelPrice("gemini-3.1-pro-preview", 2.00, 0.20, 12.00),
    "gemini-3.6-flash": ModelPrice("gemini-3.6-flash", 1.50, 0.15, 7.50),
    "gemini-3.5-flash-lite": ModelPrice("gemini-3.5-flash-lite", 0.30, 0.03, 2.50),
}


def price_for(model: str) -> ModelPrice:
    """Never raise on an unknown model. An unpriced call shows as zero cost in
    the trace, which is visibly wrong, and that is better than a 500."""
    return PRICES.get(model, ModelPrice(model, 0.0, 0.0, 0.0))


@dataclass(frozen=True)
class Budgets:
    """Set once at the edge (stage 1) and carried down. Nothing below invents
    its own. See budget.py for how the remainder is propagated."""

    # Whole-request deadline. Every per-agent timeout below must be smaller,
    # or the deadline is decorative.
    #
    # These are sized for live calls, where a top-tier plan and a fan-out of
    # flash calls is several seconds of real latency. Sizing them for replay
    # instead is a trap: the numbers look tighter, replay never notices, and
    # the first live run times out an agent that was working fine.
    request_deadline_s: float = 60.0

    # Per-agent timeouts. The fan-out inherits its slowest agent, so these are
    # what stop one bad tool from holding the whole request open.
    order_agent_timeout_s: float = 20.0
    shipping_agent_timeout_s: float = 20.0
    policy_agent_timeout_s: float = 25.0
    writer_agent_timeout_s: float = 25.0
    orchestrator_timeout_s: float = 30.0

    # Per-tool timeout, derived from the time the request has left. Well under
    # the agent slice above it, so a hung tool is cut by its own budget and the
    # agent survives to report the gap. The 12s latency toggle is set above this
    # number and below the agent timeout, deliberately.
    tool_timeout_s: float = 8.0

    # Token budget for the whole request, split across all five agents.
    request_token_budget: int = 60_000

    # Hard caps that stop infinite delegation. The orchestrator is an agent and
    # can in principle delegate to itself forever.
    max_delegation_depth: int = 2
    max_orchestrator_iterations: int = 3


BUDGETS = Budgets()

# Prompt versions travel on every span. When an answer goes wrong you need to
# know which prompt produced it, and "the one deployed at the time" is not an
# answer you can query.
PROMPT_VERSIONS: dict[str, str] = {
    "guardrail_in": "v3",
    "classifier": "v2",
    "orchestrator_plan": "v7",
    "order_agent": "v4",
    "shipping_agent": "v4",
    "policy_agent": "v6",
    "writer": "v5",
    "orchestrator_merge": "v7",
    "guardrail_out": "v3",
}

AGENT_IDS: list[str] = [
    "orchestrator",
    "order-agent",
    "shipping-agent",
    "policy-agent",
    "writer-agent",
]

DEFAULT_TENANT = os.getenv("DEFAULT_TENANT", "tenant-northwind")

REPLAY_ONLY = os.getenv("REPLAY_ONLY", "").lower() in {"1", "true", "yes"}

# Both names are in circulation for the same key. Accept either rather than
# making somebody debug why a key that is plainly present did nothing.
API_KEY = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()


def live_mode_available() -> bool:
    """Live mode needs a key. Replay mode needs nothing, and that is the point."""
    return bool(API_KEY) and not REPLAY_ONLY


def mode_reason() -> str:
    """Why the app is in the mode it is in.

    A reader who sees "live mode" and did not knowingly turn it on deserves to
    be told which environment variable caused it and how to switch back, rather
    than having to read the source to find out why their questions cost money.
    """
    key_name = "GOOGLE_API_KEY" if os.getenv("GOOGLE_API_KEY") else "GEMINI_API_KEY"
    if REPLAY_ONLY:
        return (
            "REPLAY_ONLY is set, so recorded traces are used even though a key "
            "is present."
        )
    if not API_KEY:
        return (
            "No GOOGLE_API_KEY or GEMINI_API_KEY was found, so everything runs "
            "from recorded traces. Nothing is called and nothing is billed."
        )
    return (
        f"{key_name} was found in your .env, so questions you run call the real "
        "model. Set REPLAY_ONLY=true in .env and restart to switch to recorded "
        "traces."
    )


@dataclass
class CacheSettings:
    """Prompt caching is a toggle in this project so the cost panel can show
    the same request costed both ways. In production it is not a toggle."""

    enabled: bool = True
    # Providers only cache a prefix once it is long enough to be worth it. The
    # real minimum is around 1024 to 2048 tokens depending on the model. The
    # prefixes in this project are shorter than that, so the threshold here is
    # lowered to keep the mechanism visible. Read the number below as "this
    # demo's threshold", not "the provider's".
    min_prefix_tokens: int = 200
    # Fraction of a matched stable prefix that actually reports as cached.
    hit_ratio: float = 0.92
    stable_prefixes: dict[str, str] = field(default_factory=dict)


CACHE = CacheSettings()

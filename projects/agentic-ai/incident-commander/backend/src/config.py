"""Central configuration, loaded from environment variables with safe defaults.

Every tunable that matters for the agent loop lives here so that the loop's
behavior (iteration ceiling, cost ceiling, approval policy, model tiering) is
explicit and auditable rather than scattered through the code.

Grouped by concern:
  * LLM          - model tiering, temperature, timeout, retries, circuit breaker
  * Loop         - iteration / cost / wall-clock ceilings, evidence rounds
  * Safety       - approval policy, agent role, redaction, injection handling
  * Memory       - episodic runbook store
  * Engine       - hand-rolled loop vs LangGraph StateGraph
  * Server       - host/port/CORS/auth/rate limiting/retention
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
#: The monorepo checkout (aiml-companion/), when this project lives inside one.
#: Standalone clones simply have no .env there and nothing is loaded.
REPO_DIR = PROJECT_DIR.parent.parent

# Load .env if present so users can copy .env.example to .env and just run.
# Three locations, most specific first - `load_dotenv` never overwrites a value
# already set, so the first file to define a key wins:
#   backend/.env    the local two-terminal path
#   <project>/.env  docker compose
#   <repo>/.env     a shared key for every project in the monorepo
# A learner who put the key in any of the "wrong" ones still gets a working
# system rather than a 503 they have no way to diagnose.
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env")
    load_dotenv(PROJECT_DIR / ".env")
    load_dotenv(REPO_DIR / ".env")
except ImportError:  # pragma: no cover
    pass


def _api_key() -> str:
    """The Gemini key, under either of the two names in circulation.

    Google's own tooling accepts both `GOOGLE_API_KEY` and `GEMINI_API_KEY`, and
    people copy whichever their last project used. Reading only one of them
    produced the least diagnosable failure this project has: a key that is plainly
    present in a .env file, and a 503 saying no key is configured.
    """
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in os.getenv(name, default).split(",") if v.strip())


# ---------------------------------------------------------------------------
# Model tiering
# ---------------------------------------------------------------------------
# Not every agent needs the same brain. Triage and Verification are cheap
# classification tasks; Diagnosis, remediation planning and critique are the hard
# reasoning steps. Routing per agent is a real production cost lever, so it is
# modelled explicitly here rather than hidden behind one global model name.
# "Judge" is the evaluation harness's LLM-as-judge. It sits in the fast tier so
# that, whenever tiering is on, it is a different model from the strong-tier agents
# whose reasoning it grades. (Before it was listed here it silently fell through to
# the DEFAULT model - the same one that wrote the diagnosis it was judging, while
# the docs claimed otherwise.)
FAST_TIER_AGENTS = frozenset({"Triage", "Verifier", "Postmortem", "Supervisor", "Judge"})
STRONG_TIER_AGENTS = frozenset({"Diagnostician", "Remediation", "Critic"})


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the Incident Commander."""

    # --- LLM ---------------------------------------------------------------
    google_api_key: str = field(default_factory=_api_key)
    #: Which brain drives the agents:
    #:   auto    - Gemini when a key is set, otherwise the offline agent (default)
    #:   gemini  - Gemini only; incidents are refused (503) without a key
    #:   offline - the deterministic rule-based agent in src/offline_llm.py
    llm_mode_setting: str = field(default_factory=lambda: os.getenv("IC_LLM_MODE", "auto").strip().lower())
    #: Seconds the offline agent pauses per call, so the console is watchable.
    offline_step_delay_s: float = field(default_factory=lambda: _get_float("IC_OFFLINE_STEP_DELAY_S", 0.8))
    #: Default / "strong" model: diagnosis, remediation planning, critique.
    #: Google's "-latest" aliases, so the default keeps working when a pinned
    #: version is retired or restricted to existing users. Pin a version here
    #: when you need repeatable measurements.
    model_name: str = field(default_factory=lambda: os.getenv("IC_MODEL_NAME", "gemini-flash-latest"))
    #: Cheaper model for the classification-shaped agents (see FAST_TIER_AGENTS).
    #: ON by default: measured against a live model it cut cost 45% and latency 51%
    #: on a single-iteration incident with no change in outcome. Set to "" to
    #: disable and run everything on the strong tier.
    model_fast: str = field(default_factory=lambda: os.getenv("IC_MODEL_FAST", "gemini-flash-lite-latest"))
    model_tiering: bool = field(default_factory=lambda: _get_bool("IC_MODEL_TIERING", True))
    temperature: float = field(default_factory=lambda: _get_float("IC_TEMPERATURE", 0.2))
    llm_timeout_s: float = field(default_factory=lambda: _get_float("IC_LLM_TIMEOUT_S", 60.0))
    llm_max_retries: int = field(default_factory=lambda: _get_int("IC_LLM_MAX_RETRIES", 2))
    #: Consecutive LLM failures that trip the circuit breaker for this process.
    llm_breaker_threshold: int = field(default_factory=lambda: _get_int("IC_LLM_BREAKER_THRESHOLD", 5))
    llm_breaker_cooldown_s: float = field(default_factory=lambda: _get_float("IC_LLM_BREAKER_COOLDOWN_S", 30.0))

    # --- Loop engineering knobs (the heart of the system) ------------------
    max_iterations: int = field(default_factory=lambda: _get_int("IC_MAX_ITERATIONS", 6))
    max_cost_usd: float = field(default_factory=lambda: _get_float("IC_MAX_COST_USD", 0.50))
    #: Wall-clock ceiling per incident. A third halt condition: an agent that is
    #: neither expensive nor iterating can still hang on a slow tool.
    max_wallclock_s: float = field(default_factory=lambda: _get_float("IC_MAX_WALLCLOCK_S", 900.0))
    #: How many extra evidence-gathering rounds the Diagnostician may request per
    #: iteration when it reports needs_more_data. 0 disables the branch.
    max_evidence_rounds: int = field(default_factory=lambda: _get_int("IC_MAX_EVIDENCE_ROUNDS", 2))
    #: Below this confidence the Diagnostician may not drive an action; the loop
    #: gathers more evidence instead, and escalates if it still cannot commit.
    min_action_confidence: float = field(default_factory=lambda: _get_float("IC_MIN_ACTION_CONFIDENCE", 0.6))

    # --- Safety ------------------------------------------------------------
    #: When True the human gate is bypassed for low/medium risk (demo convenience
    #: only). High risk ALWAYS requires a human. In production this stays False.
    auto_approve: bool = field(default_factory=lambda: _get_bool("IC_AUTO_APPROVE", False))
    #: RBAC role the agent acts as. See guardrails.manager.ROLE_ALLOWED.
    agent_role: str = field(default_factory=lambda: os.getenv("IC_AGENT_ROLE", "sre-bot"))
    #: Seconds to wait at the human approval gate before escalating. Without this
    #: an operator who closes the tab strands the incident forever.
    approval_timeout_s: float = field(default_factory=lambda: _get_float("IC_APPROVAL_TIMEOUT_S", 300.0))
    #: Quarantine (delimit + neutralize) telemetry text that trips the injection
    #: scan, instead of only flagging it and passing it through verbatim.
    quarantine_injections: bool = field(default_factory=lambda: _get_bool("IC_QUARANTINE_INJECTIONS", True))
    #: Redact secret-shaped and PII-shaped substrings before telemetry reaches the model.
    redact_sensitive: bool = field(default_factory=lambda: _get_bool("IC_REDACT_SENSITIVE", True))
    #: The code-side check on "recovered". The Verifier (a model) judges recovery,
    #: but code refuses to RESOLVE an incident whose metrics are still above these.
    recovery_max_error_rate: float = field(default_factory=lambda: _get_float("IC_RECOVERY_MAX_ERROR_RATE", 0.02))
    recovery_max_p99_ms: float = field(default_factory=lambda: _get_float("IC_RECOVERY_MAX_P99_MS", 1000.0))
    #: Run the Critic agent to challenge a proposed action before the human gate.
    critic_enabled: bool = field(default_factory=lambda: _get_bool("IC_CRITIC_ENABLED", True))
    #: A Critic veto sends the loop back to re-diagnose. Bounded, so a stubborn
    #: Critic and a stubborn Planner cannot argue until the budget is gone.
    max_critic_vetoes: int = field(default_factory=lambda: _get_int("IC_MAX_CRITIC_VETOES", 2))
    #: Use the LLM Supervisor to route between agents. OFF by default: it costs a
    #: model call per iteration, and `sanitize_route` clamps its answer to the same
    #: deterministic policy anyway, so on this fault catalog it buys nothing
    #: measurable while adding ~20s and a call to every incident. Turn it ON to see
    #: for yourself - that comparison is the point of keeping both.
    supervisor_enabled: bool = field(default_factory=lambda: _get_bool("IC_SUPERVISOR_ENABLED", False))

    # --- Memory ------------------------------------------------------------
    memory_enabled: bool = field(default_factory=lambda: _get_bool("IC_MEMORY_ENABLED", True))
    memory_path: Path = field(
        default_factory=lambda: Path(os.getenv("IC_MEMORY_PATH", str(BACKEND_DIR / "data" / "runbook_memory.json")))
    )
    memory_top_k: int = field(default_factory=lambda: _get_int("IC_MEMORY_TOP_K", 3))
    #: Cap on stored episodes; oldest are evicted first.
    memory_max_entries: int = field(default_factory=lambda: _get_int("IC_MEMORY_MAX_ENTRIES", 200))

    # --- Database + auth (both OPTIONAL) -----------------------------------
    #: Postgres URL. EMPTY BY DEFAULT, and that is a supported configuration: with
    #: no database the app keeps incidents in memory and runbook memory in a JSON
    #: file, exactly as it always has. Set it to persist across restarts and to
    #: enable login. `postgres://` and `postgresql://` are rewritten to the async
    #: driver automatically.
    database_url: str = field(default_factory=lambda: os.getenv("IC_DATABASE_URL", "").strip())
    db_pool_size: int = field(default_factory=lambda: _get_int("IC_DB_POOL_SIZE", 5))
    db_max_overflow: int = field(default_factory=lambda: _get_int("IC_DB_MAX_OVERFLOW", 10))
    #: Seed an initial admin on first boot so a fresh database is usable. Only ever
    #: applied when the users table is EMPTY, so it cannot resurrect a deleted
    #: account or overwrite a changed password.
    #: Empty password (the default) means "generate a random one and print it once
    #: in the backend log". A well-known default password is not a default anyone
    #: should be able to deploy by accident.
    seed_admin_username: str = field(default_factory=lambda: os.getenv("IC_SEED_ADMIN_USER", "admin"))
    seed_admin_password: str = field(default_factory=lambda: os.getenv("IC_SEED_ADMIN_PASSWORD", ""))

    #: Require a valid login for the API. Off by default AND forced off when no
    #: database is configured, because an auth wall with no user store is just a
    #: locked door with no key.
    auth_required: bool = field(default_factory=lambda: _get_bool("IC_AUTH_REQUIRED", False))
    #: Signing secret for session tokens. The default is a literal placeholder and
    #: is refused when auth is on - a shipped default secret is the same as none.
    jwt_secret: str = field(default_factory=lambda: os.getenv("IC_JWT_SECRET", "change-me-in-production"))
    jwt_ttl_minutes: int = field(default_factory=lambda: _get_int("IC_JWT_TTL_MINUTES", 720))

    # --- Engine ------------------------------------------------------------
    #: "loop" = the hand-rolled async orchestrator (the teaching reference).
    #: "graph" = the equivalent langgraph.StateGraph with a checkpointer.
    engine: str = field(default_factory=lambda: os.getenv("IC_ENGINE", "loop").strip().lower())

    # --- Server ------------------------------------------------------------
    #: Loopback by default: with login off, anyone who can reach this port can
    #: approve actions. The Docker image binds 0.0.0.0 inside its own network.
    host: str = field(default_factory=lambda: os.getenv("IC_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _get_int("IC_PORT", 8000))
    log_level: str = field(default_factory=lambda: os.getenv("IC_LOG_LEVEL", "INFO"))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: _get_csv("IC_CORS_ORIGINS", "http://localhost:5173"))
    #: Optional shared secret required in the X-API-Key header. Empty = open (local dev).
    api_key: str = field(default_factory=lambda: os.getenv("IC_API_KEY", ""))
    #: Per-client-IP request ceiling per minute on mutating endpoints. 0 disables.
    rate_limit_per_min: int = field(default_factory=lambda: _get_int("IC_RATE_LIMIT_PER_MIN", 60))
    #: Retention: how many incidents are kept in memory before the oldest terminal
    #: ones are evicted. Prevents unbounded growth in a long-running process.
    max_retained_incidents: int = field(default_factory=lambda: _get_int("IC_MAX_RETAINED_INCIDENTS", 50))

    # --- derived -----------------------------------------------------------

    @property
    def llm_enabled(self) -> bool:
        """True when a usable Gemini API key is configured."""
        return bool(self.google_api_key and self.google_api_key != "your_google_api_key_here")

    @property
    def llm_mode(self) -> str:
        """ "gemini" or "offline": which brain the agents actually run on.

        Never a silent switch: the mode is reported by /api/health, shown as a
        banner in the console, and stamped on every incident (ADR-0009).
        """
        if self.llm_mode_setting == "offline":
            return "offline"
        if self.llm_mode_setting == "gemini":
            return "gemini"
        return "gemini" if self.llm_enabled else "offline"

    @property
    def can_run_incidents(self) -> bool:
        """False only when Gemini mode was forced and no key is configured."""
        return self.llm_mode == "offline" or self.llm_enabled

    def model_for(self, agent: str) -> str:
        """Which model an agent should use (see FAST_TIER_AGENTS)."""
        if not self.model_tiering or not self.model_fast:
            return self.model_name
        return self.model_fast if agent in FAST_TIER_AGENTS else self.model_name

    @property
    def use_graph_engine(self) -> bool:
        return self.engine == "graph"

    @property
    def db_configured(self) -> bool:
        return bool(self.database_url)

    @property
    def auth_enabled(self) -> bool:
        """Auth needs somewhere to keep users, so it implies a database.

        Returning False rather than raising when `IC_AUTH_REQUIRED=true` is set
        without a database keeps the failure legible: the health endpoint reports
        auth as off, instead of every request 500ing on a missing table.
        """
        return self.auth_required and self.db_configured


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Test hook: force `get_settings()` to re-read the environment."""
    global _settings
    _settings = None

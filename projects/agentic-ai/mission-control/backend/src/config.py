"""Central configuration for Mission Control, loaded from environment variables.

This is a *control plane*, not an agent. The knobs here govern the platform the
agent fleets run on: the gateway's model routing and budgets, the runtime's
approval behaviour, the policy engine's fail-closed default and the eval gate's
promotion threshold. Keeping them in one place makes the platform's behaviour
explicit and auditable.

Execution mode (`MC_MODE`):

  - `auto` (default): live Gemini when a key is set, otherwise the offline provider.
  - `offline`: always the deterministic offline provider. No key, no network.
  - `live`: always Gemini. With no key the API refuses to start investigations
    (503) instead of quietly running offline, so a misconfigured deployment is loud.

The offline provider drives the identical gateway, policy, runtime, approval gate,
evals and event stream as Gemini. The seam is one class: `gateway/providers.py`.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    # backend/.env first, then the project root .env used by docker compose. Neither
    # overrides a variable that is already set in the environment.
    _BACKEND = Path(__file__).resolve().parent.parent
    load_dotenv(_BACKEND / ".env")
    load_dotenv(_BACKEND.parent / ".env")
except ImportError:  # pragma: no cover
    pass

MODES = ("auto", "offline", "live")


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
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


def _mode() -> str:
    raw = os.getenv("MC_MODE", "").strip().lower()
    # MC_FORCE_STUB predates MC_MODE; it is still honoured so old scripts keep working.
    if not raw and _get_bool("MC_FORCE_STUB", False):
        return "offline"
    return raw if raw in MODES else "auto"


def _gemini_installed() -> bool:
    """True when the langchain-google-genai provider is importable."""
    try:  # pragma: no cover - import probe
        import importlib.util

        return importlib.util.find_spec("langchain_google_genai") is not None
    except Exception:  # pragma: no cover
        return False


def _signing_key() -> bytes:
    """The key that signs agent workload tokens.

    A fixed key in source would let anyone who read the repository forge a token for
    any agent. With no MC_SIGNING_KEY set, each process generates its own random key:
    tokens never outlive the process anyway (300 s TTL), so nothing is lost.
    """
    configured = os.getenv("MC_SIGNING_KEY", "").strip()
    return configured.encode() if configured else secrets.token_bytes(32)


@dataclass(frozen=True)
class Operator:
    """A person who may act in the console. `roles` is a subset of {analyst, reviewer}."""

    name: str
    roles: frozenset[str]
    token: str


def _operators() -> tuple[Operator, ...]:
    """Parse MC_OPERATORS="name:role+role:token,...". Empty means login is off."""
    out: list[Operator] = []
    for entry in os.getenv("MC_OPERATORS", "").split(","):
        parts = [p.strip() for p in entry.split(":")]
        if len(parts) != 3 or not all(parts):
            continue
        name, roles, token = parts
        out.append(Operator(name=name, roles=frozenset(r for r in roles.split("+") if r), token=token))
    return tuple(out)


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the Mission Control platform."""

    # --- Gateway: model providers ------------------------------------------
    mode: str = field(default_factory=_mode)
    # Google's tooling hands out the same key under two names; accept either so live
    # mode does not silently fall back to offline over a variable name.
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", ""))
    # The `-latest` aliases are Google's stable pointers; pinned versions get retired
    # for new accounts (404) and silently break live mode.
    premium_model: str = field(default_factory=lambda: os.getenv("MC_PREMIUM_MODEL", "gemini-flash-latest"))
    cheap_model: str = field(default_factory=lambda: os.getenv("MC_CHEAP_MODEL", "gemini-flash-lite-latest"))
    temperature: float = field(default_factory=lambda: _get_float("MC_TEMPERATURE", 0.2))
    llm_timeout_s: float = field(default_factory=lambda: _get_float("MC_LLM_TIMEOUT_S", 60.0))

    # --- Gateway: semantic cache + budgets ---------------------------------
    cache_enabled: bool = field(default_factory=lambda: _get_bool("MC_CACHE_ENABLED", True))
    cache_similarity: float = field(default_factory=lambda: _get_float("MC_CACHE_SIMILARITY", 0.95))
    redact_pii: bool = field(default_factory=lambda: _get_bool("MC_REDACT_PII", True))

    # --- Runtime: ceilings + approvals ----------------------------------------
    # A workflow that exceeds a ceiling stops and goes to the dead-letter queue.
    max_workflow_steps: int = field(default_factory=lambda: _get_int("MC_MAX_STEPS", 40))
    max_cost_usd: float = field(default_factory=lambda: _get_float("MC_MAX_COST_USD", 0.25))
    max_wallclock_s: float = field(default_factory=lambda: _get_float("MC_MAX_WALLCLOCK_S", 600.0))
    # Auto-satisfies the human gate. Only the eval harness and demos set it.
    auto_approve: bool = field(default_factory=lambda: _get_bool("MC_AUTO_APPROVE", False))

    # --- People ---------------------------------------------------------------
    # With no operators configured, login is off: the console says who is acting in an
    # X-Operator header and nothing checks it. Maker-checker still holds either way.
    operators: tuple[Operator, ...] = field(default_factory=_operators)
    signing_key: bytes = field(default_factory=_signing_key, repr=False)

    # --- Policy engine ------------------------------------------------------
    # TEACHING FLAG (never set in production): disables the policy engine so learners
    # can watch what an ungoverned fleet would do. See docs/EXERCISES.md.
    disable_policy: bool = field(default_factory=lambda: _get_bool("MC_DISABLE_POLICY", False))

    # --- Eval plane ---------------------------------------------------------
    eval_promote_threshold: float = field(default_factory=lambda: _get_float("MC_EVAL_THRESHOLD", 0.80))

    # --- Observability ------------------------------------------------------
    # "" keeps spans in JSONL only; "console" or "otlp" also exports real OpenTelemetry
    # spans (needs the `otel` extra; "otlp" reads the standard OTEL_EXPORTER_OTLP_* vars).
    otel_exporter: str = field(default_factory=lambda: os.getenv("MC_OTEL_EXPORTER", "").strip().lower())

    # --- Persistence --------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MC_DATA_DIR", str(Path(__file__).resolve().parent.parent / ".work")))
    )

    # --- Server -------------------------------------------------------------
    host: str = field(default_factory=lambda: os.getenv("MC_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _get_int("MC_PORT", 8000))
    log_level: str = field(default_factory=lambda: os.getenv("MC_LOG_LEVEL", "INFO"))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in os.getenv("MC_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()
        )
    )

    @property
    def provider_available(self) -> bool:
        """True when Gemini is importable AND a key is set."""
        return bool(self.google_api_key) and _gemini_installed()

    @property
    def use_live_provider(self) -> bool:
        return self.mode != "offline" and self.provider_available

    @property
    def execution_mode(self) -> str:
        """What the health endpoint and the console banner show: `live` or `offline`."""
        return "live" if self.use_live_provider else "offline"

    @property
    def can_run(self) -> bool:
        """False only when live was demanded and cannot be honoured."""
        return not (self.mode == "live" and not self.provider_available)

    @property
    def login_required(self) -> bool:
        return bool(self.operators)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Test hook: drop the memoized settings so env overrides take effect."""
    global _settings
    _settings = None

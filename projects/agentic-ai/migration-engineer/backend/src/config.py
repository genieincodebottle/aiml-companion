"""Central configuration, loaded from environment variables with safe defaults.

Every knob that governs the agent - the model, the per-repo budget ceilings, the
approval policy, the concurrency of the fleet fan-out - lives here so the system's
behaviour is explicit and auditable rather than scattered through the code.

Unlike a raw-API agent, this project does NOT hand-roll the agent loop: the loop is
the Claude Agent SDK harness. Config here is about everything *around* that harness -
budgets, guardrails, orchestration - which is exactly where production effort belongs.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env if present so users can copy .env.example to .env and just run.
# Search order (first hit wins per variable): backend/.env, project .env, monorepo root
# .env. Guarded by depth: in Docker the app root is /app and the higher parents do not
# exist, so we only look as far up as the filesystem actually goes.
#
# The real checkout is <monorepo root>/projects/<category>/migration-engineer/backend,
# so "monorepo root" is FOUR hops up from backend_dir: migration-engineer (0) ->
# <category>, e.g. agentic-ai (1) -> projects (2) -> monorepo root (3). An earlier
# version of this list stopped at index 2 ("projects/"), one hop short of the actual
# monorepo root - so a key placed only in the true root .env (as the README and
# ADR 0008 both say is supported) was silently never picked up.
_ENV_SEARCH_PARENT_INDICES = (0, 3)


def _env_search_paths(backend_dir: Path) -> list[Path]:
    """The ordered list of .env files to load for a given backend directory."""
    parents = list(backend_dir.parents)
    bases = [backend_dir] + [parents[i] for i in _ENV_SEARCH_PARENT_INDICES if i < len(parents)]
    return [base / ".env" for base in bases]


try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    _backend_dir = Path(__file__).resolve().parent.parent
    for _env_path in _env_search_paths(_backend_dir):
        load_dotenv(_env_path)
except ImportError:  # pragma: no cover
    pass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Longest data dir we keep inside the checkout on Windows. Git's paths below the data
# dir add about 110 characters (a bare remote's incoming object path), and Windows
# rejects a path over 260 characters, or a working directory over about 248, unless
# the machine has long paths enabled in the registry. 100 leaves a safe margin.
_WINDOWS_DATA_DIR_MAX = 100


def _default_data_dir(backend_dir: Path, is_windows: bool = os.name == "nt") -> Path:
    """Where worktrees, bare remotes and traces go when ME_DATA_DIR is not set.

    Normally backend/.work. On Windows, a clone in a deep folder (Documents, OneDrive,
    a nested workspace) makes that path too long and git fails with "Filename too
    long" or "Invalid argument", so fall back to a short per-user folder instead.
    """
    local = backend_dir / ".work"
    if not is_windows or len(str(local)) <= _WINDOWS_DATA_DIR_MAX:
        return local
    base = os.getenv("LOCALAPPDATA") or str(Path.home())
    # One folder per checkout: the demo reuses its seeded bare remotes between runs, so
    # two clones sharing a folder would test against each other's fixtures.
    tag = hashlib.sha1(str(backend_dir).lower().encode("utf-8")).hexdigest()[:8]
    return Path(base) / f"migration-engineer-{tag}"


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


def _module_installed(module: str) -> bool:
    """True when the given optional package is importable."""
    try:  # pragma: no cover - import probe
        import importlib.util

        return importlib.util.find_spec(module) is not None
    except Exception:  # pragma: no cover
        return False


def _sdk_installed() -> bool:
    """True when the optional `claude-agent-sdk` package is importable."""
    return _module_installed("claude_agent_sdk")


def _genai_installed() -> bool:
    """True when the optional `google-genai` package is importable."""
    return _module_installed("google.genai")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for the Migration Engineer."""

    # --- Claude Agent SDK worker ------------------------------------------
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model_name: str = field(default_factory=lambda: os.getenv("ME_MODEL", "claude-sonnet-5"))

    # --- Gemini live worker (fallback when the Claude Agent SDK isn't configured) --
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    )
    gemini_model: str = field(default_factory=lambda: os.getenv("ME_GEMINI_MODEL", "gemini-2.5-flash"))
    # Force the offline deterministic worker even when a key + SDK are present
    # (used by CI and the test suite so runs are reproducible and free).
    force_stub: bool = field(default_factory=lambda: _get_bool("ME_FORCE_STUB", False))

    # --- TEACHING FLAG (never set in production) ----------------------------
    # Disables the PreToolUse guardrail hook so learners can watch the SECOND
    # defense layer (the reviewer's tamper gate) catch what the hook would have
    # blocked. See docs/EXERCISES.md, "Defense-in-depth lab".
    disable_guardrails: bool = field(default_factory=lambda: _get_bool("ME_DISABLE_GUARDRAILS", False))

    # --- Per-repo budget ceilings (the harness runs; WE own the ceilings) --
    max_agent_steps: int = field(default_factory=lambda: _get_int("ME_MAX_AGENT_STEPS", 12))
    max_tool_calls: int = field(default_factory=lambda: _get_int("ME_MAX_TOOL_CALLS", 60))
    max_cost_usd: float = field(default_factory=lambda: _get_float("ME_MAX_COST_USD", 0.75))

    # --- Version control (real git + GitHub) ------------------------------
    # A fine-grained PAT or GitHub App installation token with contents + pull_requests
    # write on the target repos. Absent -> only local-git (fixture/demo) targets run.
    github_token: str = field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN") or os.getenv("ME_GITHUB_TOKEN", "")
    )
    github_api_base: str = field(default_factory=lambda: os.getenv("ME_GITHUB_API", "https://api.github.com"))
    github_host: str = field(default_factory=lambda: os.getenv("ME_GITHUB_HOST", "github.com"))
    default_base_branch: str = field(default_factory=lambda: os.getenv("ME_BASE_BRANCH", "main"))

    # --- Fleet orchestration ----------------------------------------------
    # How many repos the LangGraph fan-out migrates concurrently.
    fan_out_concurrency: int = field(default_factory=lambda: _get_int("ME_FAN_OUT", 3))
    # When True the human PR-approval gate is bypassed (demo / eval convenience).
    # In production this stays False and every PR waits for a human.
    auto_approve: bool = field(default_factory=lambda: _get_bool("ME_AUTO_APPROVE", False))

    # --- Persistence -------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("ME_DATA_DIR") or _default_data_dir(Path(__file__).resolve().parent.parent)
        )
    )

    # --- Server ------------------------------------------------------------
    host: str = field(default_factory=lambda: os.getenv("ME_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _get_int("ME_PORT", 8000))
    log_level: str = field(default_factory=lambda: os.getenv("ME_LOG_LEVEL", "INFO"))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip()
            for o in os.getenv("ME_CORS_ORIGINS", "http://localhost:5173").split(",")
            if o.strip()  # filter empties after split
        )
    )

    @property
    def sdk_available(self) -> bool:
        """True when the real Claude Agent SDK is importable AND a key is set."""
        return bool(self.anthropic_api_key) and _sdk_installed()

    @property
    def gemini_available(self) -> bool:
        """True when `google-genai` is importable AND a Gemini/Google key is set."""
        return bool(self.gemini_api_key) and _genai_installed()

    @property
    def use_real_sdk(self) -> bool:
        """The effective worker choice: real Claude Agent SDK unless forced to stub."""
        return self.sdk_available and not self.force_stub

    @property
    def use_gemini(self) -> bool:
        """Gemini live worker: used when the Claude SDK path isn't configured."""
        return self.gemini_available and not self.use_real_sdk and not self.force_stub

    @property
    def use_live_worker(self) -> bool:
        """True when ANY live LLM worker (Claude SDK or Gemini) will run."""
        return self.use_real_sdk or self.use_gemini

    @property
    def execution_mode(self) -> str:
        """Human-readable mode for the health endpoint and logs."""
        if self.use_real_sdk:
            return "live-sdk"
        if self.use_gemini:
            return "live-gemini"
        return "stub"


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

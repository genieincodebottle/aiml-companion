"""Configuration: one loader, four role-scoped views.

Knobs come from ``configs/base.yaml``, secrets come from ``.env``, and nothing
else in the codebase reads either source directly. That is not tidiness for its
own sake - it is what makes `run.py`, the API, the tests and the notebook agree
about what the system is configured to do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Which environment variables carry the key for each provider, in priority
# order. `stub` is absent on purpose: needing no key is its entire reason to
# exist.
#
# Gemini accepts two names because both are in wide use - the Google Cloud
# tooling sets GOOGLE_API_KEY, the AI Studio docs say GEMINI_API_KEY, and a
# machine that has been used for both usually has one of each. Accepting only
# one produces "your key is not set" on a machine where it demonstrably is,
# which is a bad first five minutes.
_KEY_ENV: dict[str, tuple[str, ...]] = {
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai_compatible": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

_ROLES = ("generator", "judge", "reflector", "meta_judge")


@dataclass(frozen=True)
class RoleConfig:
    """Everything one role needs to make a call.

    Roles are separate values rather than one global model setting because the
    project's central claim depends on being able to point the judge and the
    generator at different models. A single `model:` key would make that
    impossible to express, and the self-preference question unanswerable.
    """

    role: str
    provider: str
    model: str
    temperature: float
    max_output_tokens: int
    #: Provider-specific reasoning budget. 0 disables thinking; None leaves the
    #: vendor default. Per-role rather than global because the right answer
    #: differs by role - see configs/base.yaml.
    thinking_budget: int | None
    api_key: str | None
    base_url: str | None
    extra: dict[str, Any]


class Config:
    def __init__(self, raw: dict[str, Any], root: Path | None = None) -> None:
        self._raw = raw
        self.root = root or ROOT

    # ------------------------------------------------------------------ roles
    def role(self, name: str) -> RoleConfig:
        if name not in _ROLES:
            raise KeyError(f"unknown role {name!r}; expected one of {_ROLES}")
        block = dict(self._raw.get(name) or {})
        provider = block.get("provider", "stub")
        per_provider = dict((self._raw.get("providers") or {}).get(provider) or {})

        env_names = _KEY_ENV.get(provider, ())
        api_key = next(
            (os.environ[n] for n in env_names if os.environ.get(n)), None
        )

        # Fail here, with the role named, rather than three layers down inside a
        # provider SDK. "role 'reflector' is configured for anthropic but
        # ANTHROPIC_API_KEY is unset" is a fixable message; a 401 from a vendor
        # client during iteration 4 of a tuning run is a puzzle.
        if env_names and not api_key:
            wanted = " or ".join(env_names)
            raise RuntimeError(
                f"role {name!r} is configured for provider {provider!r} but "
                f"{wanted} is not set.\n\n"
                "Two fixes, easiest first:\n"
                "  1. Add --offline to the command. Every role then runs on the "
                "deterministic rule engine: no key, no network, no cost. That "
                "is the recommended first run.\n"
                f"  2. Put {env_names[0]} in .env (see .env.example)."
            )

        return RoleConfig(
            role=name,
            provider=provider,
            model=block.get("model", ""),
            temperature=float(block.get("temperature", 0.0)),
            max_output_tokens=int(block.get("max_output_tokens", 1024)),
            thinking_budget=(
                None
                if block.get("thinking_budget") is None
                else int(block["thinking_budget"])
            ),
            api_key=api_key,
            base_url=per_provider.get("base_url"),
            extra=per_provider,
        )

    @property
    def roles(self) -> tuple[str, ...]:
        return _ROLES

    def providers_in_use(self) -> set[str]:
        return {(self._raw.get(r) or {}).get("provider", "stub") for r in _ROLES}

    @property
    def is_fully_offline(self) -> bool:
        """True when every role is the stub. Artefacts produced in this mode are
        stamped so their numbers can never be mistaken for model results."""
        return self.providers_in_use() == {"stub"}

    @property
    def single_model_config(self) -> bool:
        """True when the generator and judge are the same provider AND model.

        Not an error - it is the cheap default - but every report that comes out
        of such a run carries a caveat, because self-preference bias is
        uncontrolled and a judge grading its own house style will look better
        than it is.
        """
        g, j = self.role("generator"), self.role("judge")
        return (g.provider, g.model) == (j.provider, j.model)

    # ------------------------------------------------------------ plain blocks
    @property
    def domain_name(self) -> str:
        return self._raw.get("domain", "recommendation")

    @property
    def benchmark(self) -> dict[str, Any]:
        return dict(self._raw.get("benchmark") or {})

    @property
    def rart(self) -> dict[str, Any]:
        return dict(self._raw.get("rart") or {})

    @property
    def serving(self) -> dict[str, Any]:
        return dict(self._raw.get("serving") or {})

    @property
    def monitoring(self) -> dict[str, Any]:
        return dict(self._raw.get("monitoring") or {})

    @property
    def max_run_usd(self) -> float:
        return float(os.environ.get("MAX_RUN_USD", "5.00"))

    def as_dict(self) -> dict[str, Any]:
        return dict(self._raw)


def as_offline(config: Config) -> Config:
    """Every role switched to the stub provider, nothing else touched.

    Backs ``run.py --offline`` and the whole test suite. Overriding at this
    level rather than shipping a second YAML keeps ONE config file as the
    description of the system: an offline run and an online run differ in
    exactly one field per role, so anything else that differs between them is a
    real difference and not a drifted copy of the settings.
    """
    raw = dict(config.as_dict())
    for role in _ROLES:
        block = dict(raw.get(role) or {})
        block["provider"] = "stub"
        block["model"] = "stub"
        raw[role] = block
    return Config(raw, root=config.root)


def load_config(path: Path | None = None, root: Path | None = None) -> Config:
    root = root or ROOT
    path = path or root / "configs" / "base.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw, root=root)


@lru_cache(maxsize=1)
def get_config() -> Config:
    # dotenv is optional so the core package can be imported in a bare
    # environment; the .env file is a convenience, not a dependency.
    #
    # The project's own .env wins, then any .env further up the tree. The upward
    # search exists because this project lives inside a monorepo of projects
    # that share one key, and requiring a copy per project means N copies of a
    # secret to rotate and N chances to commit one.
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(ROOT / ".env")
        inherited = find_dotenv(usecwd=False)
        if inherited:
            # override=False: a key set in this project's own .env, or already
            # exported in the shell, is never silently replaced by one found
            # further up the tree.
            load_dotenv(inherited, override=False)
    except ImportError:  # pragma: no cover - convenience only
        pass
    return load_config()

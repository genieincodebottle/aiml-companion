"""Configuration loading.

Two sources, kept strictly apart:

  configs/base.yaml   tunables    - committed, safe to read, safe to share
  .env                credentials - never committed, never logged

`Config` is a thin, typed-ish accessor over both so the rest of the codebase
never touches ``os.environ`` or parses YAML again.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Substrings that mark a value as a copied-but-never-edited placeholder.
# Checking "is the variable set?" is not enough: `.env.example` ships
# `GOOGLE_API_KEY=your-google-api-key-here`, and that is a perfectly truthy
# string.  Without this check the failure surfaces 200 lines later as an
# opaque HTTP 400.
_PLACEHOLDERS = ("your-", "your_", "xxx", "changeme", "replace-me", "-here")


def _is_placeholder(value: str) -> bool:
    v = value.strip().strip('"').strip("'").lower()
    return len(v) < 8 or any(marker in v for marker in _PLACEHOLDERS)


class ConfigError(RuntimeError):
    """Raised with an actionable message when setup is incomplete."""


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str


class Config:
    def __init__(self, path: Path | str | None = None) -> None:
        self.root = PROJECT_ROOT
        cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "base.yaml"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            self._data: dict[str, Any] = yaml.safe_load(fh)

    # -- tunables -----------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        return self._data[name]

    @property
    def llm(self) -> dict[str, Any]:
        return self._data["llm"]

    @property
    def embedding(self) -> dict[str, Any]:
        return self._data["embedding"]

    @property
    def chunking(self) -> dict[str, Any]:
        return self._data["chunking"]

    @property
    def extraction(self) -> dict[str, Any]:
        return self._data["extraction"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self._data["retrieval"]

    # -- credentials --------------------------------------------------------
    @property
    def google_api_key(self) -> str:
        key = (os.getenv("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
        if not key or _is_placeholder(key):
            raise ConfigError(
                "GOOGLE_API_KEY is missing or still the placeholder from "
                ".env.example.\n"
                "  1. Get a free key: https://aistudio.google.com/app/apikey\n"
                "  2. copy .env.example .env   (Linux/macOS: cp)\n"
                "  3. Replace 'your-google-api-key-here' with the real value.\n"
                "Note: `python run.py test` runs the full unit suite with no key."
            )
        return key

    @property
    def neo4j(self) -> Neo4jSettings:
        password = (os.getenv("NEO4J_PASSWORD") or "").strip()
        if not password:
            raise ConfigError(
                "NEO4J_PASSWORD is not set.  Copy .env.example to .env - the "
                "defaults in it match the bundled docker-compose.yml, so "
                "`docker compose up -d` plus that copy is all you need."
            )
        return Neo4jSettings(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=password,
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        )

    # -- paths --------------------------------------------------------------
    @property
    def documents_dir(self) -> Path:
        return self.root / "data" / "documents"

    @property
    def structured_dir(self) -> Path:
        return self.root / "data" / "structured"

    @property
    def golden_questions(self) -> Path:
        return self.root / self._data["evaluation"]["dataset"]


_cached: Config | None = None


def get_config() -> Config:
    """Process-wide singleton.  Streamlit reruns the script on every widget
    interaction, so re-reading the YAML each time is pure waste."""
    global _cached
    if _cached is None:
        _cached = Config()
    return _cached

"""Settings for the engagement.

This file is GIVEN to you. It is deliberately boring: a customer environment is
not the place to invent a configuration system, and every hour spent on one is an
hour not spent on the thing they are paying for.

The one opinion here worth keeping: read and write credentials are separate
values. If you find yourself wanting a single token, re-read tests/test_mcp_auth.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional, the repo runs without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
CUSTOMER_DATA = ROOT / "customer" / "data"
POLICIES = CUSTOMER_DATA / "policies"
TICKETS = ROOT / "customer" / "tickets.jsonl"


@dataclass(frozen=True)
class Settings:
    """Runtime settings.

    Nothing here needs an API key. The scaffold, the customer environment and the
    whole of `make check` run offline, because "my key does not work on their
    network" is a normal Tuesday in this job.
    """

    legacy_api_url: str = field(
        default_factory=lambda: os.getenv("LEGACY_API_URL", "http://localhost:8000")
    )
    mcp_read_token: str = field(
        default_factory=lambda: os.getenv("MCP_READ_TOKEN", "dev-read-token")
    )
    mcp_write_token: str = field(
        default_factory=lambda: os.getenv("MCP_WRITE_TOKEN", "dev-write-token")
    )
    audit_log_path: Path = field(
        default_factory=lambda: Path(os.getenv("AUDIT_LOG_PATH", "./audit_log.jsonl"))
    )
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model_name: str = field(
        default_factory=lambda: os.getenv("FDE_MODEL_NAME", "gemini-3.6-flash")
    )

    @property
    def has_llm(self) -> bool:
        """True when a model call is possible.

        Guard every model call with this. A scaffold that crashes without a key
        teaches the learner nothing except that the scaffold is fragile.
        """
        return bool(self.google_api_key)


settings = Settings()

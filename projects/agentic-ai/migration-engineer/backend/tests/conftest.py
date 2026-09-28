"""Shared test setup: force the deterministic worker and an isolated data dir.

Tests never touch the network or the real Claude Agent SDK - they exercise the stub
worker through the exact same tool/hook/budget/orchestration contract, which is the
whole reason the stub exists.
"""

from __future__ import annotations

import os
import tempfile

# Set BEFORE any `src` import so the memoized Settings pick these up.
os.environ["ME_FORCE_STUB"] = "1"
os.environ.setdefault("ME_DATA_DIR", tempfile.mkdtemp(prefix="me-test-"))

from src.config import reset_settings_cache  # noqa: E402

reset_settings_cache()

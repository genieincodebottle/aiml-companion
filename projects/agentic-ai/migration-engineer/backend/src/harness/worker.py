"""Worker selection: the one place we choose the real Claude Agent SDK worker vs the
deterministic offline worker.

This is the seam the whole "inherit the harness" thesis rests on. The orchestrator
never knows which worker it got - both satisfy the same `run(ctx) -> MigrationResult`
contract - so the fan-out, guardrails, review, approval gate, and evals are identical
in either mode.
"""

from __future__ import annotations

from typing import Protocol

from ..config import Settings
from ..models.schemas import MigrationResult
from .base import WorkerContext
from .stub_agent import StubMigrator


class Migrator(Protocol):
    async def run(self, ctx: WorkerContext) -> MigrationResult: ...


def build_worker(settings: Settings) -> Migrator:
    """Return a live worker when configured (Claude SDK first, then Gemini), else the stub."""
    if settings.use_real_sdk:
        # Imported lazily so the offline path never needs the SDK installed.
        from .sdk_agent import SdkMigrator

        return SdkMigrator(model=settings.model_name)
    if settings.use_gemini:
        # Imported lazily so the offline path never needs google-genai installed.
        from .gemini_agent import GeminiMigrator

        return GeminiMigrator(model=settings.gemini_model, api_key=settings.gemini_api_key)
    return StubMigrator()

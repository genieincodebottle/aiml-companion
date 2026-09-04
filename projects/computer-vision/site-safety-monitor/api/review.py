"""Endpoint the safety officer uses to confirm or dismiss, capturing a
reason."""
from __future__ import annotations

from src.review.capture import VALID_REASONS, record

try:
    from fastapi import APIRouter

    router = APIRouter()
except ImportError:                       # the demo and the tests do not need it
    router = None


def decide(alert_id: str, confirmed: bool, reason: str | None = None):
    """reason is optional for a confirm and REQUIRED for a dismissal.

    A dismissal without a reason is an unlabelled negative. With one it
    tells you whether the detector was wrong, the zone was wrong, or the
    rule was wrong, and those need three different fixes.
    """
    if not confirmed and not reason:
        return {"error": "a dismissal needs a reason"}
    if reason is not None and reason not in VALID_REASONS:
        return {"error": f"reason must be one of {VALID_REASONS}"}
    record(alert_id, confirmed=confirmed, reason=reason, sampling="mined")
    return {"ok": True}


if router is not None:
    router.post("/alerts/{alert_id}/decision")(decide)

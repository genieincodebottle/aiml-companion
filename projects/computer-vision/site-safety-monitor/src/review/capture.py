"""Writes decisions as training data, tagged so they can never enter the
evaluation set."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

MINED = Path("data/reviewed_alerts.jsonl")

VALID_REASONS = ("wrong_detection", "outside_zone", "authorised")


def record(alert_id, confirmed, reason, sampling="mined", path: Path = None):
    """sampling='mined' keeps this out of the frozen evaluation set.

    These clips were selected because the pipeline was unsure, so they
    are harder than site traffic. Training on them is the point.
    Measuring on them makes the score drift away from reality.
    """
    p = MINED if path is None else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "alert_id": alert_id,
        "confirmed": bool(confirmed),
        "reason": reason,
        "sampling": sampling,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load(path: Path = None):
    p = MINED if path is None else Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line]


def evaluation_safe(rows):
    """The one filter that matters. Mined rows are harder by construction."""
    return [r for r in rows if r.get("sampling") != "mined"]

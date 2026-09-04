"""Turns dismissed alerts into training examples with an explicit negative
label."""
from __future__ import annotations

#: Which dismissal reasons produce training data, and which are a fix
#: somewhere else entirely.
REASON_ACTION = {
    "wrong_detection": "retrain",
    "outside_zone": "fix_polygon",
    "authorised": "fix_rule",
}


def to_training_example(alert, reason: str):
    """A dismissed alert is a hard negative, and the best kind.

    The detector was confident enough to fire, so this image sits right
    on the decision boundary. A random site photo teaches almost nothing
    by comparison.
    """
    if reason == "wrong_detection":
        return {"clip": alert.clip_path, "label": "background",
                "weight": 2.0}          # boundary cases count double
    if reason == "outside_zone":
        return None                     # a geometry fix, not a label
    if reason == "authorised":
        return None                     # a rule fix, not a label
    return {"clip": alert.clip_path, "label": "background", "weight": 1.0}


def triage(rows):
    """Groups dismissals by what has to change, which is the runbook order."""
    out = {v: 0 for v in REASON_ACTION.values()}
    out["unknown"] = 0
    for r in rows:
        if r.get("confirmed"):
            continue
        out[REASON_ACTION.get(r.get("reason"), "unknown")] += 1
    return out

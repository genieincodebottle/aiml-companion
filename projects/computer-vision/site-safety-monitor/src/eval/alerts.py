"""Precision and recall at the level of events rather than boxes."""
from __future__ import annotations

TOLERANCE_S = 10.0


def alert_metrics(fired, ground_truth_events, tolerance_s: float = TOLERANCE_S):
    """An alert is correct if a real event happened near it in time.

    Box-level mAP says nothing about this. A detector can miss half the
    frames of a 20 second violation and still produce exactly one
    correct alert, which is all the safety officer needed.

    An alert is matched against an event in the same zone whose window it
    falls in, allowing the dwell time plus a tolerance. Matching on the
    alert's own start rather than its fire time is what stops the dwell
    threshold counting as lateness.
    """
    matched, used = 0, set()
    for a in fired:
        for i, e in enumerate(ground_truth_events):
            if i in used or e.zone != a.zone:
                continue
            if e.start_ts - tolerance_s <= a.ts <= e.end_ts + tolerance_s:
                matched += 1
                used.add(i)
                break

    precision = matched / max(len(fired), 1)
    recall = matched / max(len(ground_truth_events), 1)
    return {
        "alerts": len(fired),
        "events": len(ground_truth_events),
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "false_alerts": len(fired) - matched,
    }

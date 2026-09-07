"""Sweeps the confidence threshold and reports false alerts per shift."""
from __future__ import annotations

from src.alerts import AlertSink
from src.clip_buffer import ClipBuffer, ClipStore
from src.config import settings
from src.eval.alerts import alert_metrics
from src.pipeline import run_shift

THRESHOLDS = (0.15, 0.25, 0.35, 0.45, 0.55)


def sweep(frames_by_camera, events, zones_cfg, store, detector_fn,
          thresholds=THRESHOLDS):
    """Reports the number a safety officer cares about, false alerts a shift.

    Recall on paper is not recall in practice. An alert stream where one
    in three is wrong gets ignored within a fortnight, and its effective
    recall is then zero whatever this table says.
    """
    rows = []
    for t in thresholds:
        buf = ClipBuffer(settings.fps, settings.clip_seconds)
        sink = AlertSink(buf, store)
        st = run_shift(frames_by_camera, detector_fn, zones_cfg, sink, conf=t)
        m = alert_metrics(st.fired, events)
        rows.append({
            "conf": t,
            "alerts": m["alerts"],
            "matched": m["matched"],
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "false_per_shift": m["false_alerts"],
            "raw_detections": st.raw_violation_detections,
        })
    return rows

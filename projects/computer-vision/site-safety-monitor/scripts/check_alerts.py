"""The CI gate, written in the units the customer feels.

A precision threshold means nothing to the person receiving the alerts.
Ten false alerts in a ten hour shift is one an hour, which is a rate you
can describe to a customer and hold yourself to.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alerts import AlertSink                       # noqa: E402
from src.clip_buffer import ClipBuffer, ClipStore      # noqa: E402
from src.config import settings                        # noqa: E402
from src.eval.alerts import alert_metrics              # noqa: E402
from src.pipeline import run_shift                     # noqa: E402
from src.sim.scene import ZONES, generate              # noqa: E402


def main(max_false_per_shift: int, min_event_recall: float, hours: float):
    frames, events = generate(hours=hours)
    sink = AlertSink(ClipBuffer(settings.fps, settings.clip_seconds),
                     ClipStore("artifacts/clips"))
    st = run_shift(frames, lambda f: f.boxes, ZONES, sink)
    m = alert_metrics(st.fired, events)

    # Scale the false count to a full shift, so the gate reads in the
    # same units whatever window CI runs.
    scaled = m["false_alerts"] * (settings.shift_hours / hours)

    print(f"{hours:.1f} h simulated: {m['alerts']} alerts, "
          f"{m['matched']} real, {m['false_alerts']} false")
    print(f"  event recall {m['recall']:.2f} (floor {min_event_recall})")
    print(f"  false alerts scaled to a shift: {scaled:.1f} "
          f"(ceiling {max_false_per_shift})")
    print(f"  reduction {st.reduction_ratio:.0f}:1 from "
          f"{st.raw_violation_detections:,} raw detections")

    problems = []
    if scaled > max_false_per_shift:
        problems.append(f"false alerts per shift {scaled:.1f} "
                        f"exceeds {max_false_per_shift}")
    if m["recall"] < min_event_recall:
        problems.append(f"event recall {m['recall']:.2f} "
                        f"below {min_event_recall}")

    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-false-per-shift", type=float, default=10)
    ap.add_argument("--min-event-recall", type=float, default=0.75)
    ap.add_argument("--hours", type=float, default=3.0)
    a = ap.parse_args()
    sys.exit(main(a.max_false_per_shift, a.min_event_recall, a.hours))

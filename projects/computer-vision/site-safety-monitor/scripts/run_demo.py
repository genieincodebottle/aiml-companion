"""Simulate a shift and write the artefacts the README quotes.

No weights, no camera, no network. The detections are synthetic; the
tracker, the zone test, the class vote, the dwell timer and the alert
deduplication are the real ones.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alerts import AlertSink                       # noqa: E402
from src.budget import modelled_report                 # noqa: E402
from src.clip_buffer import ClipBuffer, ClipStore      # noqa: E402
from src.config import settings                        # noqa: E402
from src.eval.alerts import alert_metrics              # noqa: E402
from src.eval.threshold_sweep import sweep             # noqa: E402
from src.pipeline import run_shift                     # noqa: E402
from src.sim.scene import ZONES, generate              # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"


def detector_fn(frame):
    """The detections are already in the frame. On the device this is
    Detector(engine), and the loop does not know the difference."""
    return frame.boxes


def main(sweep_only: bool = False):
    ART.mkdir(exist_ok=True)
    by_cam, events = generate()
    store = ClipStore(str(ART / "clips"))
    active = sum(len(v) for v in by_cam.values())

    print(f"shift: {settings.shift_hours:.0f} h, {settings.cameras} cameras "
          f"at {settings.fps} fps = {settings.frames_per_shift:,} frames")
    print(f"active frames simulated: {active:,} "
          f"({', '.join(f'{k} {len(v):,}' for k, v in sorted(by_cam.items()))})")
    print(f"ground truth violations: {len(events)}")

    b = modelled_report()
    print(f"\nframe budget {b['budget_ms']:.2f} ms")
    print(f"  a detected frame costs {b['detected_ms']:.2f} ms, "
          f"which does NOT fit")
    print(f"  detecting 1 frame in {b['detect_every']} averages "
          f"{b['average_ms']:.2f} ms, headroom {b['headroom_ms']:.2f} ms")
    print(f"  inference is {b['inference_share'] * 100:.1f}% of the frame, "
          f"so the ceiling on optimising it is {b['amdahl_ceiling']:.2f}x")

    if not sweep_only:
        buf = ClipBuffer(settings.fps, settings.clip_seconds)
        sink = AlertSink(buf, store)
        # The centre-rule comparison is the demo's whole point here, so it
        # opts in. On the device it stays off: it is a second polygon test
        # per track per frame inside a 33 ms budget.
        st = run_shift(by_cam, detector_fn, ZONES, sink,
                       compare_centre_rule=True)
        m = alert_metrics(st.fired, events)
        print(f"\nat conf {settings.conf_threshold}:")
        print(f"  raw violation detections {st.raw_violation_detections:,}")
        print(f"  alerts {m['alerts']}, of which {m['matched']} real "
              f"and {m['false_alerts']} false")
        print(f"  reduction {st.reduction_ratio:.0f}:1  |  "
              f"precision {m['precision']:.2f}  recall {m['recall']:.2f}")
        print(f"  suppressed: {st.suppressed_by_dwell:,} by dwell, "
              f"{st.suppressed_by_cooldown} by cooldown")
        print(f"  zone hits: {st.zone_hits_foot:,} by feet vs "
              f"{st.zone_hits_centre:,} by box centre "
              f"(+{st.zone_hits_centre - st.zone_hits_foot:,} false positions)")

        with (ART / "stage_budget.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["stage", "ms", "share_of_detected_frame"])
            for k, v in b["stages"].items():
                wr.writerow([k, f"{v:.2f}", f"{v / b['detected_ms']:.4f}"])

    rows = sweep(by_cam, events, ZONES, store, detector_fn)
    print("\nconf   alerts  matched  precision  recall  false/shift")
    for r in rows:
        print(f"{r['conf']:.2f}   {r['alerts']:>6}  {r['matched']:>7}  "
              f"{r['precision']:>9.2f}  {r['recall']:>6.2f}  "
              f"{r['false_per_shift']:>11}")

    with (ART / "threshold_sweep.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {ART / 'threshold_sweep.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-only", action="store_true")
    main(**vars(ap.parse_args()))

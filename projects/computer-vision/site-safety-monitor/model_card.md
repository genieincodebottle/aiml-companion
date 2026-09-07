# Model card: site safety monitor

## What it does

Detects PPE violations on two construction site cameras, keeps identity
between detections, tests whether a worker is inside a restricted zone,
and alerts when they hold that position for longer than a dwell
threshold.

## What the model is, and what it is not

The detector is a YOLO family model exported to TensorRT at FP16. This
repository contains the pipeline, the evaluation and the operating
procedure around that model. It does NOT contain trained weights, and
the numbers below come from a simulated shift rather than from site
footage.

That distinction matters. Every number in the frame budget is a costed
figure for the target device. Every number in the alert table is
measured, but measured on synthetic detections.

## Intended use

Advisory alerting to a safety officer, with a clip attached so each
alert can be judged. It is not an access control system, it does not
stop machinery, and it should never be the only control on a hazard.

## The frame budget

Two cameras at 15 fps is 30 frames a second, so the budget is 33.33 ms.

| Stage | ms | Share of a detected frame |
|---|---|---|
| capture | 4.0 | 11.9% |
| decode | 3.0 | 9.0% |
| letterbox | 2.0 | 6.0% |
| inference | 18.0 | 53.7% |
| nms | 4.0 | 11.9% |
| track | 1.5 | 4.5% |
| zone | 0.5 | 1.5% |
| emit | 0.5 | 1.5% |
| **total** | **33.50** | |

A detected frame is 33.50 ms against a 33.33 ms budget, so it does not
fit. Detecting on every second frame brings the average to 22.50 ms and
leaves 10.83 ms of headroom.

Inference is 53.7 per cent of the frame, so the best whole-frame speed-up
available from optimising the network alone is 2.16x. NMS is the stage to
watch afterwards: halving inference raises its share from 11.9 to 16.3
per cent without changing a line of it.

## Measured behaviour on a simulated shift

Ten hours, two cameras, 64 ground-truth violations.

| conf | alerts | real | false | precision | recall |
|---|---|---|---|---|---|
| 0.15 | 69 | 52 | 17 | 0.75 | 0.81 |
| 0.25 | 67 | 52 | 15 | 0.78 | 0.81 |
| 0.35 | 63 | 52 | 11 | 0.83 | 0.81 |
| **0.45** | **59** | **53** | **6** | **0.90** | **0.83** |
| 0.55 | 57 | 52 | 5 | 0.91 | 0.81 |

At the operating point the pipeline turns 6,091 raw violation detections
into 59 alerts, a reduction of about 103 to one. None of that reduction
comes from the detector. It comes from the class vote, the dwell timer
and the cooldown.

## Known limitations

- **The tracker holds rather than predicts.** There is no Kalman step, so
  a track sits at its last observed box through a skipped frame. That is
  adequate at 15 fps and walking pace and would not be at speed.
- **Small objects are the range limit.** A helmet on a worker 30 m away
  is a small object, and aggregate mAP hides how badly it does there.
  `src/eval/detection.py` stratifies by object area for that reason.
- **Cooldown can mask a second event.** Deduplication keys on zone and
  violation, so a genuinely different worker entering the same zone
  within five minutes is suppressed. That is the intended trade, and it
  is visible in the demo as ten suppressions across a shift.
- **The demo figures are not site figures.** They come from synthetic
  detections with a hand-chosen confidence distribution. They show the
  shape of the trade, not the performance of a real detector.

## Failure mode that matters most

Silence. A dropped stream, a stalled queue, or an engine loaded with the
wrong class order all produce zero alerts, and zero alerts looks exactly
like a safe site. `src/monitor/health.py` counts detections rather than
alerts for that reason, and `scripts/test_walk.py` exists because nothing
in software can detect a smudged lens.

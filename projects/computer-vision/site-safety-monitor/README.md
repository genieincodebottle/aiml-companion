# Site Safety Monitor

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/module/computerVision/cvYoloCapstone)**. Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.26-orange)
![Tests](https://img.shields.io/badge/tests-53%20passing-brightgreen)

**The frame budget decides the architecture. Everything else in this project
follows from a number you miss by 0.17 milliseconds.**

---

## 1. The problem

Two construction site cameras at 15 fps into one edge device. Detect PPE
violations, and tell a safety officer about the ones that matter.

Two cameras at 15 frames a second is 30 frames a second, so every frame has
**33.33 ms**. Time the first working build and the eight stages come to
**33.50 ms**.

```
capture 4.0 | decode 3.0 | letterbox 2.0 | inference 18.0
nms 4.0     | track 1.5  | zone 0.5      | emit 0.5      =  33.50 ms
```

It misses by 0.17 ms, which is the most annoying possible margin and also the
useful one, because it forces the right question rather than an easy one.

## 2. The ceiling on the obvious fix

Inference is 18.0 of 33.5, which is **53.7 percent**. Make the network
infinitely fast and the frame still costs 15.5 ms, so the best whole-frame
speed-up available is **2.16x**. That is Amdahl's law, and it caps the
optimisation everyone reaches for first.

The move that actually works is running the detector on every second frame and
letting the tracker carry the gap. That averages **22.50 ms** and leaves
**10.83 ms** of headroom.

```bash
python run.py budget      # prints the whole table, derived not typed
```

Every one of those numbers comes out of `src/budget.py` from the camera count
and frame rate. None of them is written down twice.

## 3. What the pipeline is actually for

A ten hour shift is 1,080,000 frames. In the simulated shift the detector
produces **6,091** frames on which a violation is visible inside a zone. The
safety officer receives **59 alerts**.

| Stage | What it removes |
|---|---|
| Class vote over 15 frames | One bad frame where a helmet is occluded |
| Bottom-centre zone test | Tall workers whose torso crosses a barrier they are standing outside |
| Dwell timer, 3 s | People walking past a line rather than crossing it |
| Cooldown by zone and violation | A group entering one zone, as one situation |

That is a reduction of about **103 to one**, and none of it comes from the
detector.

The zone test is the cheapest of these and removes the most. Testing the box
centre instead of the feet puts a tall worker inside a zone while they are
standing outside it: in the demo that is 25,164 zone hits instead of 12,400.

## 4. The confidence threshold is a real decision

```
conf   alerts  real  false  precision  recall
0.15      69     52     17       0.75    0.81
0.25      67     52     15       0.78    0.81
0.35      63     52     11       0.83    0.81
0.45      59     53      6       0.90    0.83
0.55      57     52      5       0.91    0.81
```

Recall barely moves. What moves is how many wrong alerts a person opens, and a
stream where one in four is wrong gets ignored within a fortnight. Its
effective recall is then zero, whatever the table says.

That is why `scripts/check_alerts.py` gates in **false alerts per shift** and
not in precision. Ten in a ten hour shift is one an hour, which is a rate you
can describe to a customer.

## 5. Run it

### Clone

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/computer-vision/site-safety-monitor
```

### Set up with uv

[uv](https://docs.astral.sh/uv/) resolves and installs in seconds and keeps the
environment inside the project.

```bash
pip install uv

uv venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell or cmd

uv pip install -r requirements.txt
```

Plain `pip install -r requirements.txt` works identically. Python 3.10+.

**No GPU, no camera, no weights, no network.** The install is numpy, pydantic
and pytest. OpenCV, ultralytics and TensorRT are in
`requirements-device.txt` and nothing in `src/` imports them at module level,
which is what lets the whole repository run anywhere.

### The argument, in order

```bash
python run.py budget     # the frame table and the Amdahl ceiling  (instant)
python run.py demo       # simulate a ten hour shift               (~40s)
python run.py sweep      # the confidence sweep only
python run.py gate       # the CI gate, in false alerts per shift  (~3s)
```

`run.py` needs **no install of this project**. It puts `src/` on the path
itself. There is a `Makefile` with the same targets if you prefer it, but
`make` is not on most Windows machines and `run.py` is.

Edit the polygons in `configs/base.yaml` and rerun the demo. The zones, the
cameras and the zone-to-camera mapping all come from that file, so a third zone
works without touching any code.

### Run the tests

```bash
python run.py test       # or: pytest
```

53 tests, about 5 seconds, including an end-to-end gate over a simulated three
hours.

## 6. Three things this repository learned the hard way

**A demo the tracker cannot track proves nothing.** The first scene generator
placed each actor at a random point in the zone on every frame. The pipeline
ran and printed plausible numbers while the tracker was associating different
people into a single track. A 22 second violation with 376 correct detections
produced no alert at all. See the docstring in `src/sim/scene.py`.

**Two confidence gates in one pipeline means the stricter one silently wins.**
The tracker held its own 0.50 while the config said 0.45, so only boxes above
0.50 could start a track and the sweep in section 4 came out flat from 0.15 to
0.45. The knob in the config file did nothing and nothing errored.
`src/tracker.py` now derives its gate from the setting, and a test pins it.

**A gate nobody has broken on purpose is not known to work.** Six deliberate
regressions were injected and the suite re-run for each. Five were caught. The
one that was not was the central design claim, that a skipped frame passes an
empty list to the tracker rather than skipping the tracker, so
`test_the_tracker_receives_every_frame_including_the_skipped_ones` was written
to cover it.

## 7. The notebook

`notebooks/site_safety_monitor_standalone.ipynb` is **standalone**. It needs
numpy and nothing else, and writes nothing to disk.

**This is the recommended starting point if you are new to the topic.**

The budget arithmetic, the point-in-polygon test and the dwell timer in it are
this repository's code verbatim. Section 4 of the notebook is a teaching
reduction with no tracker, no zone geometry and no cooldown, so it reports
about 200 to one where the full pipeline reports 103 to one. The difference is
exactly the work those three parts do.

**Locally**, after the setup above:

```bash
jupyter lab notebooks/site_safety_monitor_standalone.ipynb
```

**On Google Colab**, no install at all:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/projects/computer-vision/site-safety-monitor/notebooks/site_safety_monitor_standalone.ipynb)

**On Kaggle**, published as a notebook you can copy and edit:

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/genieincodebottle/site-safety-monitor-yolo)

`notebooks/_build_notebook.py` regenerates it.

## 8. If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `make: command not found` | no make on Windows | use `python run.py <cmd>` |
| `ModuleNotFoundError: src` | running a file in `src/` directly | use `python run.py`, it sets the path |
| `ModuleNotFoundError: cv2` | OpenCV is device-only | expected. Nothing in the demo needs it |
| `uv: command not found` after `pip install uv` | scripts dir not on PATH | `python -m uv venv`, or plain `pip` |
| editing `configs/base.yaml` changes nothing | PyYAML missing, so the defaults are used | `pip install PyYAML`, it is in requirements.txt |
| the confidence sweep looks flat | a second confidence gate somewhere above the tracker | see section 6, and `src/tracker.py` |
| `python run.py demo` takes ~40s | it simulates 77,000 active frames twice, once for the run and once for the sweep | expected, `run.py sweep` is the shorter half |

## 9. Layout

```
run.py                     zero-install entry point
configs/base.yaml          zones, cameras, and which camera sees which zone
src/
├── config.py              the budget, DERIVED from cameras x fps
├── site_config.py         reads configs/base.yaml
├── budget.py              stage timing, the modelled frame, the Amdahl ceiling
├── schemas.py             Box, Track, Alert. Coordinates stated once
├── capture.py             RTSP with a queue of 1 that drops the OLD frame
├── preprocess.py          letterbox and its inverse, applied in the detector
├── detector.py            engine in, boxes in ORIGINAL pixels out
├── postprocess.py         NMS, capped, quadratic in survivors
├── frame_skip.py          every Nth frame, unless nothing is being tracked
├── tracker.py             ByteTrack, including the low-confidence second pass
├── track_smoothing.py     majority class over a one second window
├── zones.py               point in polygon on the FEET, in numpy
├── dwell.py               hold the zone for 3 s, then alert once
├── clip_buffer.py         the seconds BEFORE the alert
├── alerts.py              assemble, attach the clip, deduplicate
├── eval/                  detection mAP by size, alert precision, the sweep
├── review/                decisions as training data, tagged mined
├── monitor/health.py      the failures that produce silence
├── pipeline.py            the loop, and run_shift across cameras
└── sim/                   the offline engine and the synthetic shift
scripts/export.py          ONNX to TensorRT, refusing a partitioned graph
scripts/check_alerts.py    the CI gate
scripts/test_walk.py       the only test that covers the lens
docs/RUNBOOK.md            what to do when the alerts stop
tests/                     53 tests
```

## 10. Track modules this covers

Each links straight to the module on AI-ML Companion.

| Module | Title |
|---|---|
| [`cvDetection`](https://aimlcompanion.ai/module/computerVision/cvDetection) | Object Detection |
| [`cvVideo`](https://aimlcompanion.ai/module/computerVision/cvVideo) | Video Understanding |
| [`cvEdge`](https://aimlcompanion.ai/module/computerVision/cvEdge) | Vision on the Edge |
| [`cvDeploy`](https://aimlcompanion.ai/module/computerVision/cvDeploy) | Vision in Production |
| [`cvEval`](https://aimlcompanion.ai/module/computerVision/cvEval) | Evaluating Vision Systems |
| [`cvIntro`](https://aimlcompanion.ai/module/computerVision/cvIntro) | What Computer Vision Actually Solves |

## 11. Honest limitations

**The detections are synthetic and there are no weights here.** The frame
budget is a costed figure for the target device, not a measurement on your
laptop. `src/budget.py` keeps the measured table and the modelled one apart for
exactly that reason, and the README quotes the modelled one.

**The tracker holds rather than predicts.** There is no Kalman step, so a track
sits at its last observed box through a skipped frame. That is adequate at 15
fps and walking pace and would not be at speed.

**Cooldown can mask a second event.** Deduplication keys on zone and violation,
so a genuinely different worker entering the same zone within five minutes is
suppressed. That is the intended trade and it costs about ten suppressions a
shift.

**The blog's figures and this repo's differ.** The walkthrough describes a site
producing 4,320 raw detections and 62 alerts, a 70 to one reduction. This
repository measures 6,091 and 59, about 103 to one, because its detections are
synthetic with a hand-chosen confidence distribution. The shape of the argument
is identical and the numbers are not.

**The TensorRT path is real code that is not exercised end to end.**
`scripts/export.py` needs `trtexec`. Its partition check is tested against a
recorded log, which covers the parsing and not the export.

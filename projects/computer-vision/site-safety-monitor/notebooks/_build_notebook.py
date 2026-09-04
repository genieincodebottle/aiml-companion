# -*- coding: utf-8 -*-
"""Build the standalone notebook from a list of (markdown, code) cells.

The notebook is generated rather than hand-edited so it cannot drift from
the repository, and so the JSON stays reviewable in a diff.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "site_safety_monitor_standalone.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md(r"""
# Site Safety Monitor

PPE violation alerting for two site cameras on one edge device.

This notebook is self contained. It needs numpy and nothing else: no GPU,
no camera, no weights, no network.

The budget arithmetic, the point-in-polygon test and the dwell timer
below are the repository's code verbatim. The shift simulation in section
4 is a teaching reduction of it: no tracker, no zone geometry, no
cooldown. So its reduction ratio comes out around 200 to one where the
full pipeline reports 103 to one, and the difference is exactly the work
those three parts do. Run `python run.py demo` in the repository for the real
figures.

The point of the project is not the detector. It is everything that turns
6,000 raw detections into 59 alerts a person will actually open.
"""),
    md(r"""
## 1. The budget

Two cameras at 15 fps is 30 frames a second, so each frame has 33.33 ms.
Here is what a frame that runs the detector costs on the target device.
"""),
    code(r"""
import numpy as np

CAMERAS, FPS = 2, 15
BUDGET_MS = 1000.0 / (CAMERAS * FPS)

STAGES = {
    "capture": 4.0, "decode": 3.0, "letterbox": 2.0, "inference": 18.0,
    "nms": 4.0, "track": 1.5, "zone": 0.5, "emit": 0.5,
}
SKIPPABLE = ("inference", "nms")   # letterbox is NOT skipped, see below

detected = sum(STAGES.values())
skipped = sum(v for k, v in STAGES.items() if k not in SKIPPABLE)

print(f"budget              {BUDGET_MS:.2f} ms")
print(f"a detected frame    {detected:.2f} ms   <- does not fit")
print(f"a skipped frame     {skipped:.2f} ms")
print(f"average, 1 in 2     {(detected + skipped) / 2:.2f} ms")
print(f"headroom            {BUDGET_MS - (detected + skipped) / 2:.2f} ms")
"""),
    md(r"""
It misses by 0.17 ms. That is the whole problem.

Inference is the obvious thing to optimise, and this is the number that
says how much optimising it can ever be worth.
"""),
    code(r"""
share = STAGES["inference"] / detected
print(f"inference is {share * 100:.1f}% of the frame")
print(f"Amdahl ceiling on the whole frame: {1 / (1 - share):.2f}x")

halved = dict(STAGES, inference=STAGES["inference"] / 2)
print(f"\nNMS share now:            {STAGES['nms'] / detected * 100:.1f}%")
print(f"NMS share if inference halves: "
      f"{halved['nms'] / sum(halved.values()) * 100:.1f}%")
print("NMS runs on the CPU and does not quantise, so it grows as a share")
print("every time the network gets faster.")
"""),
    md(r"""
## 2. Geometry, and the one line that removes most false alerts

A camera looking across a site sees a standing person as a tall box. Test
the centre of that box and a worker standing outside a barrier registers
as inside it, because the centre is roughly waist height and projects
past where their feet are.

Test the bottom centre instead.
"""),
    code(r"""
def point_in_polygon(px, py, poly):
    x, y = poly[:, 0], poly[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    straddles = (y > py) != (ys > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x + (py - y) * (xs - x) / np.where(ys - y == 0, np.nan, ys - y)
    return bool((straddles & (px < x_cross)).sum() % 2 == 1)


ZONE = np.array([(60.0, 260.0), (520.0, 260.0), (560.0, 470.0), (20.0, 470.0)])

# A tall worker whose feet are at y=480, ten pixels below the zone edge.
x1, y1, x2, y2 = 270.0, 320.0, 330.0, 480.0
feet = ((x1 + x2) / 2, y2)
centre = ((x1 + x2) / 2, (y1 + y2) / 2)

print(f"feet   {feet}  inside: {point_in_polygon(*feet, ZONE)}")
print(f"centre {centre}  inside: {point_in_polygon(*centre, ZONE)}   <- the bug")
"""),
    md(r"""
## 3. Dwell, and why it is the product

A violation is a person, in a place, for a time. Without the third part
every alert is about a frame, and the officer receives thousands of them.
"""),
    code(r"""
class DwellTimer:
    def __init__(self, dwell_seconds=3.0):
        self.entered, self.alerted, self.dwell = {}, set(), dwell_seconds

    def should_alert(self, track_id, zone, ts):
        key = (track_id, zone)
        if key in self.alerted:
            return False                    # one alert per track per zone
        first = self.entered.setdefault(key, ts)
        if ts - first >= self.dwell:
            self.alerted.add(key)
            return True
        return False


d = DwellTimer()
fires = [d.should_alert(1, "crane", t / 15) for t in range(0, 150)]
print(f"frames in the zone: {len(fires)}")
print(f"alerts fired:       {sum(fires)}")
print(f"first fired at frame {fires.index(True)} = "
      f"{fires.index(True) / 15:.1f} s")
"""),
    md(r"""
One hundred and fifty frames of a genuine violation produce exactly one
alert. The `alerted` set is what makes that true. Without it the timer
fires on every frame after the threshold, which is the same flood in a
different costume.

## 4. The whole shift

Now the parts together, on a simulated ten hour shift. Actors walk, so
the tracker has something to track. A tenth of the compliant workers are
systematically mislabelled at a confidence near the threshold, which is
what makes the confidence choice a real decision.
"""),
    code(r"""
from collections import Counter, deque

rng = np.random.default_rng(17)
SHIFT_S = 10 * 3600
N_VIOL, N_COMPLIANT, MISLABEL_SHARE = 64, 240, 0.10


def make_actors():
    actors, events = [], []
    for i in range(N_VIOL):
        start = rng.uniform(0, SHIFT_S - 60)
        dur = rng.uniform(6, 26)
        actors.append(dict(start=start, end=start + dur, violation=True,
                           conf=float(np.clip(rng.normal(0.70, 0.10), 0.02, 0.99))))
        events.append((start, start + dur))
    for _ in range(N_COMPLIANT):
        start = rng.uniform(0, SHIFT_S - 60)
        dur = rng.uniform(5, 30)
        mis = rng.random() < MISLABEL_SHARE
        base = rng.normal(0.40, 0.13) if mis else rng.normal(0.80, 0.06)
        actors.append(dict(start=start, end=start + dur, violation=False,
                           mislabelled=mis,
                           conf=float(np.clip(base, 0.02, 0.99))))
    return actors, events


# Class vote, dwell, and cooldown over each actor's frames.
def run(conf_threshold, actors, events):
    raw, alerts = 0, []
    for a in actors:
        reads_as_violation = a["violation"] or a.get("mislabelled")
        if not reads_as_violation:
            continue
        if a["conf"] < conf_threshold:
            continue                      # filtered before the tracker
        n = int((a["end"] - a["start"]) * FPS)
        seen = deque(maxlen=15)
        dwell = DwellTimer()
        for k in range(n):
            if rng.random() > 0.90:
                continue                  # the detector missed this frame
            ts = a["start"] + k / FPS
            seen.append("violation")
            if Counter(seen).most_common(1)[0][0] != "violation":
                continue
            raw += 1
            if dwell.should_alert(id(a), "zone", ts):
                alerts.append((ts, a["violation"]))
    return raw, alerts


actors, events = make_actors()
raw, alerts = run(0.45, actors, events)
real = sum(1 for _, v in alerts if v)
print(f"raw violation detections: {raw:,}")
print(f"alerts:                   {len(alerts)}")
print(f"  real {real}, false {len(alerts) - real}")
print(f"reduction:                {raw / max(len(alerts), 1):.0f}:1")
"""),
    md(r"""
## 5. The threshold sweep

The number a safety officer feels is not precision. It is how many wrong
alerts they open in a shift.
"""),
    code(r"""
# One population, swept. Regenerating the actors inside the loop would
# resample the confidences and the table would move for that reason
# rather than for the threshold.
population = make_actors()
print("conf   alerts  real  false  precision")
for t in (0.15, 0.25, 0.35, 0.45, 0.55):
    _raw, al = run(t, *population)
    real = sum(1 for _, v in al if v)
    p = real / max(len(al), 1)
    print(f"{t:.2f}   {len(al):>6}  {real:>4}  {len(al) - real:>5}  {p:>9.2f}")
"""),
    md(r"""
Recall barely moves across that range. What moves is the share of alerts
that are wrong, and that determines whether anyone still opens them in a
fortnight. An ignored alert stream has an effective recall of zero
however good the model is.

## 6. What this leaves out

The full repository has the parts a notebook cannot show usefully: the
TensorRT export that refuses a partitioned graph, the parity test that
asserts boxes and classes at different tolerances, the review loop that
turns a dismissal into a diagnosis, and the health check built around the
fact that a failed safety system and a safe site both produce silence.

https://github.com/genieincodebottle/aiml-companion/tree/main/projects/computer-vision/site-safety-monitor
"""),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    OUT.write_text(json.dumps(NB, indent=1), encoding="utf-8")
    print(f"wrote {OUT} with {len(CELLS)} cells")

"""A synthetic shift, generated offline.

The shift is generated at the level of DETECTIONS rather than pixels. The
thing being demonstrated is what tracking, zones and dwell do to a stream
of noisy boxes, and rendering images would add hours of compute without
changing a single number downstream.

## Actors move, and that is not a detail

The first version of this file placed each actor at a random point in
their zone on every frame. The pipeline ran, produced plausible numbers,
and was meaningless: with no spatial continuity the tracker associated
different people into one track, the class vote averaged a violator with
a compliant worker, and a 22 second violation with 376 correct detections
produced no alert at all.

A detection pipeline cannot be exercised by a scene its tracker cannot
track. Actors here walk a slow line, hold their box size, and are given
separated lanes within a zone, so association by IoU means what it means
on a real camera.

## Two populations produce the two error modes

- Real violators. A worker without a helmet in a zone for long enough to
  matter. The detector sees them on most frames and misses some.
- Compliant workers near a zone. A fraction of them are SYSTEMATICALLY
  mislabelled, the way a dark helmet against a dark hoarding is, at a
  confidence around the threshold. That is what makes the confidence
  sweep a real decision rather than a free parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import settings
from src.schemas import Box, Event, PPEClass
from src.site_config import load_zones, zone_camera

#: Both come from configs/base.yaml. Edit the polygons there and the demo,
#: the sweep and the tests all move with them.
ZONES = load_zones()
ZONE_CAMERA = zone_camera()
CAMERAS = tuple(sorted(set(ZONE_CAMERA.values())))

# --- generator constants, tuned so the shift reproduces the site's numbers ---
N_VIOLATIONS = 64           # real events across the shift
N_COMPLIANT = 240           # workers who pass through or near a zone
MISLABELLED_SHARE = 0.10    # of the compliant ones, seen wrong throughout
VIOL_SECONDS = (6.0, 26.0)
COMPLIANT_SECONDS = (5.0, 30.0)
DETECT_RATE = 0.90          # per-frame recall on a person who is there

# Base confidence is a property of the SUBJECT, not of the frame. These
# are the population distributions the confidence sweep separates.
TRUE_CONF = (0.70, 0.10)    # a genuine violation, clearly seen
FALSE_CONF = (0.40, 0.13)   # a systematic mislabel, straddling the sweep
COMPLIANT_CONF = (0.80, 0.06)
CONF_JITTER = 0.04          # frame to frame wobble around the base
SPEED_PX = (0.4, 1.6)       # pixels per frame, a walking pace at this scale
SEED = 17


@dataclass
class Frame:
    idx: int
    ts: float
    camera_id: str
    boxes: list = field(default_factory=list)


@dataclass
class Actor:
    start_ts: float
    end_ts: float
    zone: str
    camera_id: str
    true_cls: PPEClass
    is_violator: bool
    mislabelled: bool
    x0: float
    y0: float
    vx: float
    vy: float
    height: float
    conf_base: float

    def foot_at(self, t: float) -> tuple[float, float]:
        k = (t - self.start_ts) * settings.fps
        return self.x0 + self.vx * k, self.y0 + self.vy * k

    def conf_at(self, rng) -> float:
        return self.conf_base + rng.normal(0, CONF_JITTER)

    def box_at(self, t: float, cls: PPEClass, conf: float, rng) -> Box:
        fx, fy = self.foot_at(t)
        # Box jitter is small relative to the box, which is what keeps
        # frame-to-frame IoU high enough to associate.
        fx += rng.normal(0, 1.5)
        fy += rng.normal(0, 1.5)
        wd = self.height * 0.38
        return Box(x1=fx - wd / 2, y1=fy - self.height, x2=fx + wd / 2, y2=fy,
                   cls=cls, conf=float(np.clip(conf, 0.01, 0.99)))


def _lane(rng, poly, lane_i: int, lanes: int):
    """Separated start points inside a zone, so two actors are not one blob."""
    pts = np.asarray(poly)
    x_lo, x_hi = pts[:, 0].min() + 30, pts[:, 0].max() - 30
    step = (x_hi - x_lo) / max(lanes, 1)
    x = x_lo + step * (lane_i % lanes) + rng.uniform(0, step * 0.5)
    y = rng.uniform(pts[:, 1].min() + 30, pts[:, 1].max() - 15)
    return x, y


def _outside_start(rng, poly):
    """Feet below the zone's near edge, torso projecting across it.

    This is the case the bottom-centre test gets right and the box-centre
    test gets wrong, so the scene has to contain it.
    """
    pts = np.asarray(poly)
    x = rng.uniform(pts[:, 0].min() + 30, pts[:, 0].max() - 30)
    y = pts[:, 1].max() + rng.uniform(8, 30)
    return x, y


def generate(hours: float | None = None, seed: int = SEED):
    """Returns (frames_by_camera, ground_truth_events)."""
    rng = np.random.default_rng(seed)
    hours = settings.shift_hours if hours is None else hours
    total_s = hours * 3600.0
    scale = hours / settings.shift_hours
    zone_names = list(ZONES)

    n_viol = max(int(round(N_VIOLATIONS * scale)), 1)
    n_comp = max(int(round(N_COMPLIANT * scale)), 1)

    actors: list[Actor] = []
    events: list[Event] = []

    for i in range(n_viol):
        zone = zone_names[i % len(zone_names)]
        start = rng.uniform(0, total_s - 60)
        dur = rng.uniform(*VIOL_SECONDS)
        cls = PPEClass.NO_HELMET if rng.random() < 0.70 else PPEClass.NO_VEST
        x, y = _lane(rng, ZONES[zone], i, 4)
        ang = rng.uniform(0, 2 * np.pi)
        sp = rng.uniform(*SPEED_PX)
        actors.append(Actor(start, start + dur, zone, ZONE_CAMERA[zone], cls,
                            True, False, x, y, sp * np.cos(ang),
                            sp * np.sin(ang) * 0.3,
                            rng.uniform(95, 150),
                            float(np.clip(rng.normal(*TRUE_CONF), 0.02, 0.99))))
        events.append(Event(zone=zone, violation=cls, start_ts=start,
                            end_ts=start + dur))

    for i in range(n_comp):
        zone = zone_names[i % len(zone_names)]
        start = rng.uniform(0, total_s - 60)
        dur = rng.uniform(*COMPLIANT_SECONDS)
        mis = rng.random() < MISLABELLED_SHARE
        # The mislabelled ones stand inside; the rest walk the boundary.
        x, y = (_lane(rng, ZONES[zone], i + 2, 4) if mis
                else _outside_start(rng, ZONES[zone]))
        ang = rng.uniform(0, 2 * np.pi)
        sp = rng.uniform(*SPEED_PX)
        base = rng.normal(*(FALSE_CONF if mis else COMPLIANT_CONF))
        actors.append(Actor(start, start + dur, zone, ZONE_CAMERA[zone],
                            PPEClass.HELMET, False, mis, x, y,
                            sp * np.cos(ang), sp * np.sin(ang) * 0.3,
                            rng.uniform(110, 165),
                            float(np.clip(base, 0.02, 0.99))))

    events.sort(key=lambda e: e.start_ts)
    return _render(actors, rng), events


def _render(actors, rng):
    """Emit only the frames where something is happening.

    A ten hour shift is 1,080,000 frames and almost all of them are an
    empty site. The pipeline is identical on an empty frame, so the demo
    walks the active windows and counts the rest.
    """
    fps = settings.fps
    per_cam: dict[str, dict[int, Frame]] = {c: {} for c in CAMERAS}

    for a in actors:
        n = int((a.end_ts - a.start_ts) * fps)
        for k in range(n):
            ts = a.start_ts + k / fps
            tick = int(round(ts * fps))
            frames = per_cam[a.camera_id]
            f = frames.get(tick)
            if f is None:
                f = Frame(idx=0, ts=tick / fps, camera_id=a.camera_id)
                frames[tick] = f

            if rng.random() >= DETECT_RATE:
                continue                     # the detector missed this frame

            conf = a.conf_at(rng)
            if a.is_violator:
                f.boxes.append(a.box_at(ts, a.true_cls, conf, rng))
            elif a.mislabelled:
                wrong = (PPEClass.NO_HELMET if a.true_cls is PPEClass.HELMET
                         else PPEClass.NO_VEST)
                f.boxes.append(a.box_at(ts, wrong, conf, rng))
            else:
                f.boxes.append(a.box_at(ts, a.true_cls, conf, rng))

    out = {}
    for cam, frames in per_cam.items():
        ordered = [frames[k] for k in sorted(frames)]
        for i, f in enumerate(ordered):
            f.idx = i
        out[cam] = ordered
    return out

"""Point-in-polygon against configured restricted areas, using the feet rather
than the box centre.

Implemented with numpy ray casting rather than shapely. The geometry is a
handful of fixed polygons tested once per track per frame, so the
dependency buys nothing and would stop the pipeline running offline.
"""
from __future__ import annotations

import numpy as np


def point_in_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    """Crossing-number test. Odd number of crossings means inside.

    The half-open y comparison is what stops a vertex being counted
    twice, which would report a point on a boundary as outside.
    """
    x = poly[:, 0]
    y = poly[:, 1]
    xs = np.roll(x, -1)
    ys = np.roll(y, -1)

    straddles = (y > py) != (ys > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x + (py - y) * (xs - x) / np.where(ys - y == 0, np.nan, ys - y)
    crossings = straddles & (px < x_cross)
    return bool(crossings.sum() % 2 == 1)


class ZoneSet:
    def __init__(self, zones: dict[str, list[tuple[float, float]]]):
        self.polys = {name: np.asarray(pts, dtype=np.float64)
                      for name, pts in zones.items()}

    def zone_of(self, box) -> str | None:
        """Test the BOTTOM CENTRE of the box, where the person stands.

        Using the centre of the box puts a tall worker inside a zone
        while their feet are still outside it, which generates alerts
        for people walking past a barrier rather than crossing it.
        """
        fx, fy = box.foot
        for name, poly in self.polys.items():
            if point_in_polygon(fx, fy, poly):
                return name
        return None

    def zone_of_centre(self, box) -> str | None:
        """The wrong version, kept so the test can measure the difference."""
        cx = (box.x1 + box.x2) / 2.0
        cy = (box.y1 + box.y2) / 2.0
        for name, poly in self.polys.items():
            if point_in_polygon(cx, cy, poly):
                return name
        return None

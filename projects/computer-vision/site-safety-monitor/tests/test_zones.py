"""The bottom-centre rule, which is a one-line change that removes a large
share of false alerts."""
from __future__ import annotations

import numpy as np

from src.zones import ZoneSet, point_in_polygon
from tests.conftest import box

SQUARE = np.array([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)])


def test_point_in_polygon_basics():
    assert point_in_polygon(50, 50, SQUARE)
    assert not point_in_polygon(150, 50, SQUARE)
    assert not point_in_polygon(50, 150, SQUARE)
    assert not point_in_polygon(-1, 50, SQUARE)


def test_a_vertex_is_not_counted_twice():
    """The half-open y comparison. Without it a ray through a vertex counts
    two crossings and reports an inside point as outside."""
    diamond = np.array([(50.0, 0.0), (100.0, 50.0), (50.0, 100.0), (0.0, 50.0)])
    assert point_in_polygon(50, 50, diamond)      # ray passes through vertices


def test_the_feet_are_outside_while_the_centre_is_inside(zones):
    """The exact case that generates alerts for people walking past a barrier.

    A tall worker standing just below the crane zone has feet at y=480,
    outside the zone that ends at y=470. Their box centre is at waist
    height, well inside it.
    """
    zs = ZoneSet(zones)
    b = box(x_centre=300.0, y_feet=480.0, height=160.0)
    assert b.foot == (300.0, 480.0)
    assert zs.zone_of(b) is None                 # correct
    assert zs.zone_of_centre(b) == "crane_radius"  # the bug


def test_a_worker_actually_inside_is_found_by_both(zones):
    zs = ZoneSet(zones)
    b = box(x_centre=300.0, y_feet=430.0, height=120.0)
    assert zs.zone_of(b) == "crane_radius"
    assert zs.zone_of_centre(b) == "crane_radius"


def test_zones_are_distinguished(zones):
    zs = ZoneSet(zones)
    assert zs.zone_of(box(900.0, 400.0)) == "excavation"
    assert zs.zone_of(box(300.0, 400.0)) == "crane_radius"

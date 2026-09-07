from __future__ import annotations

import pytest

from src.schemas import Box, PPEClass
from src.sim.scene import ZONES


@pytest.fixture
def zones():
    return ZONES


def box(x_centre, y_feet, height=120.0, cls=PPEClass.NO_HELMET, conf=0.9):
    w = height * 0.38
    return Box(x1=x_centre - w / 2, y1=y_feet - height,
               x2=x_centre + w / 2, y2=y_feet, cls=cls, conf=conf)

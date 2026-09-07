"""The routing rules, including the one that ignores confidence entirely."""
from src.gate import NEVER_AUTO, route
from src.schemas import DefectClass, Route


def test_high_confidence_pass_is_auto_accepted():
    assert route(DefectClass.PASS, 0.995) is Route.AUTO_ACCEPT


def test_middle_band_goes_to_review():
    assert route(DefectClass.SCRATCH, 0.70) is Route.REVIEW


def test_very_low_confidence_is_auto_rejected():
    assert route(DefectClass.SCRATCH, 0.01) is Route.AUTO_REJECT


def test_structural_defects_never_auto_accept():
    """The rule an optimisation will quietly remove.

    Someone measuring review volume will notice these are nearly always right at
    0.999 and widening the gate looks like free money. The saving is not the point.
    """
    for cls in NEVER_AUTO:
        assert route(cls, 0.999) is Route.REVIEW
        assert route(cls, 0.001) is Route.REVIEW


def test_every_class_routes_somewhere():
    for cls in DefectClass:
        assert isinstance(route(cls, 0.5), Route)

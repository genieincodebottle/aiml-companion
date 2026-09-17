import random

from src.task import N_CIRCLES, evaluate, fit_radii
from tests.fakes import random_layout


def test_fitted_layouts_are_always_valid():
    for seed in range(50):
        result = evaluate(random_layout(random.Random(seed)))
        assert result.valid, result.diagnostics
        assert result.score > 0


def test_fit_radii_is_maximal_for_each_circle():
    rng = random.Random(3)
    centers = [(rng.random(), rng.random()) for _ in range(N_CIRCLES)]
    radii = fit_radii(centers)
    # No circle could grow by a visible amount without breaking a constraint.
    for i, (x, y) in enumerate(centers):
        room = min(x, 1 - x, y, 1 - y)
        for j, (a, b) in enumerate(centers):
            if j != i:
                room = min(room, ((x - a) ** 2 + (y - b) ** 2) ** 0.5 - radii[j])
        assert room - radii[i] < 1e-6


def test_overlap_is_rejected_with_a_reason():
    layout = random_layout(random.Random(1))
    layout["radii"] = [r * 3 for r in layout["radii"]]
    result = evaluate(layout)
    assert not result.valid and result.score == 0.0
    assert "overlap" in result.diagnostics or "boundary" in result.diagnostics


def test_wrong_count_and_malformed_input_score_zero():
    layout = random_layout(random.Random(1))
    short = {"centers": layout["centers"][:-1], "radii": layout["radii"][:-1]}
    assert evaluate(short).score == 0.0
    assert evaluate({"centers": "nope"}).score == 0.0
    assert evaluate({"centers": layout["centers"], "radii": [float("nan")] * N_CIRCLES}).score == 0.0


def test_negative_radius_is_rejected():
    layout = random_layout(random.Random(2))
    layout["radii"][0] = -0.01
    assert not evaluate(layout).valid

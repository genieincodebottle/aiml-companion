"""The batching curve and its ceiling, which decide the batch size."""
import pytest

from api.batcher import ceiling, throughput


def test_measured_points_match_the_module():
    assert throughput(1) == pytest.approx(100.0)
    assert throughput(16) == pytest.approx(160.0)


def test_throughput_is_monotone_and_bounded():
    c = ceiling()
    assert c == pytest.approx(166.667, abs=0.01)
    for b in (1, 2, 4, 8, 16, 32, 64, 4096):
        assert throughput(b) < c
    for b in (1, 2, 4, 8, 16, 32):
        assert throughput(b) < throughput(b * 2)


def test_batch_16_is_most_of_the_available_win():
    assert throughput(16) / ceiling() > 0.95
    assert throughput(16) / throughput(1) == pytest.approx(1.6, abs=0.01)

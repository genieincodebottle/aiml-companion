"""Splitting by batch is what stops near-duplicates leaking across the boundary."""
import pytest

from src.data.splits import assert_no_batch_leak, split_by_batch


def _rows(n=500, per_batch=25):
    return [{"image_id": f"i{i}", "batch_id": f"b{i // per_batch}", "label": "pass"}
            for i in range(n)]


def test_no_batch_appears_in_two_splits():
    splits = split_by_batch(_rows(), seed=7)
    assert_no_batch_leak(splits)


def test_every_row_lands_somewhere_exactly_once():
    rows = _rows()
    splits = split_by_batch(rows, seed=7)
    ids = [r["image_id"] for v in splits.values() for r in v]
    assert sorted(ids) == sorted(r["image_id"] for r in rows)


def test_leak_is_detected_when_present():
    splits = {"train": [{"batch_id": "b1"}], "test": [{"batch_id": "b1"}]}
    with pytest.raises(AssertionError):
        assert_no_batch_leak(splits)

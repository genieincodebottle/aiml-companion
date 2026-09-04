"""Error mass, not error rate, is the improvement ceiling."""
from src.metrics.slices import ceilings_sum_to_error_budget, slice_report


def _records(spec):
    out = []
    for name, n, acc in spec:
        hits = round(n * acc)
        out += [{"correct": True, "class": name}] * hits
        out += [{"correct": False, "class": name}] * (n - hits)
    return out


def test_ceilings_sum_to_the_error_budget():
    records = _records([("night", 100, 0.54), ("day", 900, 0.94)])
    rows = slice_report(records, "class")
    overall = sum(r["correct"] for r in records) / len(records)
    assert ceilings_sum_to_error_budget(rows, overall)


def test_the_worse_slice_can_offer_the_smaller_ceiling():
    """The inversion the module is built around.

    Night has 7.7x the error rate of day and a SMALLER ceiling, because a ceiling
    is share times error rate. If this ever flips, the teaching is wrong.
    """
    records = _records([("night", 100, 0.54), ("day", 900, 0.94)])
    rows = {r["slice"]: r for r in slice_report(records, "class")}
    assert rows["night"]["accuracy"] < rows["day"]["accuracy"]
    assert rows["night"]["ceiling"] < rows["day"]["ceiling"]


def test_sorted_by_ceiling_not_by_accuracy():
    records = _records([("night", 100, 0.54), ("day", 900, 0.94)])
    assert [r["slice"] for r in slice_report(records, "class")] == ["day", "night"]

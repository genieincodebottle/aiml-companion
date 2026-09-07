"""Phase I: splits that survive a benchmark which grows every week.

The property that matters is in `TestSplitsAreStableUnderAppend`. Phase IV adds
freshly-rated examples to the benchmark continuously, and a splitter that
reshuffles on every append makes week-over-week comparison meaningless without
ever looking wrong.
"""

from __future__ import annotations

import pytest

from src.benchmark import Benchmark, pending_synthetic
from src.domain import LabelledExample


@pytest.fixture()
def benchmark(domain, config):
    return Benchmark(domain, config.benchmark)


class TestSplitsAreDeterministic:
    def test_the_same_seed_gives_the_same_split(self, benchmark):
        first = {e.id for e in benchmark.split_for("specific").train}
        second = {e.id for e in benchmark.split_for("specific").train}
        assert first == second

    def test_a_different_seed_gives_a_different_split(self, domain, config):
        base = Benchmark(domain, config.benchmark)
        other = Benchmark(domain, {**config.benchmark, "seed": 999})
        assert {e.id for e in base.split_for("specific").test} != {
            e.id for e in other.split_for("specific").test
        }

    def test_the_three_splits_are_disjoint_and_complete(self, benchmark, domain):
        split = benchmark.split_for("specific")
        ids = [e.id for e in split.train + split.validation + split.test]
        assert len(ids) == len(set(ids))
        assert set(ids) == {e.id for e in domain.iter_examples("specific")}


class TestSplitsAreStableUnderAppend:
    def test_adding_examples_does_not_move_the_existing_ones(self, domain, config):
        """The reason splits are content-addressed rather than shuffled.

        Under a positional shuffle, every weekly append would move old examples
        between splits - so this week's test set would contain examples that
        were in last week's training set, and every week-over-week comparison
        would silently stop meaning anything. Nothing would raise.
        """
        benchmark = Benchmark(domain, config.benchmark)
        before = {
            e.id: e.split
            for name in ("train", "validation", "test")
            for e in getattr(benchmark.split_for("specific"), name)
        }

        added = [
            LabelledExample(
                id=f"appended-{i}",
                record_id="rec-001",
                artefact="a new artefact from the weekly review",
                labels={"specific": "FAIL" if i % 2 else "PASS"},
                rationales={"specific": "because"} if i % 2 else {},
            )
            for i in range(12)
        ]
        domain.examples.extend(added)
        try:
            after = {
                e.id: e.split
                for name in ("train", "validation", "test")
                for e in getattr(benchmark.split_for("specific"), name)
            }
            moved = {k for k, v in before.items() if after.get(k) != v}
            assert not moved, f"appending moved existing examples: {sorted(moved)}"
        finally:
            del domain.examples[-len(added) :]


class TestStratification:
    def test_both_classes_reach_every_split(self, benchmark, domain):
        """Stratification's actual guarantee at this scale.

        Each label group is bucketed independently, so neither class can vanish
        from a split - which is what would make specificity or recall
        uncomputable there.

        What it does NOT guarantee is a matching PROPORTION. Threshold hashing
        places each example independently, so on a group of thirteen the test
        split gets four sometimes and one other times. That is the price of
        append-stability (see `split_for`), and this test asserts the property
        the project actually relies on rather than one it cannot deliver at
        seventy examples.
        """
        for criterion in domain.must_have_criteria:
            split = benchmark.split_for(criterion.id)
            for name in ("train", "validation", "test"):
                rows = getattr(split, name)
                labels = {e.labels[criterion.id] for e in rows}
                assert labels == {"PASS", "FAIL"}, (
                    f"{criterion.id}/{name} holds only {labels}; one of the two "
                    "metrics is uncomputable there"
                )

    def test_a_split_with_one_class_only_is_warned_about(self, benchmark, caplog):
        """Recall is computed only over human-PASS examples, so a split can hold
        a comfortable twelve rows and still measure recall over one. Warning on
        the TOTAL size would miss it completely."""
        import logging

        with caplog.at_level(logging.WARNING):
            benchmark.split_for("concise")
        messages = " ".join(r.message for r in caplog.records)
        assert "PASS example" in messages or "FAIL example" in messages

    def test_no_split_is_empty_for_a_gate_criterion(self, benchmark, domain):
        for criterion in domain.must_have_criteria:
            split = benchmark.split_for(criterion.id)
            assert split.train and split.validation and split.test


class TestReporting:
    def test_the_report_flags_an_unbalanced_gate_criterion(self, benchmark):
        report = benchmark.report()["criteria"]
        for cid, row in report.items():
            if row["must_have"]:
                assert row["balanced"], f"{cid} is unbalanced and would hide specificity"

    def test_thin_splits_produce_a_warning(self, benchmark, caplog):
        """A six-example test set produces metrics that look exactly like
        metrics over six hundred, three decimal places and all."""
        import logging

        with caplog.at_level(logging.WARNING):
            benchmark.split_for("concise")
        assert any("close to meaningless" in r.message for r in caplog.records)

    def test_sources_are_counted(self, benchmark):
        sources = benchmark.report()["criteria"]["specific"]["sources"]
        assert "expert" in sources


class TestSynthesis:
    def test_unlabelled_synthetic_rows_are_held_back(self):
        """An LLM-written example labelled by an LLM measures whether two models
        agree. It would raise every metric in the project while adding no
        information."""
        rows = [
            {"id": "a", "labels": {}},
            {"id": "b", "labels": {"specific": "PASS"}},
        ]
        assert [r["id"] for r in pending_synthetic(rows)] == ["a"]


def _fail_fraction(rows, criterion_id: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for e in rows if e.labels[criterion_id] == "FAIL") / len(rows)

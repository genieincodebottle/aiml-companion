"""Phase IV: the floating band, and the check that catches what it cannot see.

The two tests worth reading are `test_the_band_widens_when_raters_disagree` and
`test_new_item_drift_is_invisible_in_the_aggregate`. Together they are the
argument for the whole phase.
"""

from __future__ import annotations

import pytest

from src.monitoring import (
    RatedItem,
    augment_examples,
    check_band,
    detect_drift,
    weekly_sample,
)
from src.serving import DROPPED, SERVED_CLEAN, SERVED_REVISED, ServedResult

CONFIG = {"band_sd_multiplier": 2.0, "check_new_items_separately": True}


def item(eid, raters, judge, *, new=False, rationale="because", mode="spoiler"):
    return RatedItem(
        id=eid,
        record_id="rec-001",
        artefact="a",
        criterion="safe",
        rater_labels=list(raters),
        judge_label=judge,
        is_new_item=new,
        rationale=rationale,
        failure_mode=mode,
    )


class TestMajority:
    def test_majority_of_three(self):
        assert item("a", ["FAIL", "FAIL", "PASS"], "FAIL").majority == "FAIL"
        assert item("b", ["PASS", "FAIL", "PASS"], "PASS").majority == "PASS"

    def test_disagreement_is_measured_not_discarded(self):
        """Rater spread is the input to the band width, not noise to average
        away. Discarding it is what turns this into a fixed threshold."""
        assert item("a", ["FAIL", "FAIL", "FAIL"], "FAIL").disagreement == 0.0
        assert item("b", ["FAIL", "FAIL", "PASS"], "FAIL").disagreement == pytest.approx(
            1 / 3
        )


class TestTheFloatingBand:
    def test_the_band_widens_when_raters_disagree(self):
        """THE idea. On a hard week the humans disagree more, sd grows, and the
        judge is not punished for finding hard what people also found hard.

        Both scenarios below have the judge at the same score. In the unanimous
        one it fails; in the contested one it passes. A fixed threshold cannot
        express that difference, and a team using one either mutes the alert
        after the third hard week or sleeps through a slow slide on easy ones.
        """
        unanimous = [
            item("a", ["FAIL"] * 3, "PASS"),
            item("b", ["FAIL"] * 3, "FAIL"),
            item("c", ["FAIL"] * 3, "FAIL"),
            item("d", ["FAIL"] * 3, "FAIL"),
        ]
        contested = [
            item("a", ["FAIL", "FAIL", "PASS"], "PASS"),
            item("b", ["FAIL", "PASS", "FAIL"], "FAIL"),
            item("c", ["PASS", "FAIL", "FAIL"], "FAIL"),
            item("d", ["FAIL", "FAIL", "PASS"], "FAIL"),
        ]
        tight = check_band(unanimous, CONFIG, metric="specificity")
        wide = check_band(contested, CONFIG, metric="specificity")

        assert tight.judge == pytest.approx(wide.judge)
        assert wide.rater_sd > tight.rater_sd
        assert wide.lower_bound < tight.lower_bound
        assert not tight.in_band and wide.in_band

    def test_unanimous_raters_leave_the_judge_no_slack(self):
        """sd of zero gives a floor equal to the rater mean. Correct: a week
        with no ambiguity is a week with nothing to be generous about."""
        rows = [item(str(i), ["FAIL"] * 3, "FAIL") for i in range(3)]
        check = check_band(rows, CONFIG, metric="specificity")
        assert check.rater_sd == 0.0
        assert check.lower_bound == pytest.approx(check.rater_mean)
        assert check.in_band

    def test_a_judge_with_no_verdicts_is_not_reported_as_healthy(self):
        """Absence of evidence is not evidence of alignment."""
        rows = [
            RatedItem("a", "rec-001", "x", "safe", ["FAIL"] * 3, judge_label=None)
        ]
        assert not check_band(rows, CONFIG, metric="specificity").in_band

    def test_an_empty_slice_does_not_raise(self):
        assert check_band([], CONFIG, metric="specificity").judge is None


class TestNewItemChecking:
    def test_new_item_drift_is_invisible_in_the_aggregate(self):
        """The other half of Phase IV, and the reason it is one line of config.

        Ten established items the judge handles correctly, four new ones it gets
        almost entirely wrong. The aggregate passes because the good majority
        outweighs the bad minority. The judge is now wrong about most of what is
        newly being recommended, and it will keep looking healthy for as long as
        new titles are a minority of traffic.

        Note that the established items carry some rater disagreement, and they
        have to. The first version of this test made every rater unanimous, so
        sd was zero, the floor equalled the rater mean, and the aggregate check
        correctly failed - there is no slack in a week nobody found ambiguous.
        The band was behaving properly and the scenario was unrealistic. Real
        weeks contain disagreement, and that disagreement is precisely what buys
        the judge the room the aggregate check then hides the drift in.
        """
        established = [item(f"e{i}", ["FAIL"] * 3, "FAIL") for i in range(6)]
        established += [item(f"e6{i}", ["FAIL", "PASS", "FAIL"], "FAIL") for i in range(2)]
        established += [item(f"e8{i}", ["FAIL", "FAIL", "PASS"], "FAIL") for i in range(2)]

        new = [
            item("n0", ["FAIL", "FAIL", "PASS"], "PASS", new=True),
            item("n1", ["FAIL", "FAIL", "PASS"], "PASS", new=True),
            item("n2", ["FAIL", "FAIL", "FAIL"], "PASS", new=True),
            item("n3", ["FAIL", "FAIL", "FAIL"], "FAIL", new=True),
        ]

        report = detect_drift(established + new, CONFIG, week=6, criterion="safe")

        assert all(check.in_band for check in report.overall), "aggregate should pass"
        assert any(not check.in_band for check in report.new_items)
        assert report.alert
        assert report.alert_source == "new_items_only"

    def test_no_new_items_is_reported_as_a_gap_not_a_pass(self):
        """A check that did not run is not a check that passed."""
        report = detect_drift(
            [item("e", ["FAIL"] * 3, "FAIL")], CONFIG, week=6, criterion="safe"
        )
        assert any("did not run" in note for note in report.notes)

    def test_the_check_can_be_switched_off(self):
        config = {**CONFIG, "check_new_items_separately": False}
        report = detect_drift(
            [item("n", ["FAIL"] * 3, "PASS", new=True)], config, week=6, criterion="safe"
        )
        assert report.new_items == []


class TestShippedWeeks:
    def test_week_five_is_healthy(self, context):
        from src.services import MonitoringService

        result = MonitoringService(context).check(5)
        assert not result["alert"]

    def test_week_six_alerts_only_on_new_items(self, context):
        """The scenario the repo ships to demonstrate the point end to end."""
        from src.services import MonitoringService

        result = MonitoringService(context).check(6)
        assert result["alert"]
        assert result["reports"][0]["alert_source"] == "new_items_only"

    def test_a_drift_alert_does_not_auto_deploy(self, context):
        """An unsupervised system that re-tunes and self-deploys is editing its
        own success criteria, and it will eventually decide it is doing well."""
        from src.services import MonitoringService

        result = MonitoringService(context).check(6)
        assert result["action"]["retune"] is True
        assert result["action"]["deploy"] is False


class TestSampling:
    def _served(self, record_id, outcome):
        return ServedResult(record_id=record_id, outcome=outcome, artefact="a")

    def test_dropped_artefacts_are_sampled_too(self, domain):
        """Sampling only what was SERVED makes false rejections structurally
        invisible: they never reach a user, so they are never reviewed, so
        recall can decay indefinitely and the only symptom is coverage quietly
        falling."""
        ids = list(domain.records)[:9]
        results = [
            self._served(ids[0], SERVED_CLEAN),
            self._served(ids[1], SERVED_REVISED),
            self._served(ids[2], DROPPED),
        ]
        sample = weekly_sample(results, domain, {"sample_size": 3}, week=1)
        assert {r.outcome for r in sample} == {SERVED_CLEAN, SERVED_REVISED, DROPPED}

    def test_strata_get_equal_shares_not_proportional_ones(self, domain):
        """Drops are rare by design. Proportional sampling would put two or
        three in a 300-item sample, and the false-rejection rate would be
        measured over almost nothing."""
        ids = list(domain.records)
        results = [self._served(i, SERVED_CLEAN) for i in ids[:10]]
        results += [self._served(ids[10], DROPPED)]
        sample = weekly_sample(results, domain, {"sample_size": 6}, week=1)
        assert sum(1 for r in sample if r.outcome == DROPPED) == 1
        assert sum(1 for r in sample if r.outcome == SERVED_CLEAN) == 2


class TestAugmentation:
    def test_passes_are_subsampled_to_preserve_balance(self):
        """Appending a week unfiltered drags the benchmark toward production's
        ~95% PASS one week at a time. Nothing breaks; specificity just stops
        carrying information after a couple of months."""
        rows = [item(f"f{i}", ["FAIL"] * 3, "FAIL") for i in range(5)]
        rows += [item(f"p{i}", ["PASS"] * 3, "PASS") for i in range(50)]
        appended = augment_examples(rows, 0.5)
        fails = sum(1 for r in appended if "FAIL" in r["labels"].values())
        assert fails == 5
        assert len(appended) == pytest.approx(10, abs=1)

    def test_a_fail_without_a_rationale_is_skipped(self):
        """It would be rejected at load anyway. Skipping it here says why,
        rather than breaking the next run with a validation error."""
        rows = [item("f", ["FAIL"] * 3, "FAIL", rationale="")]
        assert augment_examples(rows, 0.5) == []

    def test_rationales_and_modes_are_carried_across(self):
        rows = [item("f", ["FAIL"] * 3, "FAIL")]
        appended = augment_examples(rows, 1.0)
        assert appended[0]["rationales"]["safe"] == "because"
        assert appended[0]["failure_modes"]["safe"] == "spoiler"

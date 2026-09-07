"""Phase IV - Monitoring: keeping a judge aligned after it stops being new.

A judge that is well-aligned on the day it ships will not stay aligned. The
catalogue changes, the generator changes, and what counts as "good" changes too.
Phase IV closes the loop: sample what the judge actually did, have humans
re-label it, and compare.

The acceptance band, which is the idea worth stealing
-----------------------------------------------------
    judge_score  >=  mean(rater_scores)  -  2 * sd(rater_scores)

The threshold FLOATS. It is not 0.85, or any other fixed number, and the reason
is that human raters do not agree with each other by a constant amount.

On a week of genuinely ambiguous artefacts the raters disagree more, ``sd``
widens, and the band widens with it - so the judge is not penalised for finding
hard what people also found hard. On an easy week ``sd`` narrows, the band
tightens, and a real slide still trips it.

A fixed threshold fails in both directions at once. It fires every hard week,
which produces alert fatigue, which produces a muted alert, which is not a
monitor. And on easy weeks it stays silent through a slow degradation that a
tight band would have caught. The band is not a refinement of the fixed
threshold; it is the thing that makes the monitor survive contact with a real
on-call rotation.

The other half: check new items separately
------------------------------------------
The band is applied twice - to the whole weekly sample, and again to items added
to the catalogue recently. A judge can be perfectly fine on the established
catalogue and wrong about everything new, and that is not a hypothetical, it is
what catalogue drift IS. Averaged into a sample dominated by familiar items, it
does not move the aggregate at all.

Checking new items separately is the single line that makes this a shift
detector rather than a generic regression test. ``docs/results/`` walks through
a week where the aggregate check passes and the new-item check fails.

And what happens on an alert
----------------------------
A drift alert triggers Phase II re-tuning on the augmented benchmark. The new
rubric is STAGED. A human reads the diff and decides. ``auto_deploy`` is off and
should stay off: a system that re-tunes and self-deploys is editing its own
success criteria without supervision, and it will eventually conclude it is
doing well.
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from .domain import Domain
from .metrics import FAIL, PASS
from .serving import DROPPED, SERVED_CLEAN, SERVED_REVISED, ServedResult

log = logging.getLogger(__name__)

STRATA = (SERVED_CLEAN, SERVED_REVISED, DROPPED)


@dataclass
class RatedItem:
    """One artefact, one criterion, several independent human verdicts."""

    id: str
    record_id: str
    artefact: str
    criterion: str
    rater_labels: list[str]
    outcome: str = SERVED_CLEAN
    rationale: str = ""
    failure_mode: str | None = None
    judge_label: str | None = None
    is_new_item: bool = False

    @property
    def majority(self) -> str:
        """Ground truth for the week: the majority verdict of the panel.

        Requires an odd panel to be well-defined. With an even one a tie has to
        break somewhere, and any consistent rule silently biases the whole
        week's measurement in that direction. ``raters_per_item: 3`` in the
        config is a floor, not a suggestion.
        """
        return FAIL if self.rater_labels.count(FAIL) > len(self.rater_labels) / 2 else PASS

    @property
    def disagreement(self) -> float:
        """Fraction of raters differing from the majority. Feeds the band width."""
        if not self.rater_labels:
            return 0.0
        majority = self.majority
        return sum(1 for label in self.rater_labels if label != majority) / len(
            self.rater_labels
        )


@dataclass
class BandCheck:
    metric: str
    judge: float | None
    rater_mean: float
    rater_sd: float
    lower_bound: float
    n: int
    in_band: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "judge": None if self.judge is None else round(self.judge, 4),
            "rater_mean": round(self.rater_mean, 4),
            "rater_sd": round(self.rater_sd, 4),
            "lower_bound": round(self.lower_bound, 4),
            "n": self.n,
            "in_band": self.in_band,
        }


@dataclass
class DriftReport:
    week: int
    criterion: str
    overall: list[BandCheck] = field(default_factory=list)
    new_items: list[BandCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def alert(self) -> bool:
        return any(not c.in_band for c in self.overall + self.new_items)

    @property
    def alert_source(self) -> str:
        if any(not c.in_band for c in self.overall):
            return "overall"
        if any(not c.in_band for c in self.new_items):
            # The interesting one. The judge is fine on the catalogue it was
            # tuned against and wrong on what arrived since.
            return "new_items_only"
        return ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "week": self.week,
            "criterion": self.criterion,
            "alert": self.alert,
            "alert_source": self.alert_source,
            "overall": [c.as_dict() for c in self.overall],
            "new_items": [c.as_dict() for c in self.new_items],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def weekly_sample(
    results: list[ServedResult],
    domain: Domain,
    config: dict[str, Any],
    *,
    week: int = 0,
) -> list[ServedResult]:
    """Stratify by what the judge DID, not by what the catalogue looks like.

    The strata are ``served_without_revision``, ``served_after_revision`` and
    ``dropped``, and the third is the one every naive sampler omits. Sampling
    only served artefacts makes the judge's false rejections structurally
    invisible: they never reach a user, so they are never reviewed, so recall
    can decay indefinitely without anyone seeing it. The bill arrives as
    coverage quietly falling and nobody able to say why.

    The sample is also biased toward recently-added records, because drift
    arrives with new content first. ``new_item_oversample`` controls how hard.
    """
    size = int(config.get("sample_size", 300))
    oversample = float(config.get("new_item_oversample", 0.0))
    lookback = int(config.get("new_item_lookback_weeks", 2))
    strata = list(config.get("strata") or STRATA)

    buckets: dict[str, list[ServedResult]] = {s: [] for s in strata}
    for result in results:
        if result.outcome in buckets:
            buckets[result.outcome].append(result)

    # An equal share per stratum, not a share proportional to volume. Drops are
    # rare by design, so proportional sampling would put two or three of them in
    # a 300-item sample and the false-rejection rate would be measured over
    # almost nothing.
    per_stratum = max(1, size // max(1, len(strata)))
    sample: list[ServedResult] = []
    for name in strata:
        rows = buckets.get(name, [])
        rows = sorted(
            rows,
            key=lambda r: (
                # Newly-added records sort first in proportion to the
                # oversample weight; the hash keeps it deterministic and stable
                # week to week rather than merely random.
                -oversample
                if domain.records[r.record_id].added_week >= week - lookback
                else 0.0,
                _hash(week, r.record_id),
            ),
        )
        sample.extend(rows[:per_stratum])
    return sample


# ---------------------------------------------------------------------------
# The band
# ---------------------------------------------------------------------------


def check_band(
    items: list[RatedItem], config: dict[str, Any], *, metric: str
) -> BandCheck:
    """Score the judge and each rater against the same majority label.

    ``metric`` is "specificity" or "recall". Reasoning agreement is not checked
    here: it would need a written rationale from every rater on every item,
    which is a real cost a real team may or may not choose to pay. This
    teaching-scale sample does not carry them, and inventing the number would be
    worse than omitting it.
    """
    multiplier = float(config.get("band_sd_multiplier", 2.0))
    relevant = [i for i in items if i.majority == (FAIL if metric == "specificity" else PASS)]
    n_raters = max((len(i.rater_labels) for i in items), default=0)

    if not relevant or n_raters == 0:
        return BandCheck(metric, None, 0.0, 0.0, 0.0, 0, True)

    judge_score = _agreement(
        [(i.judge_label, i.majority) for i in relevant if i.judge_label is not None]
    )
    rater_scores = [
        _agreement([(i.rater_labels[r], i.majority) for i in relevant])
        for r in range(n_raters)
    ]

    mean = statistics.fmean(rater_scores)
    # Population sd, and it is zero when every rater agrees with the majority on
    # every item. That is correct: a week where the raters were unanimous gives
    # the judge no slack, because there was no ambiguity to be generous about.
    sd = statistics.pstdev(rater_scores) if len(rater_scores) > 1 else 0.0
    lower = mean - multiplier * sd

    return BandCheck(
        metric=metric,
        judge=judge_score,
        rater_mean=mean,
        rater_sd=sd,
        lower_bound=lower,
        n=len(relevant),
        # Judge is None when it produced no verdict on any relevant item, which
        # is not evidence of health. Treat it as out of band and say why.
        in_band=judge_score is not None and judge_score >= lower - 1e-9,
    )


def detect_drift(
    items: list[RatedItem], config: dict[str, Any], *, week: int, criterion: str
) -> DriftReport:
    report = DriftReport(week=week, criterion=criterion)
    new_items = [i for i in items if i.is_new_item]

    for metric in ("specificity", "recall"):
        report.overall.append(check_band(items, config, metric=metric))

    if config.get("check_new_items_separately", True) and new_items:
        for metric in ("specificity", "recall"):
            check = check_band(new_items, config, metric=metric)
            report.new_items.append(check)
            if not check.in_band:
                report.notes.append(
                    f"{metric} is inside the band across the full sample and "
                    f"outside it on the {len(new_items)} recently-added items. "
                    "A threshold calibrated on established content would have "
                    "reported this week as healthy. This is what catalogue "
                    "drift looks like before it is large enough to move an "
                    "aggregate."
                )

    if not new_items and config.get("check_new_items_separately", True):
        report.notes.append(
            "No recently-added items in this week's sample, so the shift check "
            "did not run. That is a gap, not a pass: raise "
            "monitoring.new_item_oversample or confirm nothing was added."
        )

    return report


# ---------------------------------------------------------------------------
# Benchmark augmentation
# ---------------------------------------------------------------------------


def augment_examples(
    items: list[RatedItem], target_fail_fraction: float
) -> list[dict[str, Any]]:
    """Turn a rated week into benchmark rows, subsampled to preserve balance.

    Phase I is not finished at the end of Phase I. Without this the benchmark
    ages out: it keeps describing the catalogue as it was on the day it was
    written, and judge-human agreement measured against it slowly stops meaning
    anything about live traffic.

    Subsampling matters. Production is mostly PASS, so appending a week
    unfiltered would drag the benchmark toward the natural distribution one week
    at a time, and specificity would quietly become unmeasurable over a couple
    of months. Nothing would break; the number would just stop carrying
    information.
    """
    fails = [i for i in items if i.majority == FAIL]
    passes = [i for i in items if i.majority == PASS]

    if fails and target_fail_fraction > 0:
        keep_passes = int(len(fails) * (1 - target_fail_fraction) / target_fail_fraction)
    else:
        keep_passes = len(passes)

    passes = sorted(passes, key=lambda i: _hash(i.id))[:keep_passes]

    rows: list[dict[str, Any]] = []
    for item in fails + passes:
        row: dict[str, Any] = {
            "id": item.id,
            "record_id": item.record_id,
            "artefact": item.artefact,
            "labels": {item.criterion: item.majority},
            "rationales": {},
            "failure_modes": {},
            "source": f"hitl_week_{item.id.split('-')[1] if '-' in item.id else '0'}",
        }
        if item.majority == FAIL:
            if not item.rationale:
                # src/domain.py refuses to load this, and it should. A FAIL with
                # no rationale is invisible to reasoning agreement, so appending
                # one shrinks the training signal while growing the row count.
                log.warning(
                    "skipping %s: majority FAIL with no rationale, so it cannot "
                    "be used for reasoning alignment", item.id,
                )
                continue
            row["rationales"] = {item.criterion: item.rationale}
            if item.failure_mode:
                row["failure_modes"] = {item.criterion: item.failure_mode}
        rows.append(row)
    return rows


def load_rated_items(
    rows: Iterable[dict[str, Any]],
    domain: Domain,
    *,
    week: int,
    lookback_weeks: int = 2,
) -> list[RatedItem]:
    """Attach the "recently added" flag that the shift check keys on.

    ``lookback_weeks`` is a window, not an equality test. A judge does not
    become wrong about a title the instant it is added and correct again seven
    days later, and an exact-week comparison would put a fortnight-old batch of
    arrivals into the established bucket - which is precisely the bucket where
    the drift they carry gets averaged away.
    """
    items: list[RatedItem] = []
    for row in rows:
        record_id = row["record_id"]
        record = domain.records.get(record_id)
        recently_added = bool(
            record
            and record.added_week > 0
            and record.added_week >= week - lookback_weeks
        )
        items.append(
            RatedItem(
                id=row["id"],
                record_id=record_id,
                artefact=row["artefact"],
                criterion=row["criterion"],
                rater_labels=list(row["rater_labels"]),
                outcome=row.get("outcome", SERVED_CLEAN),
                rationale=row.get("rationale", ""),
                failure_mode=row.get("failure_mode"),
                judge_label=row.get("judge_label"),
                is_new_item=recently_added,
            )
        )
    return items


def _agreement(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(1 for got, want in pairs if got == want) / len(pairs)


def _hash(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")

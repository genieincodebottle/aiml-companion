"""Phase I - Birth: the benchmark, and the splits that keep it honest.

The most human-intensive phase, and the one it is most tempting to skip. You
cannot. Every later phase is measured against this data: RART early-stops on
it, the drift monitor compares the judge to raters labelling under the same
guidelines, and every number the project reports is a number about this file.
A judge tuned against a careless benchmark is carefully aligned to nothing.

Three sources, deliberately
---------------------------
i.   **Expert-written.** Hand-authored, each with a known failure mode and a
     rationale. Covers the cases the other two sources cannot reliably produce.
ii.  **LLM-synthesised near the boundary.** Naturalistic sampling almost never
     surfaces hard cases, because hard cases are rare by definition. Synthesis
     targets them directly - and every synthetic example still needs a HUMAN
     label before it counts. An LLM-labelled LLM-written example teaches the
     judge to agree with the synthesiser, which is not the target and is
     invisible in the metrics.
iii. **Sampled from production.** The reality check. Whatever you imagined the
     failure distribution looked like, this is what it actually looks like.

Class balance, and why the numbers are not what they seem
---------------------------------------------------------
The benchmark is held near 50/50 per criterion. Production defect rates are a
few percent. A naturalistically-sampled benchmark would therefore be ~95% PASS,
and a judge that answered PASS to everything would score 95% while catching
nothing at all.

The consequence has to be repeated wherever these numbers appear, because it is
the single easiest thing to get wrong when reporting them: **alignment metrics
computed here are not defect rates.** They measure agreement on a
difficulty-enriched, class-balanced set. The live defect rate comes from Phase
IV's weekly sample, and it is a different number with a different meaning.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

from .domain import Domain, LabelledExample

TRAIN, VALIDATION, TEST = "train", "validation", "test"


@dataclass
class Split:
    train: list[LabelledExample]
    validation: list[LabelledExample]
    test: list[LabelledExample]

    def sizes(self) -> dict[str, int]:
        return {
            TRAIN: len(self.train),
            VALIDATION: len(self.validation),
            TEST: len(self.test),
        }


class Benchmark:
    def __init__(self, domain: Domain, config: dict[str, Any]) -> None:
        self.domain = domain
        self.config = config
        self.seed = int(config.get("seed", 0))
        self.ratios = dict(config.get("split") or {})

    def split_for(self, criterion_id: str) -> Split:
        """Stratified, deterministic, and content-addressed.

        The bucket for an example is a hash of its id and the criterion, not a
        shuffle. Three consequences, all of them load-bearing:

        * Runs are reproducible without carrying a shuffled index around.
        * **Adding examples does not reshuffle the existing ones.** Phase IV
          appends ~300 newly-rated examples every week. Under a positional
          shuffle, each append would move old examples between splits, so
          this week's test set would contain examples that were in last
          week's training set, and every week-over-week comparison would be
          meaningless. Under a content hash, new examples land in their own
          buckets and the old ones do not move.
        * Renaming an example moves it, which is the correct behaviour: a
          rewritten artefact is a different example.

        Stratification is per label: the two label groups are bucketed
        independently, so each split gets roughly the same share of each class
        rather than a random draw from the mixture.

        THE TRADE-OFF, stated plainly, because it is real and it is not
        resolvable at this benchmark size. Threshold hashing guarantees
        append-stability but not exact split proportions: each example lands
        independently, so on a group of thirteen the test split gets four
        sometimes and one other times. Rank-based assignment gives exact
        proportions and is NOT append-stable.

        This project takes stability, for two reasons. Comparability across
        weekly appends is the entire point of Phase IV, and it fails silently
        when broken. Proportion wobble fails loudly instead: the warnings below
        fire, and the Wilson intervals beside every metric are wide enough to
        show it. Prefer the failure you can see.
        """
        by_label: dict[str, list[LabelledExample]] = {}
        for example in self.domain.iter_examples(criterion_id):
            by_label.setdefault(example.labels[criterion_id], []).append(example)

        train_ratio = self.ratios.get(TRAIN, 0.55)
        val_ratio = self.ratios.get(VALIDATION, 0.20)

        buckets: dict[str, list[LabelledExample]] = {TRAIN: [], VALIDATION: [], TEST: []}
        for label, examples in by_label.items():
            for example in sorted(examples, key=lambda e: e.id):
                # Each example's bucket is a function of its OWN id and nothing
                # else - not its position in a sorted list, not the size of the
                # group. That is what makes the split stable under append.
                #
                # An earlier version ranked the group by hash and cut at
                # round(n * ratio). It looked equivalent and was not: adding
                # twelve examples moved the cut points, and three existing
                # examples crossed a boundary. Under weekly appends that means
                # this week's test set contains examples that were in last
                # week's training set, so every week-over-week comparison
                # quietly stops meaning anything. It is caught by
                # tests/test_benchmark.py and by nothing else - no error, no
                # warning, just numbers that are no longer comparable.
                position = (_hash(self.seed, criterion_id, example.id) % 10_000) / 10_000
                bucket = (
                    TRAIN
                    if position < train_ratio
                    else VALIDATION
                    if position < train_ratio + val_ratio
                    else TEST
                )
                example.split = bucket
                buckets[bucket].append(example)

        split = Split(buckets[TRAIN], buckets[VALIDATION], buckets[TEST])
        _warn_on_thin_splits(criterion_id, split, self.domain, criterion_id)
        return split

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {"domain": self.domain.name, "criteria": {}}
        target = float(self.config.get("target_fail_fraction", 0.5))
        for criterion in self.domain.criteria:
            split = self.split_for(criterion.id)
            examples = list(self.domain.iter_examples(criterion.id))
            fails = sum(1 for e in examples if e.labels[criterion.id] == "FAIL")
            fraction = fails / len(examples) if examples else 0.0
            out["criteria"][criterion.id] = {
                "must_have": criterion.must_have,
                "n": len(examples),
                "fail_fraction": round(fraction, 3),
                "target_fail_fraction": target,
                # Only meaningful for criteria you intend to tune. A soft
                # criterion that is right by construction does not need
                # balancing, and forcing it would be ceremony.
                "balanced": abs(fraction - target) <= 0.15 or not criterion.must_have,
                "splits": split.sizes(),
                "sources": _count(e.source for e in examples),
            }
        return out


def _warn_on_thin_splits(
    label: str, split: Split, domain: Domain, criterion_id: str
) -> None:
    """Say it out loud when a split is too thin to support a conclusion.

    A six-example test set produces metrics that look exactly like metrics
    computed over six hundred - three decimal places and all - and nothing in
    the output hints that the 95% interval spans forty points. Printing the
    warning next to the number is the cheapest available defence against
    someone quoting it in a slide.

    BOTH classes are checked, and the second one matters more than it looks.
    Specificity is measured only over human-FAIL examples and recall only over
    human-PASS ones, so a split can hold a comfortable twelve examples and
    still compute recall over a single one. Warning on the total would miss it
    entirely.
    """
    import logging

    log = logging.getLogger(__name__)
    for name, rows in (
        (TRAIN, split.train),
        (VALIDATION, split.validation),
        (TEST, split.test),
    ):
        for outcome, metric in (("FAIL", "specificity"), ("PASS", "recall")):
            count = sum(1 for e in rows if e.labels.get(criterion_id) == outcome)
            if count < 3:
                log.warning(
                    "criterion %r: the %s split holds only %d %s example(s), so "
                    "%s over it is close to meaningless - one flipped verdict "
                    "moves it by %d points. Grow the benchmark before trusting "
                    "a comparison.",
                    label, name, count, outcome, metric,
                    round(100 / max(count, 1)),
                )


def pending_synthetic(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Synthesised examples still waiting for a human label.

    They are written to a separate file with ``label: null`` and are NOT loaded
    into the benchmark until a human fills it in. The gap is on purpose. An
    LLM-written example labelled by an LLM measures whether two models agree,
    and it will happily inflate every metric in the project while looking like
    more data.
    """
    return [r for r in rows if not r.get("labels")]


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _hash(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")

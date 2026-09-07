"""The pluggable seam: what is being judged, and by what criteria.

Everything downstream of this module - the benchmark, RART, the serving loop,
the drift monitor - is written against :class:`Domain` and knows nothing about
films or support tickets. Swapping ``domain:`` in configs/base.yaml swaps the
whole subject matter without touching a line of Python.

That is not architectural decoration. It is the claim the paper's framework
makes, tested: if the lifecycle is a general method rather than a Netflix
story, then a second domain should need data and prose, not code. The repo
ships two domains for exactly this reason, and
``tests/test_domain.py::test_no_domain_specific_logic_in_src`` fails the build
if a domain name ever appears in the engine.

The one structural idea worth stealing
--------------------------------------
A criterion has a single ``guideline`` field, and it is used twice: it is the
instruction humans label against, and it is the seed rubric the judge is tuned
from. One text, two readers.

Most teams write these separately, and they drift apart within a month - the
rater guidance gets a clarification, the judge prompt does not, and now
judge-human disagreement is measuring a documentation gap rather than a model
failure. You will spend a week tuning the judge before anyone notices. Making
them the same field makes that class of bug unrepresentable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml

from .config import ROOT

VALID_LABELS = ("PASS", "FAIL")


@dataclass(frozen=True)
class Criterion:
    id: str
    display: str
    must_have: bool
    #: Doubles as the human labelling guideline AND the seed rubric. See above.
    guideline: str
    failure_modes: tuple[str, ...] = ()

    @property
    def is_gate(self) -> bool:
        """Must-have criteria block serving. Soft ones are advisory.

        The split matters at deployment: a must-have failure drops the artefact,
        a soft failure is recorded and served. Collapsing the two gives you
        either a gate that rejects for tone, or a gate that ships falsehoods.
        """
        return self.must_have


@dataclass
class Record:
    """One subject the artefact is about, plus everything it may claim.

    Neutral field names on purpose. ``subject`` is the film / the ticket,
    ``references`` is what it is being compared or responded to, and ``facts``
    is the closed set of assertions a grounded artefact may draw on. A domain
    that cannot express itself in those three is probably two domains.
    """

    id: str
    subject: dict[str, Any]
    references: list[dict[str, Any]] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    #: Set by the monitor. Newly-added records are where drift shows first.
    added_week: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "references": self.references,
            "facts": self.facts,
            "context": self.context,
        }


@dataclass
class LabelledExample:
    """One artefact, with a human verdict per criterion.

    ``rationales`` is not decoration and not documentation. It is a training
    signal: the reasoning meta-judge compares the judge's stated reason against
    the human's, and without a rationale there is nothing to compare, so
    reasoning agreement is unmeasurable and RART degenerates to label-only
    tuning. If you take one thing from Phase I into your own work, take this:
    **collect the reason, not just the verdict.** It costs a rater ten extra
    seconds and it is the difference between a judge that is right and a judge
    that is right for the right reason.
    """

    id: str
    record_id: str
    artefact: str
    labels: dict[str, str]
    rationales: dict[str, str] = field(default_factory=dict)
    failure_modes: dict[str, str] = field(default_factory=dict)
    source: str = "expert"  # expert | synthesised | production | hitl_week_N
    split: str = ""  # train | validation | test

    def label_for(self, criterion_id: str) -> str | None:
        return self.labels.get(criterion_id)


class Domain:
    def __init__(self, name: str, spec: dict[str, Any], path: Path) -> None:
        self.name = name
        self.path = path
        self._spec = spec

        self.display_name: str = spec.get("display_name", name)
        self.artefact_noun: str = spec.get("artefact_noun", "artefact")
        self.subject_noun: str = spec.get("subject_noun", "item")
        self.generation: dict[str, Any] = dict(spec.get("generation") or {})

        self.criteria: list[Criterion] = [
            Criterion(
                id=c["id"],
                display=c.get("display", c["id"]),
                must_have=bool(c.get("must_have", False)),
                guideline=c["guideline"].strip(),
                failure_modes=tuple(c.get("failure_modes") or ()),
            )
            for c in spec.get("criteria") or []
        ]
        _validate_criteria(self.criteria, name)

        self.records: dict[str, Record] = {
            r["id"]: Record(
                id=r["id"],
                subject=r.get("subject") or {},
                references=r.get("references") or [],
                facts=r.get("facts") or [],
                context=r.get("context") or {},
                added_week=int(r.get("added_week", 0)),
            )
            for r in _load_json(path / spec.get("records", "records.json"))
        }

        self.examples: list[LabelledExample] = [
            LabelledExample(
                id=e["id"],
                record_id=e["record_id"],
                artefact=e["artefact"],
                labels=dict(e.get("labels") or {}),
                rationales=dict(e.get("rationales") or {}),
                failure_modes=dict(e.get("failure_modes") or {}),
                source=e.get("source", "expert"),
            )
            for e in _load_jsonl(path / spec.get("seed_labels", "labels.jsonl"))
        ]
        self._validate_examples()

    # ------------------------------------------------------------- accessors
    def criterion(self, criterion_id: str) -> Criterion:
        for c in self.criteria:
            if c.id == criterion_id:
                return c
        raise KeyError(
            f"domain {self.name!r} has no criterion {criterion_id!r}; "
            f"available: {[c.id for c in self.criteria]}"
        )

    @property
    def must_have_criteria(self) -> list[Criterion]:
        return [c for c in self.criteria if c.must_have]

    def seed_rubric(self, criterion_id: str) -> str:
        """The starting point for RART: the human labelling guideline, verbatim.

        Starting from the guideline rather than from an empty prompt means
        iteration 0 is already a defensible judge, and every later iteration is
        measured against a baseline someone is willing to defend in a review.
        A tuned rubric that cannot beat the guideline it started from is a
        result worth reporting, not a bug to hide - and on one criterion in the
        reference domain, that is exactly what happens.
        """
        return self.criterion(criterion_id).guideline

    def record(self, record_id: str) -> Record:
        try:
            return self.records[record_id]
        except KeyError:
            raise KeyError(
                f"domain {self.name!r} has no record {record_id!r}"
            ) from None

    def iter_examples(self, criterion_id: str) -> Iterator[LabelledExample]:
        """Only examples carrying a verdict for this criterion.

        Not every artefact is labelled on every criterion, and that is fine.
        Silently treating a missing label as PASS would inflate specificity by
        counting unlabelled examples as agreements, which is the most flattering
        possible bug and the hardest to notice.
        """
        for example in self.examples:
            if criterion_id in example.labels:
                yield example

    # ------------------------------------------------------------ validation
    def _validate_examples(self) -> None:
        known = {c.id for c in self.criteria}
        seen: set[str] = set()
        problems: list[str] = []

        for example in self.examples:
            if example.id in seen:
                problems.append(f"duplicate example id {example.id!r}")
            seen.add(example.id)

            if example.record_id not in self.records:
                problems.append(
                    f"{example.id}: record_id {example.record_id!r} does not exist"
                )

            for criterion_id, label in example.labels.items():
                if criterion_id not in known:
                    problems.append(
                        f"{example.id}: label for unknown criterion {criterion_id!r}"
                    )
                if label not in VALID_LABELS:
                    problems.append(
                        f"{example.id}: label {label!r} is not one of {VALID_LABELS}"
                    )
                # A FAIL without a rationale is the one that has to be fatal.
                # It is silently unusable for reasoning agreement - the metric
                # simply skips it - so the benchmark looks the right size while
                # the signal RART depends on quietly shrinks.
                if label == "FAIL" and not example.rationales.get(criterion_id):
                    problems.append(
                        f"{example.id}: FAIL on {criterion_id!r} with no rationale. "
                        "Reasoning agreement cannot be computed for it."
                    )

        if problems:
            raise ValueError(
                f"domain {self.name!r} has {len(problems)} data problem(s):\n  "
                + "\n  ".join(problems[:20])
            )

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "domain": self.name,
            "records": len(self.records),
            "examples": len(self.examples),
            "criteria": {},
        }
        for criterion in self.criteria:
            labels = [e.labels[criterion.id] for e in self.iter_examples(criterion.id)]
            total = len(labels)
            fails = sum(1 for label in labels if label == "FAIL")
            out["criteria"][criterion.id] = {
                "must_have": criterion.must_have,
                "labelled": total,
                "fail": fails,
                "fail_fraction": round(fails / total, 3) if total else 0.0,
            }
        return out


def _validate_criteria(criteria: list[Criterion], domain_name: str) -> None:
    if not criteria:
        raise ValueError(f"domain {domain_name!r} defines no criteria")
    if not any(c.must_have for c in criteria):
        raise ValueError(
            f"domain {domain_name!r} has no must-have criterion, so the serving "
            "gate would never reject anything and Phase III is a no-op."
        )
    ids = [c.id for c in criteria]
    if len(ids) != len(set(ids)):
        raise ValueError(f"domain {domain_name!r} has duplicate criterion ids: {ids}")
    for c in criteria:
        # A rubric this short cannot carry pass conditions, fail conditions and
        # a boundary case, so it is not a guideline a human could label against
        # either. Catch it at load rather than after a tuning run.
        if len(c.guideline.split()) < 25:
            raise ValueError(
                f"criterion {c.id!r} in domain {domain_name!r} has a "
                f"{len(c.guideline.split())}-word guideline. It has to be usable "
                "by a human rater AND as a seed rubric; write the pass "
                "condition, the fail condition and at least one boundary case."
            )


def _load_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} line {line_number} is not valid JSON: {exc}"
                ) from None
    return rows


def load_domain(name: str, root: Path | None = None) -> Domain:
    root = root or ROOT
    path = root / "domains" / name
    spec_path = path / "domain.yaml"
    if not spec_path.exists():
        available = sorted(
            p.name for p in (root / "domains").iterdir() if p.is_dir()
        )
        raise FileNotFoundError(
            f"no domain {name!r} at {spec_path}. Available: {available}. "
            "See docs/adding-a-domain.md."
        )
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}
    return Domain(name, spec, path)


@lru_cache(maxsize=4)
def get_domain(name: str) -> Domain:
    return load_domain(name)

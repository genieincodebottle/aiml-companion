"""The domain contract, and the shipped data.

The most valuable test here is `test_no_domain_specific_logic_in_src`: it fails
the build if a domain name is ever hardcoded in the engine. That is the whole
portability claim, checked rather than asserted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.config import ROOT
from src.domain import Domain, load_domain

DOMAIN_NAMES = ["recommendation", "support"]


@pytest.fixture(params=DOMAIN_NAMES)
def any_domain(request) -> Domain:
    return load_domain(request.param)


class TestShippedDomainsLoad:
    def test_domain_loads_and_validates(self, any_domain):
        assert any_domain.records
        assert any_domain.examples
        assert any_domain.must_have_criteria

    def test_every_example_points_at_a_real_record(self, any_domain):
        for example in any_domain.examples:
            assert example.record_id in any_domain.records

    def test_every_fail_carries_a_rationale(self, any_domain):
        """Enforced at load, pinned here because it is the quiet one.

        A FAIL without a rationale is invisible to reasoning agreement - the
        metric skips it - so the benchmark keeps its row count while the signal
        RART depends on shrinks. Nothing errors and the numbers still look fine.
        """
        for example in any_domain.examples:
            for cid, label in example.labels.items():
                if label == "FAIL":
                    assert example.rationales.get(cid), f"{example.id}/{cid}"

    def test_failure_modes_come_from_the_declared_vocabulary(self, any_domain):
        """A mode outside the vocabulary scores as a reason mismatch forever,
        and it looks like a model problem rather than a typo in the data."""
        for example in any_domain.examples:
            for cid, mode in example.failure_modes.items():
                declared = any_domain.criterion(cid).failure_modes
                assert mode in declared, f"{example.id}/{cid}: {mode!r} not in {declared}"

    def test_gate_criteria_are_near_class_balance(self, any_domain):
        """Only the criteria you intend to tune. A soft criterion that is right
        by construction does not need balancing and forcing it is ceremony."""
        stats = any_domain.stats()["criteria"]
        for criterion in any_domain.must_have_criteria:
            fraction = stats[criterion.id]["fail_fraction"]
            assert 0.35 <= fraction <= 0.65, (
                f"{criterion.id} is {fraction:.2f} FAIL; specificity is not "
                "measurable on a lopsided set"
            )

    def test_the_seed_rubric_is_the_labelling_guideline(self, any_domain):
        """One text, two readers. Keeping them as separate fields is how rater
        guidance and judge prompt drift apart, after which judge-human
        disagreement is measuring a documentation gap."""
        for criterion in any_domain.criteria:
            assert any_domain.seed_rubric(criterion.id) == criterion.guideline


class TestValidationRejectsBadData:
    def _domain(self, examples, criteria=None):
        spec = {
            "criteria": criteria
            or [
                {
                    "id": "c1",
                    "must_have": True,
                    "guideline": "PASS when it is fine. FAIL when it is not. "
                    "BOUNDARY cases sit close to the line and this sentence "
                    "exists only to clear the minimum guideline length check "
                    "that the loader applies to every criterion it reads.",
                }
            ],
            "records": "records.json",
            "seed_labels": "labels.jsonl",
        }
        return spec, examples

    def test_a_domain_with_no_must_have_criterion_is_rejected(self):
        """Without one, the serving gate rejects nothing and Phase III is a
        no-op that still reports a pass rate."""
        spec, _ = self._domain([])
        spec["criteria"][0]["must_have"] = False
        with pytest.raises(ValueError, match="no must-have criterion"):
            Domain("t", spec, ROOT / "domains" / "recommendation")

    def test_a_stub_guideline_is_rejected(self):
        spec, _ = self._domain([])
        spec["criteria"][0]["guideline"] = "Be good."
        with pytest.raises(ValueError, match="guideline"):
            Domain("t", spec, ROOT / "domains" / "recommendation")

    def test_an_unknown_domain_names_the_available_ones(self):
        with pytest.raises(FileNotFoundError) as exc:
            load_domain("does-not-exist")
        assert "recommendation" in str(exc.value)


class TestTheEngineIsDomainAgnostic:
    def test_no_domain_specific_logic_in_src(self):
        """THE portability test.

        The lifecycle is presented as a general method. If that is true, adding
        a domain costs data and prose and nothing else - so no domain name may
        appear anywhere in src/. The moment one does, the second domain is
        working because somebody special-cased it, and the claim is false.

        What it checks is STRING LITERALS, not text. The first version of this
        test grepped the source for the words and failed on `support` used as
        an ordinary noun - a local variable holding the support corpus, and
        half the prose in the comments. Text search cannot tell a domain name
        from English.

        A hardcoded domain name is a string literal: `if domain == "support"`,
        `Path("domains/recommendation")`, a dict keyed by domain. Docstrings are
        excluded because prose is allowed to name the domains, and
        `src/config.py` is excluded because it carries the default, which is
        configuration rather than logic.
        """
        known = {p.name.lower() for p in (ROOT / "domains").iterdir() if p.is_dir()}
        offenders: list[str] = []

        for path in (ROOT / "src").rglob("*.py"):
            if path.name == "config.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            }
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if id(node) in docstrings:
                    continue
                if node.value.strip().lower() in known:
                    offenders.append(f"{path.name}:{node.lineno}: {node.value!r}")

        assert not offenders, (
            "a domain name is hardcoded in engine code, so the second domain is "
            f"not actually a drop-in: {offenders}"
        )

    def test_both_domains_expose_the_same_interface(self):
        """What the engine relies on, and nothing more."""
        for name in DOMAIN_NAMES:
            domain = load_domain(name)
            assert domain.artefact_noun and domain.subject_noun
            assert domain.generation.get("instruction")
            for criterion in domain.criteria:
                assert criterion.guideline
                assert isinstance(criterion.must_have, bool)

    def test_the_two_domains_share_no_criterion_ids(self):
        """The mechanism ports; the criteria do not. A team reusing another
        team's rubrics is porting the wrong half."""
        first = {c.id for c in load_domain("recommendation").criteria}
        second = {c.id for c in load_domain("support").criteria}
        assert (first & second) == {"concise"}, (
            "only the soft length criterion should be shared; anything else "
            "suggests one domain was written by copying the other"
        )


def test_hitl_weeks_are_wellformed():
    directory = Path(ROOT / "domains" / "recommendation" / "hitl")
    files = sorted(directory.glob("week_*.jsonl"))
    assert files, "no weekly samples shipped"
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            import json

            row = json.loads(line)
            # An even panel makes the majority undefined on a tie, and any
            # consistent tie-break silently biases the whole week.
            assert len(row["rater_labels"]) % 2 == 1, row["id"]
            assert len(row["rater_labels"]) >= 3, row["id"]

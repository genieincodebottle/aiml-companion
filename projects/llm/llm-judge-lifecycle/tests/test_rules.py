"""The rule engine, and the parser bug that nearly went unnoticed.

The engine is the offline baseline and the offline evaluator RART optimises
against, so a parsing bug here does not raise - it quietly changes what every
metric in the project means.
"""

from __future__ import annotations

import pytest

from src.rules import RuleEngine, add_banned_phrases, parse_rubric

RECORD = {
    "subject": {"title": "The Quiet Harbour", "attributes": ["slow-burn mystery"]},
    "references": [{"title": "Nine Winters"}],
    "facts": ["Runs 104 minutes."],
}


class TestTagParsing:
    def test_apostrophes_survive_the_parser(self):
        """REGRESSION, and the most valuable test in the file.

        `_QUOTED_RE` originally accepted the single quote as a delimiter, so
        `[banned: "you'll love it", "a must-watch"]` parsed to the fragments
        `you`, `,` and `,`. A one-character banned phrase matches the comma in
        nearly every sentence, so the engine rejected almost the whole corpus.

        Nothing raised. No tag was reported as malformed. Recall collapsed to
        0.29 while specificity stayed plausible, so it read as a judge that was
        merely too strict - a tuning problem, not a parser problem. That is the
        entire reason this test exists.
        """
        rules = parse_rubric('- text [banned: "you\'ll love it", "a must-watch"]')
        assert [r.args for r in rules] == [("you'll love it", "a must-watch")]

    def test_short_fragments_are_dropped_not_matched(self):
        """Defence in depth for the same class of bug. A one-character phrase
        rule is never what anyone meant, and it rejects everything silently."""
        rules = parse_rubric('- text [banned: "a", "ok", "genuinely awful"]')
        assert rules[0].args == ("genuinely awful",)

    def test_unknown_tags_are_skipped_not_fatal(self):
        """The reflector writes rubrics. An invented tag should cost one clause,
        not kill a tuning run at iteration four."""
        rules = parse_rubric("- text [vibes: good] [min_words: 10]")
        assert [r.kind for r in rules] == ["min_words"]

    def test_numeric_args_are_parsed_as_numbers(self):
        assert parse_rubric("[max_words: 40]")[0].args == (40,)

    def test_malformed_number_degrades_to_no_argument(self):
        assert parse_rubric("[max_words: forty]")[0].args == ()


class TestEvaluation:
    def test_clean_artefact_passes(self):
        rubric = "[grounded]\n[require: subject.title]\n[min_words: 5]"
        verdict = RuleEngine(rubric).evaluate(
            "The Quiet Harbour is a slow-burn mystery, much like Nine Winters.", RECORD
        )
        assert verdict.label == "PASS"

    def test_invented_number_is_caught(self):
        verdict = RuleEngine("[grounded]").evaluate(
            "The Quiet Harbour was rated 4.8 by viewers.", RECORD
        )
        assert verdict.label == "FAIL"
        assert verdict.failure_mode == "unsupported_claim"

    def test_real_number_on_the_wrong_noun_is_NOT_caught(self):
        """The documented ceiling of a token-presence check, pinned as a test.

        104 is the runtime. Calling it a rating is false, and every token in the
        sentence appears in the record, so the rule engine passes it. This is
        not a bug to fix here - it is the gap a model judge is for, and pinning
        it keeps anyone from quietly "fixing" the baseline into looking better
        than a substring matcher can be.
        """
        verdict = RuleEngine("[grounded]").evaluate(
            "The Quiet Harbour is rated 104 by viewers.", RECORD
        )
        assert verdict.label == "PASS"

    def test_sentence_initial_capitals_are_not_treated_as_claims(self):
        verdict = RuleEngine("[grounded]").evaluate(
            "Nine Winters was good. Similar mysteries suit this viewer.", RECORD
        )
        assert verdict.label == "PASS"

    def test_missing_subject_is_caught(self):
        verdict = RuleEngine("[require: subject.title]").evaluate(
            "This one is a slow-burn mystery worth your evening.", RECORD
        )
        assert verdict.failure_mode == "missing_subject"

    def test_safety_kinds_report_distinct_failure_modes(self):
        """Not cosmetic. Reasoning agreement compares the judge's mode against
        the human's, so collapsing spoiler and sensitive_framing into one
        `unsafe_content` bucket scores every safety catch as a reason mismatch.
        """
        spoiler = RuleEngine('[spoiler: "the twist is that"]').evaluate(
            "The Quiet Harbour, where the twist is that she did it.", RECORD
        )
        sensitive = RuleEngine('[sensitive: "harrowing"]').evaluate(
            "The Quiet Harbour is harrowing and unmissable.", RECORD
        )
        assert spoiler.failure_mode == "spoiler"
        assert sensitive.failure_mode == "sensitive_framing"

    def test_severity_order_puts_safety_first(self):
        """When several clauses fire, the reported mode is the most serious one.
        A human rater does the same, and reasoning agreement is only meaningful
        if both parties resolve a multi-way failure the same way."""
        rubric = '[spoiler: "the twist is that"]\n[banned: "you\'ll love it"]'
        verdict = RuleEngine(rubric).evaluate(
            "You'll love it, and the twist is that she did it.", RECORD
        )
        assert verdict.failure_mode == "spoiler"


class TestRubricEditing:
    def test_phrases_are_appended_to_an_existing_banned_clause(self):
        before = '- filler [banned: "highly rated"]'
        after = add_banned_phrases(before, ["perfect for anyone"])
        assert RuleEngine(after).banned_phrases() == {"highly rated", "perfect for anyone"}

    def test_a_bullet_is_created_when_none_exists(self):
        after = add_banned_phrases("- some rubric text", ["an instant classic"])
        assert "an instant classic" in RuleEngine(after).banned_phrases()

    def test_duplicates_are_not_re_added(self):
        before = '[banned: "highly rated"]'
        assert add_banned_phrases(before, ["highly rated"]) == before

    def test_empty_input_is_a_no_op(self):
        before = '[banned: "highly rated"]'
        assert add_banned_phrases(before, []) == before


class TestRubricsInTheShippedDomains:
    def test_every_seed_rubric_parses_to_at_least_one_rule(self, domain):
        """A rubric whose tags do not parse silently becomes a judge with no
        checks, which passes everything and looks merely lenient."""
        for criterion in domain.criteria:
            rules = parse_rubric(domain.seed_rubric(criterion.id))
            assert rules, f"criterion {criterion.id!r} has no machine-readable rule"

    def test_no_phrase_rule_is_dangerously_short(self, domain):
        for criterion in domain.criteria:
            for rule in parse_rubric(domain.seed_rubric(criterion.id)):
                for arg in rule.args:
                    if isinstance(arg, str):
                        assert len(arg) >= 3, (
                            f"{criterion.id}: fragment {arg!r} would match "
                            "nearly every artefact"
                        )

"""Tests for the measurement logic.

These matter more than usual for this project, because the whole thing exists
to make a claim about numbers. A metrics bug here does not crash, it just
tells the learner something false and convincing.
"""

import pytest

from src.metrics import Delta, compare, uncached_input, billable_work
from src.state_db import Session


def sess(**kw) -> Session:
    base = dict(
        id="s",
        source="cli",
        model="anthropic/claude-opus-4",
        title=None,
        started_at=0.0,
        ended_at=10.0,
    )
    base.update(kw)
    return Session(**base)


class TestUncachedInput:
    def test_subtracts_cache_reads(self):
        s = sess(input_tokens=1000, cache_read_tokens=400)
        assert uncached_input(s) == 600

    def test_floors_at_zero_when_cache_reads_sit_outside_input(self):
        # Some OpenAI-compatible gateways report cache reads separately from
        # input_tokens. Subtracting then goes negative, and a negative token
        # count is a worse lie than an undercount.
        s = sess(input_tokens=1000, cache_read_tokens=1500)
        assert uncached_input(s) == 0

    def test_billable_work_adds_output(self):
        s = sess(input_tokens=1000, cache_read_tokens=400, output_tokens=250)
        assert billable_work(s) == 850


class TestDelta:
    def test_pct_change_is_none_on_zero_baseline(self):
        # Guards the single most misleading thing this tool could print:
        # "infinitely better" because the baseline happened to be zero.
        d = Delta("cache_read_tokens", 0, 31000, lower_is_better=False)
        assert d.pct_change is None

    def test_lower_is_better_direction(self):
        assert Delta("tool_calls", 11, 4).improved is True
        assert Delta("tool_calls", 4, 11).improved is False

    def test_higher_is_better_direction(self):
        assert Delta("cache_read_tokens", 10, 20, lower_is_better=False).improved
        assert not Delta("cache_read_tokens", 20, 10, lower_is_better=False).improved

    def test_no_change_is_not_an_improvement(self):
        assert Delta("tool_calls", 7, 7).improved is False


class TestVerdict:
    def test_fewer_tool_calls_credits_the_skill(self):
        c = compare(sess(tool_call_count=11), sess(tool_call_count=4))
        assert c.verdict == "skill shortened the procedure"

    def test_same_tool_calls_but_cheaper_blames_caching(self):
        # The confound this project is built to expose. Identical procedure,
        # smaller bill, because the prompt prefix was cached.
        base = sess(tool_call_count=5, input_tokens=40000, output_tokens=2000)
        cand = sess(
            tool_call_count=5,
            input_tokens=40000,
            cache_read_tokens=30000,
            output_tokens=2000,
        )
        c = compare(base, cand)
        assert "likely caching" in c.verdict

    def test_more_tool_calls_says_the_skill_did_not_match(self):
        c = compare(sess(tool_call_count=4), sess(tool_call_count=9))
        assert "did not match" in c.verdict

    def test_identical_sessions_report_no_change(self):
        c = compare(sess(tool_call_count=5, input_tokens=10, output_tokens=5),
                    sess(tool_call_count=5, input_tokens=10, output_tokens=5))
        assert c.verdict == "no change"


class TestWarnings:
    def test_flags_different_models(self):
        c = compare(
            sess(model="anthropic/claude-opus-4", tool_call_count=9),
            sess(model="openai/gpt-4o", tool_call_count=3),
        )
        assert any("Different models" in w for w in c.warnings)

    def test_flags_continuation_sessions(self):
        # A continuation inherits context the baseline never had, so the
        # comparison is not measuring the skill.
        c = compare(
            sess(tool_call_count=9),
            sess(tool_call_count=3, parent_session_id="sess_cold_0001"),
        )
        assert any("parent_session_id" in w for w in c.warnings)

    def test_flags_cache_asymmetry(self):
        c = compare(
            sess(tool_call_count=9, cache_read_tokens=0),
            sess(tool_call_count=3, cache_read_tokens=31000),
        )
        assert any("prompt caching" in w for w in c.warnings)

    def test_flags_tool_call_recount_mismatch(self):
        c = compare(
            sess(tool_call_count=11),
            sess(tool_call_count=4),
            baseline_recount=11,
            candidate_recount=2,
        )
        assert any("ended abnormally" in w for w in c.warnings)

    def test_flags_a_task_with_no_tool_calls_at_all(self):
        c = compare(sess(tool_call_count=0), sess(tool_call_count=0))
        assert any("no procedure to shorten" in w for w in c.warnings)

    def test_clean_comparison_has_no_spurious_warnings(self):
        # Distinct ids: two runs of the same task are two different sessions,
        # and comparing one session with itself is its own warning.
        c = compare(
            sess(id="sess_cold", tool_call_count=11, cache_read_tokens=100),
            sess(id="sess_warm", tool_call_count=4, cache_read_tokens=200),
            baseline_recount=11,
            candidate_recount=4,
        )
        assert c.warnings == []


class TestSerialisation:
    def test_to_dict_round_trips_through_json(self):
        import json

        c = compare(sess(tool_call_count=11), sess(tool_call_count=4))
        payload = json.loads(json.dumps(c.to_dict()))
        assert payload["verdict"] == "skill shortened the procedure"
        names = [d["name"] for d in payload["deltas"]]
        assert "tool_calls" in names and "billable_work" in names

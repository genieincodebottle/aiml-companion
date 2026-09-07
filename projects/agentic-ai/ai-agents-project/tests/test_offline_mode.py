"""Tests for the offline stand-in and the silent-success bug it exposed.

None of these need an API key or a network. Run with:

    python run.py test

Every agent in this project catches its own exceptions and returns a degraded
state, so a broken run still completes and still prints something. That is why
these tests assert on the pipeline trace and the produced artefacts rather than
on "did it finish".
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import (  # noqa: E402
    OfflineFixtureError,
    OfflineLLM,
    get_llm,
    is_offline,
)
from src.models.state import AnalystOutput, PlannerOutput, ReviewOutput  # noqa: E402
from src.token_usage import structured_call, token_count  # noqa: E402
from src.tools.search_offline import offline_search  # noqa: E402


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def test_no_key_selects_the_offline_stand_in(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCH_OFFLINE", raising=False)
    assert is_offline()
    assert isinstance(get_llm(), OfflineLLM)


def test_offline_can_be_forced_while_holding_a_key(monkeypatch):
    """So you can exercise the graph without spending tokens."""
    monkeypatch.setenv("GOOGLE_API_KEY", "real-looking-key")
    monkeypatch.setenv("RESEARCH_OFFLINE", "1")
    assert is_offline()


def test_a_key_without_the_flag_selects_the_real_client(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "real-looking-key")
    monkeypatch.delenv("RESEARCH_OFFLINE", raising=False)
    assert not is_offline()


# ---------------------------------------------------------------------------
# Fixtures must satisfy the REAL schemas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("schema,prompt", [
    (PlannerOutput, "Break this research query into 1-3 focused sub-topics.\n\nQuery: batteries\n"),
    (AnalystOutput, "Extract claims with evidence.\n\nQuery: batteries\n"),
    (ReviewOutput, "You are a research report reviewer. Score this report 1-10.\n\nQuery: batteries\n"),
])
def test_structured_fixtures_parse_through_the_real_schema(schema, prompt):
    """`structured_call` raises on a parsing error, and every agent catches it
    and returns a degraded state. A fixture that no longer matches its model
    would therefore surface as a quietly empty report, not an error."""
    llm = OfflineLLM()
    parsed, tokens = structured_call(
        llm.with_structured_output(schema, include_raw=True), prompt
    )
    assert isinstance(parsed, schema)
    assert tokens > 0, "token_count() saw no usage metadata; the budget guard is blind"


def test_planner_respects_the_configured_sub_topic_limit():
    llm = OfflineLLM()
    parsed, _ = structured_call(
        llm.with_structured_output(PlannerOutput, include_raw=True),
        "Break this research query into 1-2 focused sub-topics.\n\nQuery: batteries\n",
    )
    assert len(parsed.sub_topics) <= 2


def test_analyst_fixture_actually_produces_a_conflict():
    """A conflicts list that is always empty means the cross-referencing half of
    this pipeline never runs and the reader never sees it work."""
    llm = OfflineLLM()
    parsed, _ = structured_call(
        llm.with_structured_output(AnalystOutput, include_raw=True), "Query: batteries\n"
    )
    assert len(parsed.claims) >= 4
    assert parsed.conflicts, "the fixtures must disagree or the analyst has nothing to find"
    confidences = {c.confidence for c in parsed.claims}
    assert "low" in confidences, "one claim should be weakly supported"


def test_reviewer_fails_the_first_draft_and_passes_the_revision():
    """The revision loop is a node and a conditional edge. A reviewer that
    always passes means neither ever executes in a demo run.

    The markers here are the REAL prompt strings. The first version of this
    matched "revision" and "reviewer feedback", neither of which the writer's
    prompts contain, so the writer never revised and the demo published a
    report the reviewer had scored 5 out of 10.
    """
    llm = OfflineLLM()
    chain = llm.with_structured_output(ReviewOutput, include_raw=True)

    first, _ = structured_call(chain, "Score this report.\n\nReport to review:\n# Draft\n")
    assert first.passed is False
    assert first.score < 7
    assert first.issues

    revised, _ = structured_call(
        chain,
        "Score this report.\n\nReport to review:\n# Draft\n"
        "> Revised after review: the unverified claim is now labelled.\n",
    )
    assert revised.passed is True
    assert revised.score >= 7


def test_writer_marks_a_revised_draft():
    llm = OfflineLLM()
    initial = llm.invoke("You are a technical writer. Write a structured research report.")
    assert "Revised after review" not in initial.content

    revised = llm.invoke(
        "You are a technical writer revising a research report.\n"
        "Reviewer issues to fix:\n- something\n"
    )
    assert "Revised after review" in revised.content


# ---------------------------------------------------------------------------
# Failing loudly is the feature
# ---------------------------------------------------------------------------

def test_unknown_schema_raises_rather_than_inventing_a_reply():
    from pydantic import BaseModel

    class SomethingNew(BaseModel):
        value: str

    with pytest.raises(OfflineFixtureError) as excinfo:
        OfflineLLM().with_structured_output(SomethingNew, include_raw=True).invoke("hi")
    assert "No offline fixture" in str(excinfo.value)


def test_unknown_text_prompt_raises():
    with pytest.raises(OfflineFixtureError):
        OfflineLLM().invoke("Summarise the weather in Bengaluru.")


def test_responses_carry_usage_metadata():
    """`token_count` returns 0 and warns that the budget guardrail is blind when
    usage metadata is missing, so a stand-in without it would make every
    offline run look free."""
    msg = OfflineLLM().invoke("Write a structured research report.")
    assert token_count(msg) > 0


# ---------------------------------------------------------------------------
# Offline search
# ---------------------------------------------------------------------------

def test_offline_search_returns_the_shape_web_search_returns():
    results = offline_search("battery chemistry", max_results=5)
    assert len(results) == 5
    for r in results:
        assert set(r) == {"title", "url", "snippet", "date", "tool"}
        assert r["tool"] == "offline"


def test_offline_search_is_deterministic_and_query_dependent():
    a = offline_search("battery chemistry")
    b = offline_search("battery chemistry")
    c = offline_search("something entirely different")
    assert a == b, "two runs of the demo must be comparable"
    assert a != c, "every sub-topic returning identical sources defeats the fan-out"


def test_offline_search_includes_a_weak_source():
    """The quality gate needs something to actually reject."""
    joined = " ".join(s["snippet"] for s in offline_search("batteries"))
    assert "no sources cited" in joined.lower() or "opinion" in joined.lower()


def test_web_search_falls_back_to_offline_without_a_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from src.tools.search import web_search

    results = web_search("batteries")
    assert results, "no key used to mean an empty source list and an empty report"
    assert results[0]["tool"] == "offline"


# ---------------------------------------------------------------------------
# End to end, offline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def offline_result():
    os.environ["RESEARCH_OFFLINE"] = "1"
    from src.agents.graph import build_graph

    return build_graph().invoke({"query": "What are the latest trends in AI agents?"})


def test_pipeline_produces_a_report_without_any_key(offline_result):
    assert offline_result.get("final_report"), (
        "the whole point of offline mode: a reader with no key can watch it work"
    )


def test_no_agent_recorded_an_error(offline_result):
    """Every agent catches its own exceptions, so completion proves nothing."""
    assert not offline_result.get("errors"), offline_result.get("errors")


def test_the_fan_out_actually_ran_in_parallel(offline_result):
    trace = [t.get("agent") for t in offline_result["pipeline_trace"]]
    assert trace.count("researcher") >= 2, f"expected several researchers, got {trace}"
    assert len(offline_result["sources"]) >= 10


def test_the_graph_went_backwards(offline_result):
    """The writer and reviewer both run twice: the reviewer rejected draft one."""
    trace = [t.get("agent") for t in offline_result["pipeline_trace"]]
    assert trace.count("writer") >= 2, f"the revision loop did not fire: {trace}"
    assert trace.count("reviewer") >= 2


def test_the_published_report_passed_review(offline_result):
    """A report published on a failing score is the demo contradicting itself."""
    review = offline_result.get("review") or {}
    assert review.get("passed") is True
    assert review.get("score", 0) >= 7


def test_the_analyst_found_the_planted_disagreement(offline_result):
    assert offline_result.get("conflicts"), "the fixtures disagree; the analyst must notice"


def test_tokens_were_actually_counted(offline_result):
    assert offline_result.get("token_count", 0) > 0, (
        "a zero here means the budget guardrail is blind for the whole run"
    )


# ---------------------------------------------------------------------------
# The silent success
# ---------------------------------------------------------------------------

def test_entry_point_exits_non_zero_when_no_report(monkeypatch, capsys):
    """`python -m src.agents` used to print "No report generated." and return 0.

    Every agent catches its own exceptions, so a run in which all eight nodes
    failed still reached the final print and still told the shell it succeeded.
    With no keys that was the ACTUAL behaviour: five pydantic errors, an empty
    report, and exit code 0.
    """
    from src.agents import __main__ as entry

    class _EmptyGraph:
        def invoke(self, _state):
            return {"errors": ["Analyst error: no API key"]}

    monkeypatch.setattr(entry, "build_graph", lambda: _EmptyGraph())
    monkeypatch.setattr(sys, "argv", ["prog", "anything"])

    assert entry.main() == 1
    assert "No report generated" in capsys.readouterr().err


def test_entry_point_exits_zero_on_a_real_report(monkeypatch, capsys):
    from src.agents import __main__ as entry

    class _GoodGraph:
        def invoke(self, _state):
            return {"final_report": "# Report\n\nSomething useful."}

    monkeypatch.setattr(entry, "build_graph", lambda: _GoodGraph())
    monkeypatch.setattr(sys, "argv", ["prog", "anything"])

    assert entry.main() == 0
    assert "Something useful" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Placeholder keys are not keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("placeholder", [
    "your-tavily-api-key-here",
    "your-google-api-key-here",
    "your_api_key",
    "changeme",
    "xxx",
    "",
    "   ",
    "short",
])
def test_placeholder_keys_count_as_absent(monkeypatch, placeholder):
    """`graph.py` calls load_dotenv(), and .env.example fills every key with a
    placeholder. A placeholder is a non-empty string, so the original
    `not os.getenv(...)` test said a key was present, the offline fallback never
    fired, the real Tavily client raised, the exception was caught, and
    web_search returned []. Every researcher came back empty and the pipeline
    still printed a report.
    """
    from src.llm import has_real_key

    monkeypatch.setenv("TAVILY_API_KEY", placeholder)
    assert not has_real_key("TAVILY_API_KEY"), f"{placeholder!r} was accepted as a key"


def test_a_realistic_key_is_accepted(monkeypatch):
    from src.llm import has_real_key

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-a1b2c3d4e5f6g7h8i9j0")
    assert has_real_key("TAVILY_API_KEY")


def test_search_falls_back_when_the_key_is_a_placeholder(monkeypatch):
    """The measured symptom: tools=['tavily'] chosen, zero sources returned."""
    monkeypatch.setenv("TAVILY_API_KEY", "your-tavily-api-key-here")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyRealLookingKeyValue123")
    from src.tools.search import web_search

    results = web_search("anything at all")
    assert results, "a placeholder key produced an empty source list again"
    assert results[0]["tool"] == "offline"


def test_offline_mode_not_triggered_by_a_real_google_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSyRealLookingKeyValue123")
    monkeypatch.delenv("RESEARCH_OFFLINE", raising=False)
    from src.llm import is_offline

    assert not is_offline()

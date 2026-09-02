"""
Regression tests for token accounting.

The bug these lock down
-----------------------
Every agent used to report an invented token count:

    planner 500 | researcher 300 | analyst 1200 | reviewer 800

and the two agents that looked like they measured something read the OpenAI
response shape, which Gemini never populates:

    response.response_metadata.get("token_usage", {}).get("total_tokens", 1500)

Measured against gemini-3.6-flash: `response_metadata` holds only
finish_reason, model_name, safety_ratings, model_provider -- no `token_usage`
key -- so the writer recorded 1500 tokens for a call that actually cost 462,
on every call, forever.

The consequence was not cosmetic. The whole pipeline reported the same total
for every query regardless of length, the 50,000-token budget guardrail was
being enforced against a constant, and the UI's per-step token column was
decoration. Real usage lives on `response.usage_metadata`.

The old suite passed with all six constants in place, which is why these exist:
every assertion below fails against the pre-fix code.

Run: pytest tests/test_token_accounting.py -v
"""
import pytest

from src.token_usage import token_count, structured_call


class _Msg:
    """Stand-in for an AIMessage carrying LangChain's standard usage field."""

    def __init__(self, total=None, text="draft"):
        self.text = text
        self.content = text
        # Deliberately the shape the buggy code looked for -- and never finds.
        self.response_metadata = {"finish_reason": "STOP", "model_name": "x"}
        self.usage_metadata = (
            None if total is None
            else {"input_tokens": 10, "output_tokens": total - 10, "total_tokens": total}
        )


def test_token_count_reads_usage_metadata_not_response_metadata():
    assert token_count(_Msg(total=462)) == 462


def test_token_count_sums_halves_when_total_is_absent():
    msg = _Msg(total=462)
    del msg.usage_metadata["total_tokens"]
    assert token_count(msg) == 462


def test_missing_usage_reports_zero_not_a_plausible_guess():
    """A zero is visibly wrong; a hardcoded 1500 looks like a measurement."""
    assert token_count(_Msg(total=None)) == 0


class _Structured:
    """Fake `with_structured_output(..., include_raw=True)` chain."""

    def __init__(self, parsed, total=525, error=None):
        self._out = {"parsed": parsed, "raw": _Msg(total=total), "parsing_error": error}

    def invoke(self, prompt):
        return self._out


def test_structured_call_returns_both_the_parsed_model_and_real_tokens():
    parsed, tokens = structured_call(_Structured("PARSED", total=525), "prompt")
    assert parsed == "PARSED"
    assert tokens == 525


def test_structured_call_raises_rather_than_returning_an_unparsed_none():
    chain = _Structured(None, error=ValueError("bad json"))
    with pytest.raises(ValueError):
        structured_call(chain, "prompt")


@pytest.mark.parametrize(
    "module,attr,old_constant",
    [("src.agents.planner", "planner", 500),
     ("src.agents.analyst", "analyst", 1200),
     ("src.agents.reviewer", "reviewer", 800),
     ("src.agents.writer", "writer", 1500),
     ("src.agents.synthesizer", "synthesizer", 800)],
)
def test_no_agent_hardcodes_a_token_count(module, attr, old_constant):
    """The literal must not reappear as a token value in the source."""
    import importlib, inspect
    src = inspect.getsource(importlib.import_module(module))
    assert f'"token_count": {old_constant}' not in src
    assert f'"tokens": {old_constant}' not in src
    assert f'"total_tokens", {old_constant}' not in src


def test_researcher_reports_zero_tokens_because_it_never_calls_the_llm(monkeypatch):
    """It searches; it does not generate. It used to claim 300 tokens a call."""
    import src.agents.researcher as r
    monkeypatch.setattr(r, "select_tools", lambda q: ["wikipedia"])
    monkeypatch.setattr(
        r, "wiki_search",
        lambda q, max_results=3: [{"url": "https://en.wikipedia.org/wiki/A", "title": "A"}])

    out = r.researcher({"query": "test topic", "token_count": 0})
    assert out["token_count"] == 0
    assert out["pipeline_trace"][0]["tokens"] == 0
    assert len(out["sources"]) == 1


def test_the_pipeline_total_now_varies_with_actual_usage():
    """Pre-fix, a 3-subtopic run summed to exactly 5,700 tokens for ANY query:
    500 + 3*300 + 1200 + 800 + 1500 + 800. That constant total is what made the
    budget guardrail meaningless."""
    cheap = sum(token_count(_Msg(total=t)) for t in (120, 200, 90))
    costly = sum(token_count(_Msg(total=t)) for t in (4000, 9000, 3000))
    assert cheap != costly
    assert cheap < costly

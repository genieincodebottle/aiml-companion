"""
Regression tests for search provenance and failure handling.

The bug
-------
`_tavily_search` ended with:

    except (URLError, json.JSONDecodeError) as e:
        logger.error("Tavily search failed: %s", e)
        return _mock_search(query, max_results)

So a timeout, a revoked key, or a 429 did not fail -- it returned placeholder
text reading "This is a mock search result about '<query>'. In production, this
would contain real web content", which went straight into the research prompt
under the instruction "Cite sources using [Source N]". The model summarised it,
and the agent returned a confident report whose `sources` list contained
https://example.com/result-1 with nothing marking it as fake. A run during an
outage was indistinguishable from a real one.

Two changes, both tested here:
  * a configured-but-failing backend raises `SearchUnavailable` instead of
    fabricating hits, and the research agent reports the failure;
  * every result carries `provenance`, and the text handed to the model labels
    placeholders explicitly -- the model only sees the string, so the warning
    has to be in the string.

Run: pytest tests/test_search_provenance.py -v
"""
import json
from unittest.mock import patch
from urllib.error import URLError

import pytest

from tools.web_search import (
    SearchUnavailable, _mock_search, _tavily_search, format_search_results,
)
from agents import research_agent


# === A configured backend must fail loudly, not quietly invent sources ===

@pytest.mark.parametrize("error", [
    URLError("timed out"),
    json.JSONDecodeError("bad", "", 0),
])
def test_search_failure_raises_instead_of_returning_placeholders(error):
    with patch("tools.web_search.urlopen", side_effect=error):
        with pytest.raises(SearchUnavailable):
            _tavily_search("quantum computing", 5)


def test_research_agent_reports_the_outage_rather_than_researching_fiction():
    with patch("agents.research_agent.web_search",
               side_effect=SearchUnavailable("429 rate limited")):
        with patch("agents.research_agent.call_llm") as llm:
            out = research_agent.run({"query": "quantum computing"})

    assert out["search_mode"] == "unavailable"
    assert out["sources"] == []
    assert "could not be completed" in out["result"]
    # The decisive assertion: no LLM call at all. Answering from model memory
    # would look exactly like a researched answer to the caller.
    llm.assert_not_called()


# === Placeholders must be traceable end to end ===

def test_every_placeholder_result_is_tagged():
    results = _mock_search("anything", 3)
    assert results and all(r["provenance"] == "mock" for r in results)


def test_real_results_are_tagged_as_such():
    payload = json.dumps({"results": [
        {"title": "T", "url": "https://arxiv.org/abs/1", "content": "C"}]}).encode()

    class _Resp:
        def read(self): return payload
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch("tools.web_search.urlopen", return_value=_Resp()):
        results = _tavily_search("q", 5)
    assert results[0]["provenance"] == "tavily"


def test_the_prompt_text_warns_the_model_about_placeholders():
    """The model cannot read `provenance`; it reads this string and nothing else."""
    text = format_search_results(_mock_search("q", 3))
    assert "PLACEHOLDER" in text
    assert "Do not cite them as sources" in text


def test_real_results_get_no_warning_banner():
    """Check the label discriminates -- a banner on every run teaches nothing."""
    text = format_search_results(
        [{"title": "T", "url": "https://arxiv.org/abs/1", "content": "C",
          "provenance": "tavily"}])
    assert "PLACEHOLDER" not in text
    assert "WARNING" not in text


def test_provenance_survives_into_the_agents_source_list():
    with patch("agents.research_agent.web_search", return_value=_mock_search("q", 2)):
        with patch("agents.research_agent.call_llm", return_value="## Findings"):
            out = research_agent.run({"query": "q"})

    assert out["search_mode"] == "placeholder"
    assert all(s["provenance"] == "mock" for s in out["sources"])
    # And the caller-facing text says so, not just the metadata.
    assert "not researched" in out["result"]


def test_a_live_run_is_not_labelled_as_placeholder():
    live = [{"title": "T", "url": "https://arxiv.org/abs/1", "content": "C",
             "provenance": "tavily"}]
    with patch("agents.research_agent.web_search", return_value=live):
        with patch("agents.research_agent.call_llm", return_value="## Findings"):
            out = research_agent.run({"query": "q"})

    assert out["search_mode"] == "live"
    assert "not researched" not in out["result"]

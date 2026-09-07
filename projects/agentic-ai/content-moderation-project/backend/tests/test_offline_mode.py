"""Tests for the offline stand-in, the keyword fix, and the enforcement mapping.

None of these need an API key, a server, or a network. Run them with:

    python run.py test

The point of the schema tests is that `parse_llm_response` swallows every
exception and returns None, and each call site then falls back to a bland
default. A wrong offline reply would therefore produce a demo that runs to
completion with every agent silently degraded. These tests are what stop that
being invisible.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MODERATION_OFFLINE", "1")

from src.agents.llm_offline import (  # noqa: E402
    OfflineFixtureError,
    OfflineLLM,
    _extract_content,
)
from src.core.llm_schemas import (  # noqa: E402
    ToxicityAnalysisResponse,
    TopicExtractionResponse,
    create_structured_prompt,
    parse_llm_response,
)
from src.core.models import ContentStatus, DecisionType  # noqa: E402
from src.ml.keyword_detectors import (  # noqa: E402
    keyword_hate_speech_detection,
    keyword_toxicity_detection,
)


CLEAN = "Thanks for writing this up, the section on retries was really helpful."
ABUSIVE = "You are all worthless idiots and I hope every one of you gets hurt."


# ---------------------------------------------------------------------------
# The keyword layer: word boundaries, not substrings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,contains",
    [
        ("Hello everyone, how are you?", "hell inside Hello"),
        ("The class passed the assessment.", "ass inside class"),
        ("I am grateful for your help.", "rat inside grateful"),
        ("Thanks for the offer.", "off inside offer"),
        ("I studied the audience data.", "die inside audience"),
        ("My grandmother passed away last year.", "ass inside passed"),
        ("The rate of change was slow.", "rat inside rate"),
        ("Great classification results on the grass survey.", "ass twice"),
    ],
)
def test_innocuous_sentences_score_zero(text, contains):
    """Each of these fired before the fix, purely on a substring."""
    result = keyword_toxicity_detection(text)
    assert result["toxicity_score"] == 0.0, (
        f"{text!r} scored {result['toxicity_score']} on {contains}; "
        f"categories={result['categories']}"
    )
    assert result["categories"] == ["safe"]


def test_a_greeting_no_longer_outscores_an_insult():
    """The headline symptom, pinned.

    Measured before the fix:
        greeting 0.65, insult 0.40.
    A moderation system that ranks "Hello everyone" as more toxic than
    "You are all worthless idiots" is not doing moderation.
    """
    greeting = keyword_toxicity_detection(
        "Hello everyone, I am grateful for the offer. Let us discuss the strategy at the office."
    )["toxicity_score"]
    insult = keyword_toxicity_detection("You are all worthless idiots.")["toxicity_score"]
    assert greeting == 0.0
    assert insult > greeting


def test_real_toxicity_still_detected():
    """Word boundaries must not blunt actual detection."""
    for text, floor in (
        ("You are all worthless idiots.", 0.3),
        ("I will find you and hurt you badly, you pathetic loser.", 0.6),
    ):
        assert keyword_toxicity_detection(text)["toxicity_score"] >= floor, text


def test_plurals_still_match():
    """A bare word boundary made "idiots" miss while "idiot" hit.

    That silently halved the score of a real insult when boundaries were first
    introduced, which is why the matcher carries an optional plural.
    """
    singular = keyword_toxicity_detection("You are an idiot.")
    plural = keyword_toxicity_detection("You are idiots.")
    assert plural["insult_count"] >= 1
    assert plural["toxicity_score"] == singular["toxicity_score"]


def test_hate_speech_generalisation_needs_a_real_word():
    """'all' was a substring test, so "finally" or "really" triggered it."""
    benign = keyword_hate_speech_detection(
        "Women in engineering is a really important topic and I finally read up on it."
    )
    assert not any(p.startswith("group_generalization") for p in benign["patterns"]), (
        f"benign sentence produced {benign['patterns']}"
    )


# ---------------------------------------------------------------------------
# The offline stand-in: shapes the agents can actually parse
# ---------------------------------------------------------------------------

def test_topic_reply_satisfies_the_real_schema():
    prompt = create_structured_prompt(
        f'Analyze this social media content:\n\nContent: "{CLEAN}"\n',
        TopicExtractionResponse,
    )
    reply = OfflineLLM().invoke(prompt).text
    parsed = parse_llm_response(reply, TopicExtractionResponse)
    assert parsed is not None, "the agent's own parser could not read the reply"
    assert isinstance(parsed.topics, list)


def test_toxicity_reply_satisfies_the_real_schema():
    prompt = create_structured_prompt(
        f'Content: "{ABUSIVE}"\n\nToxicity Analysis (from ML classifier):\n- Score: 0.70\n',
        ToxicityAnalysisResponse,
    )
    reply = OfflineLLM().invoke(prompt).text
    parsed = parse_llm_response(reply, ToxicityAnalysisResponse)
    assert parsed is not None
    assert parsed.toxicity_score == pytest.approx(0.70)
    assert parsed.decision.value in {"approve", "flag", "warn", "remove"}


def test_toxicity_reply_reuses_the_score_already_in_the_prompt():
    """The classifier has already run. A second, disagreeing opinion is worse
    than no opinion, because the report would then contain two numbers."""
    prompt = create_structured_prompt(
        f'Content: "{CLEAN}"\n\n- Score: 0.42\n', ToxicityAnalysisResponse
    )
    parsed = parse_llm_response(OfflineLLM().invoke(prompt).text, ToxicityAnalysisResponse)
    assert parsed.toxicity_score == pytest.approx(0.42)


def test_fast_mode_reply_is_the_bare_json_the_agent_parses():
    prompt = (
        "You are a content moderation AI.\n\n"
        f'**Content to moderate:**\n"{ABUSIVE}"\n\n'
        "Provide ONLY the JSON response, no additional text.\n"
    )
    data = json.loads(OfflineLLM().invoke(prompt).text)
    assert set(data) == {
        "toxicity_score", "policy_violations", "decision", "reason", "confidence"
    }
    assert isinstance(data["policy_violations"], list)


@pytest.mark.parametrize(
    "prompt,allowed",
    [
        (
            'Content: "x y z"\nShould this proceed to toxicity detection? (APPROVE or FLAG)',
            ("APPROVE", "FLAG"),
        ),
        (
            'Content: "x y z"\nCommunity Guidelines:\n1. No hate speech\n',
            ("APPROVE", "WARN", "REMOVE", "SUSPEND_USER", "BAN_USER"),
        ),
        (
            "- Total Violations: 4\n- Risk Score: 0.20\n- Toxicity: 0.20\n"
            "Is this user a repeat offender?",
            ("APPROVE", "WARN", "REMOVE", "SUSPEND_USER", "BAN_USER"),
        ),
    ],
)
def test_keyword_scanned_replies_contain_exactly_one_action_token(prompt, allowed):
    """The agents string-match these tokens in priority order.

    A reply containing two of them silently resolves to whichever the caller
    tests for first, which would make the decision depend on the wording of the
    explanation rather than on the analysis.
    """
    reply = OfflineLLM().invoke(prompt).text
    present = [token for token in allowed if token in reply]
    # SUSPEND_USER contains no other token; REMOVE and WARN are distinct.
    assert len(present) == 1, f"expected one action token, found {present} in:\n{reply}"


def test_appeal_reply_carries_one_verdict():
    prompt = (
        'Original Content: "You are being an idiot about this."\n'
        "- Toxicity Score: 0.20\n"
        "UPHOLD: Original decision was correct\n"
    )
    reply = OfflineLLM().invoke(prompt).text
    present = [v for v in ("UPHOLD", "OVERTURN", "PARTIAL") if v in reply]
    assert len(present) == 1, f"found {present}"


def test_every_enforcement_action_has_user_facing_wording():
    """A missing key here shipped the sentence "Your post was suspend_user"."""
    for action in (
        "approve", "warn", "remove", "suspend_user", "ban_user",
        "approved", "warned", "removed", "user_suspended", "user_banned",
    ):
        prompt = (
            "Generate a user-friendly explanation for this moderation action:\n"
            f"Action: {action}\nContent: \"whatever\"\n"
        )
        reply = OfflineLLM().invoke(prompt).text
        assert action not in reply, f"raw enum {action!r} leaked into user copy: {reply}"
        assert len(reply) > 40


# ---------------------------------------------------------------------------
# Failing loudly is the feature
# ---------------------------------------------------------------------------

def test_unknown_prompt_raises_rather_than_inventing_a_reply():
    """If an eleventh LLM call is added, the demo must break, not degrade.

    `parse_llm_response` returns None on anything it cannot read and the call
    sites fall back to defaults, so a plausible-looking wrong reply would be
    completely invisible.
    """
    with pytest.raises(OfflineFixtureError) as excinfo:
        OfflineLLM().invoke("Summarise the weather in Bengaluru.")
    assert "no handler" in str(excinfo.value)


def test_missing_content_marker_raises():
    """A stand-in that quietly moderated an empty string would approve
    everything and look entirely plausible doing it."""
    with pytest.raises(OfflineFixtureError):
        _extract_content("Community Guidelines:\n1. No hate speech\n")


def test_offline_llm_rejects_unimplemented_langchain_methods():
    with pytest.raises(OfflineFixtureError):
        OfflineLLM().with_structured_output


# ---------------------------------------------------------------------------
# The enforcement status mapping
# ---------------------------------------------------------------------------

def test_suspend_and_ban_are_not_recorded_as_approved():
    """`else: status = APPROVED` caught SUSPEND_USER and BAN_USER.

    A banned user's post carried content_removed=True, user_suspended=True,
    moderation_action="user_suspended" and status="approved" at the same time,
    so the ban was invisible to the moderator queue, the analytics counts and
    the user's own view.
    """
    import inspect

    from src.agents import agents as agents_module

    source = inspect.getsource(agents_module.ContentModerationAgents.action_enforcement_agent)
    marker = "# Set final status"
    assert marker in source
    mapping = source[source.index(marker):]
    assert "DecisionType.SUSPEND_USER.value" in mapping
    assert "DecisionType.BAN_USER.value" in mapping
    # And approval must be explicit rather than the catch-all.
    approved_line = "state[\"status\"] = ContentStatus.APPROVED.value"
    before = mapping[: mapping.index(approved_line)]
    assert "DecisionType.APPROVE.value" in before, (
        "APPROVED is still the fall-through branch"
    )


# ---------------------------------------------------------------------------
# End to end, offline
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph_and_db():
    os.environ["MODERATION_OFFLINE"] = "1"
    os.environ["ENABLE_FAST_MODE"] = "false"
    from src.agents.workflow import create_moderation_workflow
    from src.database.moderation_db import ModerationDatabase

    db = ModerationDatabase()
    return create_moderation_workflow(db, use_checkpointer=False, enable_fast_mode=False), db


def _run(graph, text, **profile_kwargs):
    from src.agents.workflow import process_content
    from src.core.state_factory import build_initial_state, demo_user

    state = build_initial_state(text, user_profile=demo_user(**profile_kwargs))
    return process_content(graph, state)


def test_clean_content_is_approved_without_a_human(graph_and_db):
    graph, _ = graph_and_db
    final = _run(graph, CLEAN)
    assert final["status"] == ContentStatus.APPROVED.value
    assert not final.get("hitl_required")
    assert not final.get("content_removed")


def test_no_agent_silently_errored(graph_and_db):
    """Every agent catches its own exceptions and carries on.

    That means a broken run still produces a final state and a status. The only
    reliable tell is the flag each handler sets on its decision, so this asserts
    on that rather than on the log.
    """
    graph, _ = graph_and_db
    final = _run(graph, CLEAN)
    errored = [
        d.agent_name
        for d in final.get("agent_decisions", [])
        if "processing_error" in (d.flags or [])
    ]
    assert not errored, f"agents failed silently: {errored}"


def test_identical_text_gets_different_outcomes_by_account(graph_and_db):
    """The whole argument for a reputation agent, in one assertion."""
    graph, _ = graph_and_db
    text = "You are being an idiot about this."

    newcomer = _run(graph, text)
    repeat = _run(
        graph, text, total_violations=4, previous_warnings=3, reputation_score=0.25
    )
    verified = _run(graph, text, verified=True, follower_count=50000)

    assert newcomer["status"] == ContentStatus.APPROVED.value
    assert repeat["status"] == ContentStatus.REMOVED.value
    assert repeat["moderation_action"] == "user_suspended"
    assert verified["status"] == ContentStatus.PENDING_HUMAN_REVIEW.value
    assert verified["hitl_required"]


def test_abusive_content_exits_early_to_the_moderator_queue(graph_and_db):
    """Deliberate design, not a bug: agent one flags it and the graph exits.

    `main.py` includes "flagged" in the moderator queue, so this lands in front
    of a human. Five more model calls cannot improve on a decision a human has
    to make anyway.
    """
    graph, _ = graph_and_db
    final = _run(graph, ABUSIVE)
    assert final["status"] == ContentStatus.FLAGGED.value
    assert final["requires_human_review"]
    assert len(final["agent_decisions"]) == 1


def test_context_rescues_toxic_words_in_innocent_use(graph_and_db):
    graph, _ = graph_and_db
    text = (
        "This documentary about the massacre at Wounded Knee is essential "
        "viewing for history students."
    )
    final = _run(graph, text)
    assert final["toxicity_score"] > 0, "the keyword layer should flag this"
    assert final["status"] == ContentStatus.APPROVED.value, "context should rescue it"


def test_the_pipeline_needs_no_api_key(monkeypatch):
    """The original failure: no key meant the graph could not even be built."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MODERATION_OFFLINE", raising=False)
    from src.agents.agents import ContentModerationAgents

    agents = ContentModerationAgents()
    assert agents.offline
    assert isinstance(agents.llm_flash, OfflineLLM)

"""A deterministic stand-in for Gemini, so the pipeline runs with no API key.

Why this exists
---------------
Without a key you could not run this project at all. `ChatGoogleGenerativeAI`
validates the key in its constructor, so `create_moderation_workflow` raised a
pydantic error before a single agent executed. A reader could read the code and
never watch it work.

What this is, and what it is not
--------------------------------
This is NOT a model, and it is not pretending to be one. It is a rule engine
that answers the ten prompts the agents actually send, in the exact shapes the
agents parse. It exists so you can watch the orchestration -- the routing, the
state accumulation, the HITL escalation, the guardrails -- without paying for
tokens or waiting on a network.

Where it needs a toxicity judgement it calls THIS PROJECT'S OWN keyword
detectors rather than inventing a number, so the signal you see in the offline
run is the same signal the real pipeline computes. What the real LLM adds on
top is context: satire, quotation, and educational discussion. The stand-in
approximates that with a handful of explicit markers, and it says so.

Loud failure is the point
-------------------------
`parse_llm_response` catches every exception and returns None, and every call
site then falls back to a bland default. That means a WRONG offline fixture
would produce a demo that appears to work while every agent quietly degrades to
its fallback. So:

- Every JSON payload is validated against the real Pydantic schema before it is
  returned, and a mismatch raises `OfflineFixtureError`.
- An unrecognised prompt raises rather than returning a plausible-looking
  string. If someone adds an eleventh LLM call, the offline demo breaks loudly
  instead of silently answering it wrong.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..core.llm_schemas import (
    ToxicityAnalysisResponse,
    TopicExtractionResponse,
)
from ..ml.keyword_detectors import (
    keyword_hate_speech_detection,
    keyword_toxicity_detection,
)


class OfflineFixtureError(RuntimeError):
    """An offline reply could not be built, or did not match its schema.

    Deliberately fatal. The alternative is a demo that runs to completion with
    every agent silently degraded, which is worse than a crash because it looks
    like success.
    """


# --------------------------------------------------------------------------
# Pulling the content back out of the prompt
# --------------------------------------------------------------------------

_CONTENT_PATTERNS = (
    re.compile(r'\*\*Content to moderate:\*\*\s*\n\s*"(.*?)"', re.S),
    re.compile(r'^\s*Content:\s*"(.*?)"\s*$', re.M | re.S),
    re.compile(r'Content:\s*"(.*?)"', re.S),
)


def _extract_content(prompt: str) -> str:
    """Recover the moderated text from a prompt, or raise.

    The agents interpolate the content into their prompts; they do not hand it
    to the model separately. Rather than guess when the marker is missing, this
    raises, because a stand-in that silently moderates an empty string would
    return "approve" for everything and look entirely plausible.
    """
    for pattern in _CONTENT_PATTERNS:
        found = pattern.search(prompt)
        if found and found.group(1).strip():
            return found.group(1).strip()
    raise OfflineFixtureError(
        "Offline stand-in could not find the content in this prompt. Expected a "
        'Content: "..." marker. First 200 characters:\n' + prompt[:200]
    )


_SCORE_IN_PROMPT = re.compile(
    r"^\s*-\s*(?:Score|Toxicity(?:\s+Score)?):\s*([0-9.]+)", re.M
)


def _signal(prompt: str) -> Dict[str, Any]:
    """The toxicity signal for this prompt, preferring the pipeline's own.

    The toxicity agent has already run the classifier and put the score in the
    prompt. Reusing it keeps the stand-in consistent with the rest of the run
    instead of producing a second, disagreeing opinion.
    """
    text = _extract_content(prompt)
    keyword = keyword_toxicity_detection(text)
    hate = keyword_hate_speech_detection(text)

    in_prompt = _SCORE_IN_PROMPT.search(prompt)
    score = float(in_prompt.group(1)) if in_prompt else keyword["toxicity_score"]

    lowered = text.lower()
    # A crude stand-in for the context judgement a real model makes. Named
    # honestly: these are markers, not comprehension.
    quoted = '"' in text or lowered.startswith("he said") or "quote" in lowered
    educational = any(
        marker in lowered
        for marker in ("documentary", "history", "research", "study", "the term", "definition")
    )
    satire = any(marker in lowered for marker in ("/s", "satire", "sarcasm", "joking"))

    return {
        "text": text,
        "score": score,
        "categories": keyword["categories"],
        "hate": hate,
        "is_quote": quoted,
        "is_educational": educational,
        "is_satire": satire,
        "has_context": quoted or educational or satire,
    }


def _action_for(score: float, has_context: bool) -> str:
    """Map a score to one action token, the same ladder the prompts describe."""
    if has_context and score < 0.8:
        # Context does not make severe content acceptable, but it is exactly
        # what stops a quoted slur or a history lesson being removed.
        return "approve"
    if score >= 0.8:
        return "remove"
    if score >= 0.6:
        return "warn"
    if score >= 0.35:
        return "flag"
    return "approve"


# --------------------------------------------------------------------------
# The reply object
# --------------------------------------------------------------------------


class _OfflineResponse:
    """Mimics the attribute the agents read off a LangChain response.

    The agents use `response.text`. `.content` is provided too, because that is
    what LangChain's own message objects expose and a reader stepping through
    in a debugger will look for it.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.content = text

    def __repr__(self) -> str:
        return f"<OfflineResponse {self.text[:60]!r}>"


def _validated(schema, payload: Dict[str, Any]) -> str:
    """Serialise `payload`, but only after the real schema accepts it."""
    try:
        schema.model_validate(payload)
    except Exception as exc:
        raise OfflineFixtureError(
            f"Offline reply does not satisfy {schema.__name__}: {exc}"
        ) from exc
    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------
# One handler per prompt the agents actually send
# --------------------------------------------------------------------------


def _topic_extraction(prompt: str) -> str:
    sig = _signal(prompt)
    words = [w.strip(".,!?\"'").lower() for w in sig["text"].split()]
    topics = [w for w in words if len(w) > 5][:5]

    sensitive: List[str] = []
    lowered = sig["text"].lower()
    for label, markers in (
        ("politics", ("election", "government", "party", "vote")),
        ("health", ("vaccine", "covid", "medical", "doctor")),
        ("finance", ("crypto", "investment", "stock", "money")),
        ("religion", ("church", "muslim", "jewish", "christian")),
    ):
        if any(m in lowered for m in markers):
            sensitive.append(label)

    return _validated(
        TopicExtractionResponse,
        {
            "topics": topics or ["general"],
            "category": "opinion" if sig["score"] > 0.2 else "personal",
            "entities": [w for w in sig["text"].split() if w[:1].isupper()][:5],
            "sensitive_topics": sensitive,
            "explicit_content": sig["score"] >= 0.6,
            "language": "en",
        },
    )


def _toxicity_analysis(prompt: str) -> str:
    sig = _signal(prompt)
    action = _action_for(sig["score"], sig["has_context"])

    categories = [
        {"category": name, "score": round(sig["score"], 2), "detected": True}
        for name in sig["categories"]
        if name != "safe"
    ]

    if sig["has_context"]:
        notes = (
            "Toxic-looking terms appear inside quotation, satire or educational "
            "framing, which the keyword layer cannot distinguish from use."
        )
    else:
        notes = "No mitigating context detected."

    return _validated(
        ToxicityAnalysisResponse,
        {
            "decision": action,
            "confidence": 0.82 if sig["score"] >= 0.6 or sig["score"] <= 0.1 else 0.64,
            "toxicity_score": round(sig["score"], 2),
            "toxicity_level": (
                "severe" if sig["score"] >= 0.9
                else "high" if sig["score"] >= 0.7
                else "medium" if sig["score"] >= 0.5
                else "low" if sig["score"] >= 0.3
                else "none"
            ),
            "categories": categories,
            "is_satire": sig["is_satire"],
            "is_quote": sig["is_quote"],
            "is_educational": sig["is_educational"],
            "context_notes": notes,
            "reasoning": (
                f"Keyword layer scored {sig['score']:.2f} across "
                f"{', '.join(sig['categories'])}. Recommending {action}."
            ),
        },
    )


def _initial_assessment(prompt: str) -> str:
    sig = _signal(prompt)
    # The caller tests `if "FLAG" in reasoning`, so exactly one of these tokens
    # may appear anywhere in this reply.
    verdict = "FLAG" if sig["score"] >= 0.35 else "APPROVE"
    return (
        f"Initial assessment: {verdict}\n"
        f"Confidence: {0.8 if sig['score'] >= 0.35 else 0.7}\n"
        f"The keyword layer scored this {sig['score']:.2f}. "
        "Proceeding to toxicity detection for a fuller reading."
    )


_POLICY_TOKEN = {
    "approve": "APPROVE",
    "flag": "WARN",
    "warn": "WARN",
    "remove": "REMOVE",
}


def _policy_check(prompt: str) -> str:
    sig = _signal(prompt)
    action = _action_for(sig["score"], sig["has_context"])
    # The caller string-matches BAN_USER, then SUSPEND_USER, then REMOVE, then
    # WARN, in that order. Emitting more than one token would silently pick the
    # first, so this reply carries exactly one.
    token = _POLICY_TOKEN[action]
    if sig["hate"]["detected"] and sig["score"] >= 0.8:
        token = "SUSPEND_USER"

    guidelines = "none" if sig["score"] < 0.35 else ", ".join(sig["categories"])
    return (
        f"Recommended action: {token}\n"
        f"Guidelines engaged: {guidelines}\n"
        f"Severity: {'high' if sig['score'] >= 0.7 else 'low'}\n"
        f"Assessment derived from a keyword score of {sig['score']:.2f}."
    )


def _reputation(prompt: str) -> str:
    """The one prompt that carries no content, only the user's record.

    This handler deliberately does NOT call `_signal`. The reputation prompt is
    about the account, not the post, so it has no `Content: "..."` marker and
    asking for one raised `OfflineFixtureError` mid-pipeline. The agent caught
    that, logged it, and set status to `under_review`, so every single post came
    out "under review" and the demo looked like it had merely been cautious.
    """
    def _num(pattern: str, default: float) -> float:
        found = re.search(pattern, prompt)
        return float(found.group(1)) if found else default

    total_violations = _num(r"-\s*Total Violations:\s*(\d+)", 0)
    recent = _num(r"-\s*Recent Violations \(7 days\):\s*(\d+)", 0)
    risk = _num(r"-\s*Risk Score:\s*([0-9.]+)", 0.0)
    toxicity = _num(r"-\s*Toxicity:\s*([0-9.]+)", 0.0)

    decision = re.search(r"Current Content Decision:\s*(\w+)", prompt)
    recommended = decision.group(1) if decision else "approve"

    # Exactly one action token: the caller string-matches BAN_USER, then
    # SUSPEND_USER, then falls through, so a second token would be ignored.
    if risk > 0.85 and total_violations >= 5:
        token = "BAN_USER"
    elif risk > 0.7 or (total_violations >= 3 and toxicity >= 0.3):
        token = "SUSPEND_USER"
    elif recommended in ("remove", "warn") or toxicity >= 0.35:
        token = "WARN"
    else:
        token = "APPROVE"

    return (
        f"Recommended action: {token}\n"
        f"Repeat offender: {'yes' if total_violations >= 3 else 'no'} "
        f"({int(total_violations)} lifetime, {int(recent)} in the last 7 days)\n"
        f"Risk score {risk:.2f} weighed against this item at {toxicity:.2f}.\n"
        "Justification: account history changes the response to identical text."
    )


def _appeal(prompt: str) -> str:
    sig = _signal(prompt)
    # An appeal on content that only ever tripped the keyword layer, with
    # mitigating context, is exactly the false positive appeals exist to catch.
    if sig["has_context"] and sig["score"] < 0.6:
        verdict = "OVERTURN"
        why = "Mitigating context was present and the original pass missed it."
    elif sig["score"] < 0.75:
        verdict = "PARTIAL"
        why = "The violation is real but the original severity was too high."
    else:
        verdict = "UPHOLD"
        why = "The original decision matches the evidence."
    return f"Appeal decision: {verdict}\nConfidence: 0.78\nReasoning: {why}"


_APPEAL_LINE = " You can appeal this from your profile if you think we have it wrong."

# Keyed by DecisionType value AND by the past-tense moderation_action, because
# the prompt is built from `final_action` and the two vocabularies differ. A
# missing key here produced the user-facing sentence "Your post was
# suspend_user", which is the kind of thing that ships.
_ACTION_WORDING = {
    "approve": "Your post is live. Thanks for keeping the community civil.",
    "approved": "Your post is live. Thanks for keeping the community civil.",
    "warn": "Your post stays up, but part of it breaches our guidelines on "
            "respectful language. Please review them before posting again.",
    "warned": "Your post stays up, but part of it breaches our guidelines on "
              "respectful language. Please review them before posting again.",
    "remove": "Your post has been taken down because it breaches our guidelines "
              "on harassment." + _APPEAL_LINE,
    "removed": "Your post has been taken down because it breaches our guidelines "
               "on harassment." + _APPEAL_LINE,
    "suspend_user": "Your post has been taken down and your account is suspended, "
                    "because this follows earlier warnings." + _APPEAL_LINE,
    "user_suspended": "Your post has been taken down and your account is suspended, "
                      "because this follows earlier warnings." + _APPEAL_LINE,
    "ban_user": "Your account has been permanently closed for repeated serious "
                "breaches of our guidelines." + _APPEAL_LINE,
    "user_banned": "Your account has been permanently closed for repeated serious "
                   "breaches of our guidelines." + _APPEAL_LINE,
    "flag": "Your post is being reviewed by a moderator and is not visible yet. "
            "We aim to decide within 24 hours.",
    "needs_review": "Your post is being reviewed by a moderator and is not visible "
                    "yet. We aim to decide within 24 hours.",
}


def _action_reason(prompt: str) -> str:
    match = re.search(r"Action:\s*(\w+)", prompt)
    action = (match.group(1) if match else "").lower()
    if action not in _ACTION_WORDING:
        # Better a loud gap than a sentence with an enum value in it.
        raise OfflineFixtureError(
            f"No user-facing wording for enforcement action {action!r}. "
            "Add it to _ACTION_WORDING rather than letting the raw value reach "
            "the user."
        )
    return _ACTION_WORDING[action]


def _react_synthesis(prompt: str) -> str:
    agree = re.search(r"Consensus Level:\s*(\w+)", prompt)
    consensus = agree.group(1) if agree else "unknown"
    return (
        f"Agent agreement: {consensus}.\n"
        "Risk level follows the toxicity signal the earlier agents recorded.\n"
        "Conflicting signals, where present, are the reason to involve a human "
        "rather than to average the two readings."
    )


def _react_thought(prompt: str) -> str:
    return (
        "The earlier agents have supplied a toxicity reading and a policy view. "
        "The next step is to check whether they point the same way, because "
        "disagreement between them is the strongest signal that a human should "
        "look at this rather than the pipeline deciding alone."
    )


def _fast_mode(prompt: str) -> str:
    sig = _signal(prompt)
    action = _action_for(sig["score"], sig["has_context"])
    violations = [] if action == "approve" else [c for c in sig["categories"] if c != "safe"]
    return json.dumps(
        {
            "toxicity_score": round(sig["score"], 2),
            "policy_violations": violations,
            "decision": action,
            "reason": (
                "Short content handled on the single-pass path; "
                f"keyword score {sig['score']:.2f}."
            ),
            "confidence": 0.75,
        },
        indent=2,
    )


# Order matters: the two schema-driven prompts are matched on their schema
# title, which is unambiguous, before anything falls through to phrase matching.
_ROUTES = (
    ("'title': 'TopicExtractionResponse'", _topic_extraction),
    ("'title': 'ToxicityAnalysisResponse'", _toxicity_analysis),
    ("Provide ONLY the JSON response", _fast_mode),
    ("Should this proceed to toxicity detection?", _initial_assessment),
    ("Community Guidelines:", _policy_check),
    ("Is this user a repeat offender?", _reputation),
    ("UPHOLD: Original decision was correct", _appeal),
    ("Generate a user-friendly explanation", _action_reason),
    ("THINK Phase - Analyze:", _react_synthesis),
    ("Provide your thought in 2-3 sentences", _react_thought),
)


class OfflineLLM:
    """Drop-in replacement for `ChatGoogleGenerativeAI` in this project.

    Implements the one method the agents call. Records every prompt it saw, so
    tests can assert which agents actually ran rather than trusting the log.
    """

    def __init__(self, name: str = "offline") -> None:
        self.name = name
        self.prompts: List[str] = []

    def invoke(self, prompt: str) -> _OfflineResponse:
        self.prompts.append(prompt)
        for marker, handler in _ROUTES:
            if marker in prompt:
                return _OfflineResponse(handler(prompt))
        raise OfflineFixtureError(
            "Offline stand-in has no handler for this prompt. An LLM call was "
            "added to the agents without a matching offline reply, so the demo "
            "would otherwise have degraded silently. First 300 characters:\n"
            + prompt[:300]
        )

    # LangChain objects are often chained; make the failure obvious rather than
    # letting an AttributeError surface somewhere far from the cause.
    def __getattr__(self, item: str):
        raise OfflineFixtureError(
            f"OfflineLLM does not implement {item!r}. It supports .invoke() only, "
            "which is all this project's agents use."
        )

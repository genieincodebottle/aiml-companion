"""Judges for the Northwind Freight evaluation harness.

Two judges live here:

  HeuristicJudge   WORKING. Deterministic, offline, no API key, no network.
                   It is the default so that `make eval` and CI produce a real
                   number on a fresh clone. It is also deliberately shallow: it
                   scores lexical overlap and mode keywords, which means it can
                   be gamed by a system that echoes the ticket back. Do not ship
                   a customer report backed only by this judge.

  LLMJudge         SKELETON. The contract, the rubric, and the parsing are
                   specified below. The model call itself raises
                   NotImplementedError. You implement it.

Why a heuristic fallback exists at all
--------------------------------------
A customer network is frequently the place where your API key does not work.
An eval harness that cannot run without egress is an eval harness that will not
run during the demo. The heuristic judge is the floor that keeps the gate
honest when the model is unreachable, and it is what CI runs.

Why LLM-as-Judge needs a rubric and not a vibe
----------------------------------------------
"Is this good?" produces a number that drifts every time the model changes. A
judge is only useful if two runs a month apart are comparable, which means the
rubric has to be explicit, the scale has to be small, and the output has to be
parseable. Anchor every point on the scale to an observable property of the
answer, not to a feeling about it.

Contract shared by both judges
------------------------------
    judge.score(case: dict, actual: str) -> Verdict

    case    a golden-set case: id, failure_mode, input, expected, notes
    actual  the string your system produced for case["input"]
    returns Verdict(faithfulness, mode_correct, predicted_mode, rationale)

    faithfulness    float in [0.0, 1.0]
    mode_correct    bool, did the answer land on case["failure_mode"]
    predicted_mode  str, the mode the judge believes the answer chose, or ""
    rationale       short human-readable string, shown on failures only

Judges MUST NOT raise on a malformed `actual`. A system under evaluation is
allowed to produce garbage, and scoring garbage as zero is the whole point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# The five named failure modes for this engagement. Kept here as the single
# source of truth so the golden set, the judge, and the runner cannot drift.
FAILURE_MODES: tuple[str, ...] = (
    "wrong_address",
    "missing_recipient",
    "damaged_goods",
    "customs_hold",
    "site_access",
)

# Rubric anchors for the LLM judge. Keep the scale short. A 1-10 scale invites
# the model to split hairs it cannot actually see, and the extra resolution is
# noise you will mistake for signal.
FAITHFULNESS_RUBRIC: dict[str, str] = {
    "1.0": (
        "Every claim in the answer is supported by the ticket text or by "
        "Northwind policy. No invented tracking numbers, dates, names, or "
        "policy rules."
    ),
    "0.5": (
        "The routing decision is supportable but the answer adds at least one "
        "detail that the ticket does not contain and policy does not imply."
    ),
    "0.0": (
        "The answer asserts something contradicted by the ticket or by policy, "
        "or fabricates an identifier, entitlement, or rule."
    ),
}

JUDGE_SYSTEM_PROMPT = """\
You are grading a triage system for a freight carrier. You are not helping the
system. You are checking it.

You will be given a support ticket, the expected handling, and the system's
actual answer. Return ONLY a JSON object with these keys:

  {"faithfulness": <0.0|0.5|1.0>,
   "predicted_mode": "<one of the five failure modes, or empty string>",
   "rationale": "<one sentence>"}

Faithfulness rubric:
{rubric}

Valid failure modes: {modes}

Grade only what is written. Do not reward an answer for being longer, more
confident, or better formatted. If the answer is empty or unparseable, return
faithfulness 0.0 and an empty predicted_mode.
"""


@dataclass
class Verdict:
    """One judge's decision about one case."""

    faithfulness: float
    mode_correct: bool
    predicted_mode: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        # Clamp rather than raise. A judge that crashes the harness is worse
        # than a judge that returns a bad number you can see in the table.
        self.faithfulness = max(0.0, min(1.0, float(self.faithfulness)))


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 2}


# ---------------------------------------------------------------------------
# Working, deterministic, offline
# ---------------------------------------------------------------------------

# Keyword evidence per failure mode. This is not a model. It is a tripwire, and
# it exists so the gate produces a number with no API key. Note how thin it is:
# "seal number does not match the manifest" carries no word in the damaged_goods
# list, which is precisely the kind of case a real judge should catch and this
# one will not. That gap is the argument for replacing it.
MODE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "wrong_address": (
        "address", "postcode", "postal", "street", "unit", "geocode",
        "misdeliver", "misdelivery", "locality", "correction",
    ),
    "missing_recipient": (
        "recipient", "consignee", "nobody", "signature", "attempt", "contact",
        "unavailable", "neighbour", "neighbor", "reception",
    ),
    "damaged_goods": (
        "damage", "damaged", "crushed", "broken", "torn", "wet", "claim",
        "claims", "punctured", "seal",
    ),
    "customs_hold": (
        "customs", "duty", "duties", "import", "export", "border", "tariff",
        "clearance", "commodity", "restricted",
    ),
    "site_access": (
        "access", "gate", "code", "dock", "escort", "permit", "security",
        "hours", "barrier", "site",
    ),
}


class HeuristicJudge:
    """Deterministic keyword and overlap judge. No network, no key, no drift.

    Scoring, stated plainly so nobody mistakes it for something cleverer:

      predicted_mode  the mode whose keyword list has the most hits in `actual`,
                      ties broken by the declared mode order, no hits -> "".
      faithfulness    token overlap between `actual` and `expected`, bucketed
                      onto the same 0.0 / 0.5 / 1.0 scale the LLM judge uses so
                      the two are at least comparable in shape.

    Known and intended weakness: an answer that parrots the ticket back scores
    respectably. Treat every HeuristicJudge number as a lower bound on how much
    work is left, never as evidence the system is ready.
    """

    name = "heuristic"
    requires_api_key = False

    def __init__(self, high: float = 0.45, low: float = 0.20) -> None:
        # Overlap fractions at which faithfulness buckets up. These are two more
        # numbers you are choosing. They are tuned so a blank answer scores 0.0
        # and a substantive answer is not punished for wording.
        self.high = high
        self.low = low

    def score(self, case: dict, actual: str) -> Verdict:
        actual = actual or ""
        expected = str(case.get("expected", ""))
        want_mode = str(case.get("failure_mode", ""))

        predicted = self._predict_mode(actual)

        exp_tokens = _tokens(expected)
        act_tokens = _tokens(actual)
        if not exp_tokens or not act_tokens:
            overlap = 0.0
        else:
            overlap = len(exp_tokens & act_tokens) / len(exp_tokens)

        if overlap >= self.high:
            faithfulness = 1.0
        elif overlap >= self.low:
            faithfulness = 0.5
        else:
            faithfulness = 0.0

        return Verdict(
            faithfulness=faithfulness,
            mode_correct=(predicted == want_mode and predicted != ""),
            predicted_mode=predicted,
            rationale=(
                f"heuristic: keyword mode={predicted or 'none'}, "
                f"expected-token overlap={overlap:.2f}"
            ),
        )

    @staticmethod
    def _predict_mode(actual: str) -> str:
        text = (actual or "").lower()
        best, best_hits = "", 0
        for mode in FAILURE_MODES:
            hits = sum(1 for kw in MODE_KEYWORDS[mode] if kw in text)
            if hits > best_hits:
                best, best_hits = mode, hits
        return best


# ---------------------------------------------------------------------------
# Skeleton - you implement this
# ---------------------------------------------------------------------------


@dataclass
class LLMJudge:
    """LLM-as-Judge. SKELETON. The model call is yours to write.

    Everything except the call is specified: the prompt, the rubric, the JSON
    contract, the parse, and the failure behaviour. Implement `_call_model` and
    this class becomes live without touching the runner.

    Things to decide before you implement, because they are all defensible
    either way and all worth an ADR:

    - Which model, and can it run inside Northwind's boundary? A judge that
      requires public egress cannot be used on the data it needs to judge.
    - Temperature 0 and a pinned model version, or you cannot compare this
      month's number to last month's.
    - Do you judge with the same model family that generates? Self-preference
      is real. If you do, say so next to the number.
    - What happens on a rate limit or a timeout mid-run? A judge that silently
      scores 0.0 on a network blip will send you debugging the wrong system.
      Distinguish "the system answered badly" from "the judge did not answer".
    """

    model: str = "TODO-pin-a-model-version"
    temperature: float = 0.0
    name: str = "llm"
    requires_api_key: bool = True
    _fallback: HeuristicJudge = field(default_factory=HeuristicJudge, repr=False)

    def build_prompt(self, case: dict, actual: str) -> str:
        """Render the grading prompt. Working - no reason to change it."""
        rubric = "\n".join(f"  {k}: {v}" for k, v in sorted(FAITHFULNESS_RUBRIC.items()))
        header = JUDGE_SYSTEM_PROMPT.format(
            rubric=rubric, modes=", ".join(FAILURE_MODES)
        )
        return (
            f"{header}\n"
            f"--- TICKET ---\n{case.get('input', '')}\n\n"
            f"--- EXPECTED HANDLING ---\n{case.get('expected', '')}\n\n"
            f"--- SYSTEM ANSWER ---\n{actual or '(empty)'}\n"
        )

    def score(self, case: dict, actual: str) -> Verdict:
        raw = self._call_model(self.build_prompt(case, actual))
        return self.parse(raw, case)

    def _call_model(self, prompt: str) -> str:
        """TODO: send `prompt` to your model and return the raw text response.

        Requirements the rest of this class already assumes:
          - returns a string containing a single JSON object
          - temperature 0 (or whatever you pin, but pin it and record it)
          - retries on transient errors, and raises rather than returning ""
            when the judge genuinely could not run
        """
        raise NotImplementedError(
            "LLMJudge._call_model is not implemented. This is expected on a "
            "fresh clone.\n"
            "  - `make eval` and CI run the HeuristicJudge instead, offline.\n"
            "  - Implement this method (and set a real `model`) when you want a "
            "judge that can read a ticket rather than pattern-match it.\n"
            "  - See eval/judge.py docstrings for the JSON contract and rubric."
        )

    def parse(self, raw: str, case: dict) -> Verdict:
        """Parse the model's JSON. Working - handles garbage without raising."""
        import json

        want_mode = str(case.get("failure_mode", ""))
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            return Verdict(0.0, False, "", "judge returned no parseable JSON")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return Verdict(0.0, False, "", f"judge JSON invalid: {exc}")

        predicted = str(data.get("predicted_mode", "") or "")
        if predicted not in FAILURE_MODES:
            predicted = ""
        return Verdict(
            faithfulness=data.get("faithfulness", 0.0),
            mode_correct=(predicted == want_mode and predicted != ""),
            predicted_mode=predicted,
            rationale=str(data.get("rationale", ""))[:200],
        )


def get_judge(prefer_llm: bool = False) -> object:
    """Return the judge to use.

    Defaults to the heuristic judge so the harness always runs. Pass
    prefer_llm=True (run_eval.py does this with --llm) to try the real judge and
    fall back loudly, never silently, if it is not implemented or has no key.
    """
    if not prefer_llm:
        return HeuristicJudge()
    judge = LLMJudge()
    try:
        judge._call_model("ping")
    except NotImplementedError:
        print(
            "[judge] LLMJudge is not implemented yet, falling back to "
            "HeuristicJudge. Scores below are lexical, not semantic."
        )
        return HeuristicJudge()
    return judge


def validate_modes(modes: Iterable[str]) -> list[str]:
    """Return any mode names that are not one of the five declared modes."""
    return sorted({m for m in modes if m not in FAILURE_MODES})

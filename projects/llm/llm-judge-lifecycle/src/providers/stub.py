"""The offline provider: no key, no network, no cost, and no pretending.

It is easy to build a mock that returns a canned success and makes a demo look
finished. That teaches nothing, and worse, it teaches something false - that
the pipeline works. This stub is built the other way round: it is a real,
deterministic implementation of each of the four roles, weak in ways you can
read in the source, and it fails often enough that every branch of the system
gets exercised.

What each role actually does here:

    judge       runs src/rules.py over the rubric it was handed. It reads the
                rubric, so RART genuinely optimises against it.
    generator   composes an explanation from the record and injects a defect
                chosen by hashing (record id, attempt). Given a critique, it
                repairs the named defect - sometimes introducing another. That
                is what makes the retry curve rise and then flatten, rather
                than march to 1.0 the way a scripted mock would.
    meta_judge  compares the judge's cited failure mode against the human's.
    reflector   mines the focus set for the phrase most often present in
                artefacts the judge wrongly passed, and adds it to the rubric's
                banned list. A crude text gradient, but a real one.

Determinism is seeded from ``providers.stub.seed`` in configs/base.yaml, so two
runs of the same command produce the same numbers and the test suite can assert
on them.

**Never quote a stub number as a model result.** Every artefact written while a
role is on this provider is stamped ``"provider": "stub"``, and `run.py` prints
a banner. What the stub measures is the quality of the rules in the rubric, not
the quality of anyone's judgement.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any

from ..config import RoleConfig
from ..rules import RuleEngine, add_banned_phrases
from . import Completion
from ._common import read_block, read_text_block

# Defects the generator can inject. Roughly balanced so the first draft fails
# often enough to make the revision loop worth watching, but not so often that
# every run bottoms out.
_DEFECTS = (
    None,  # clean
    "generic_filler",
    "unsupported_claim",
    "missing_subject",
    None,  # clean
    "too_short",
)


class StubProvider:
    name = "stub"

    def __init__(self, role_config: RoleConfig) -> None:
        self._cfg = role_config
        self.model = "stub"
        self.seed = int(role_config.extra.get("seed", 0))

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        handler = {
            "judge": self._judge,
            "generator": self._generate,
            "meta_judge": self._meta_judge,
            "reflector": self._reflect,
        }.get(self._cfg.role)
        if handler is None:
            raise ValueError(f"stub provider has no behaviour for role {self._cfg.role!r}")

        text = handler(prompt)
        return Completion(
            text=text,
            # Rough token accounting so cost reporting and the per-request call
            # cap behave the same offline as online. Four characters per token
            # is the usual back-of-envelope.
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            finish_reason="stop",
            model=self.model,
            provider=self.name,
        )

    # --------------------------------------------------------------- roles
    def _judge(self, prompt: str) -> str:
        record = read_block(prompt, "RECORD") or {}
        artefact = read_text_block(prompt, "ARTEFACT") or ""
        rubric = read_text_block(prompt, "RUBRIC") or ""
        verdict = RuleEngine(rubric).evaluate(artefact, record)
        return json.dumps(
            {
                "label": verdict.label,
                "reason": verdict.reason,
                "failure_mode": verdict.failure_mode,
            },
            ensure_ascii=False,
        )

    def _generate(self, prompt: str) -> str:
        record = read_block(prompt, "RECORD") or {}
        critique = read_text_block(prompt, "CRITIQUE")
        attempt = int(read_block(prompt, "ATTEMPT") or 0)

        subject = record.get("subject") or {}
        title = str(subject.get("title") or "this item")
        attributes = [str(a) for a in (subject.get("attributes") or [])][:2]
        references = record.get("references") or []
        ref_title = str((references[0] or {}).get("title", "")) if references else ""

        defect = _DEFECTS[
            _hash(self.seed, record.get("id", ""), attempt) % len(_DEFECTS)
        ]

        # Reason-guided revision. The critique names a failure mode; a competent
        # generator fixes THAT and does not regress elsewhere. This stub fixes
        # the named mode and then rerolls the remaining defects - which is why
        # the pass-rate curve climbs steeply for the first two or three retries
        # and then flattens, instead of reaching 1.0. A generator that always
        # repaired everything would produce a curve nobody has ever measured in
        # production, and the flattening is the whole point of Figure 3 in the
        # paper.
        if critique:
            named = _named_failure_mode(critique)
            if named and defect == named:
                defect = _DEFECTS[
                    _hash(self.seed, record.get("id", ""), attempt, "repair")
                    % len(_DEFECTS)
                ]
                if defect == named:
                    defect = None

        return _compose(title, attributes, ref_title, defect)

    def _meta_judge(self, prompt: str) -> str:
        """Do the judge and the human agree about WHY it failed?

        Compare the failure mode each of them named. When the judge's reason
        carries no explicit mode, fall back to token overlap against the human's
        rationale - deliberately generous, because the expensive mistake for a
        meta-judge is calling a genuine agreement a mismatch and sending the
        reflector off to fix a rubric that was fine.
        """
        judge_reason = read_text_block(prompt, "JUDGE_REASON") or ""
        human_rationale = read_text_block(prompt, "HUMAN_RATIONALE") or ""
        judge_mode = read_text_block(prompt, "JUDGE_FAILURE_MODE") or ""
        human_mode = read_text_block(prompt, "HUMAN_FAILURE_MODE") or ""

        if judge_mode.strip() and human_mode.strip():
            agree = judge_mode.strip().lower() == human_mode.strip().lower()
        else:
            agree = _overlap(judge_reason, human_rationale) >= 0.28

        return json.dumps(
            {
                "verdict": "RATIONALE_AGREEMENT" if agree else "RATIONALE_MISMATCH",
                "note": (
                    f"judge cited {judge_mode or 'no explicit mode'}; "
                    f"human cited {human_mode or 'no explicit mode'}"
                ),
            },
            ensure_ascii=False,
        )

    def _reflect(self, prompt: str) -> str:
        """Propose a revised rubric from the focus set. A real text gradient.

        The focus set holds two error types (see src/rart.py): examples the
        judge labelled wrongly, and examples it labelled right for the wrong
        reason. This stub acts on the tractable half - artefacts the judge
        PASSED that a human FAILED for generic filler - by finding the phrase
        most common across them and banning it.

        Crude, and it is meant to be legible rather than clever: you can read
        the rubric diff between iterations and see exactly what the optimiser
        decided and why. Watching a rubric grow a clause that lifts specificity
        by four points is worth more than a curve.
        """
        rubric = read_text_block(prompt, "RUBRIC") or ""
        focus = read_block(prompt, "FOCUS") or []

        false_passes = [
            item.get("artefact", "")
            for item in focus
            if item.get("judge_label") == "PASS" and item.get("human_label") == "FAIL"
        ]
        already = RuleEngine(rubric).banned_phrases()
        counts: Counter[str] = Counter()
        for artefact in false_passes:
            for phrase in _candidate_phrases(artefact):
                if phrase not in already:
                    counts[phrase] += 1

        # Require a phrase to appear in at least two failures. A rule inferred
        # from one example is memorisation, and it will cost recall on the test
        # set without buying anything.
        winners = [phrase for phrase, count in counts.most_common(3) if count >= 2]
        return add_banned_phrases(rubric, winners) if winners else rubric

    def estimated_usd(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_FILLER_CANDIDATES = (
    "you'll love it",
    "a must-watch",
    "highly rated",
    "perfect for anyone",
    "one of the greats",
    "you won't be disappointed",
    "an instant classic",
    "something for everyone",
)


def _compose(title: str, attributes: list[str], ref_title: str, defect: str | None) -> str:
    attrs = " and ".join(attributes) if attributes else "worth a look"
    like = f", much like {ref_title}" if ref_title else ""
    base = f"{title} is {attrs}{like}, which is why it is on your list."

    if defect == "generic_filler":
        return base + " You'll love it."
    if defect == "unsupported_claim":
        return base + " Rated 4.8 out of 5 by 12,000 viewers this month."
    if defect == "missing_subject":
        return f"This one is {attrs}{like}, which is why it is on your list."
    if defect == "too_short":
        return f"{title}. Recommended."
    return base


def _named_failure_mode(critique: str) -> str | None:
    for mode in (
        "generic_filler",
        "unsupported_claim",
        "missing_subject",
        "too_short",
        "too_long",
        "unsafe_content",
    ):
        if mode in critique:
            return mode
    return None


def _candidate_phrases(artefact: str) -> list[str]:
    low = (artefact or "").lower()
    return [phrase for phrase in _FILLER_CANDIDATES if phrase in low]


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z']+", (text or "").lower())
        if len(word) > 3
    }


def _hash(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def build(role_config: RoleConfig) -> StubProvider:
    return StubProvider(role_config)

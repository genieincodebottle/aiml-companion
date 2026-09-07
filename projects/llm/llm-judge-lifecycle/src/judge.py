"""The judge: one criterion, one rubric, one verdict.

One judge per criterion, not one judge scoring everything at once. That costs
three calls where a combined judge costs one, and it is worth it for reasons
that only show up later:

* A combined judge cannot be tuned. Fixing a groundedness weakness means
  editing a prompt that also governs safety, and every rubric change risks
  every criterion. RART needs an isolated parameter to optimise.
* A combined judge blurs the gate. Must-have and soft criteria carry different
  consequences - drop versus record - and a single verdict cannot express both.
* Failures correlate. A model that gets one criterion wrong on an artefact
  tends to get the others wrong on the same artefact, because it has formed a
  view. Separate calls do not eliminate that but they stop one confident
  misreading from taking all three down at once.

The rubric arrives as a parameter, never baked into the prompt. That is what
makes Phase II attributable: between iterations the rubric is the only thing
that changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from .domain import Criterion, Domain, Record
from .prompts import JUDGE_SCHEMA, JUDGE_SYSTEM, judge_prompt
from .runtime import Runtime

log = logging.getLogger(__name__)

PASS, FAIL = "PASS", "FAIL"


@dataclass
class Verdict:
    label: str
    reason: str
    failure_mode: str | None = None
    criterion_id: str = ""

    @property
    def passed(self) -> bool:
        return self.label == PASS


class Judge:
    def __init__(
        self,
        runtime: Runtime,
        domain: Domain,
        criterion: Criterion,
        rubric: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.domain = domain
        self.criterion = criterion
        self.rubric = rubric or domain.seed_rubric(criterion.id)

    def judge(self, record: Record, artefact: str) -> Verdict:
        completion = self.runtime.call(
            "judge",
            judge_prompt(self.domain, self.criterion, self.rubric, record, artefact),
            system=JUDGE_SYSTEM,
            json_schema=JUDGE_SCHEMA,
        )
        return self._parse(completion.text)

    def _parse(self, text: str) -> Verdict:
        """Parse and validate. Both halves matter.

        Not every provider enforces a schema - several OpenAI-compatible
        servers implement only ``json_object``, which constrains syntax and not
        shape - so a well-formed object with a missing or invented ``label`` is
        a thing that actually arrives. Trusting the schema and reading
        ``data["label"]`` directly gives a KeyError deep inside a tuning loop,
        or worse, a label of "Pass" that never equals "PASS" and silently
        counts as a failure everywhere downstream.
        """
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError:
            log.error("judge returned unparseable output: %.300s", text)
            return self._unparseable("output was not JSON")

        if not isinstance(data, dict):
            return self._unparseable("output was JSON but not an object")

        label = str(data.get("label", "")).strip().upper()
        if label not in (PASS, FAIL):
            return self._unparseable(f"label {data.get('label')!r} is not PASS or FAIL")

        reason = str(data.get("reason", "")).strip()
        mode = data.get("failure_mode")
        mode = str(mode).strip() if mode else None

        # A FAIL with no reason is not usable. In Phase III the reason IS the
        # revision instruction, so an empty one produces a retry with no
        # guidance that burns the budget and changes nothing.
        if label == FAIL and not reason:
            reason = (
                f"Fails the {self.criterion.display!r} criterion. "
                "(The judge returned no reason; revision guidance is degraded.)"
            )

        if mode and self.criterion.failure_modes and mode not in self.criterion.failure_modes:
            # Keep it - the verdict is still valid and the reason still steers
            # the generator - but say so. A mode outside the declared vocabulary
            # scores as a reason mismatch in Phase II, and without this line you
            # would spend a while wondering why reasoning agreement collapsed
            # after an otherwise sensible rubric edit.
            log.warning(
                "judge for %r returned failure_mode %r, which is not in the "
                "declared vocabulary %s. It will count as a reason mismatch.",
                self.criterion.id,
                mode,
                list(self.criterion.failure_modes),
            )

        return Verdict(label, reason, mode, self.criterion.id)

    def _unparseable(self, why: str) -> Verdict:
        """Fail closed.

        An unreadable verdict is treated as FAIL, which drops the artefact.
        Defaulting to PASS would mean that whenever the judge breaks, everything
        it was supposed to gate sails through - the guardrail fails open exactly
        when it has stopped working, and nothing in the metrics would show it
        because the artefacts were never judged at all.

        The failure mode is deliberately outside every domain vocabulary so
        these show up as reason mismatches in Phase II rather than blending in.
        """
        return Verdict(
            FAIL,
            f"Judge output could not be read ({why}); failing closed.",
            "judge_error",
            self.criterion.id,
        )


class JudgePanel:
    """All criteria for one domain, with the gate/soft split preserved.

    ``gate_failures`` is what Phase III acts on. ``soft_failures`` is recorded
    and reported and never blocks. Collapsing the two gives you either a gate
    that drops accurate explanations over a word count, or one that ships
    spoilers - and teams that discover they need the distinction usually
    discover it from the first of those.
    """

    def __init__(
        self, runtime: Runtime, domain: Domain, rubrics: dict[str, str] | None = None
    ) -> None:
        rubrics = rubrics or {}
        self.domain = domain
        self.judges = {
            c.id: Judge(runtime, domain, c, rubrics.get(c.id))
            for c in domain.criteria
        }

    def judge_all(self, record: Record, artefact: str) -> dict[str, Verdict]:
        return {cid: judge.judge(record, artefact) for cid, judge in self.judges.items()}

    def gate_failures(self, verdicts: dict[str, Verdict]) -> list[Verdict]:
        return [
            v
            for cid, v in verdicts.items()
            if not v.passed and self.domain.criterion(cid).must_have
        ]

    def soft_failures(self, verdicts: dict[str, Verdict]) -> list[Verdict]:
        return [
            v
            for cid, v in verdicts.items()
            if not v.passed and not self.domain.criterion(cid).must_have
        ]

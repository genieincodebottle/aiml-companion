"""The reasoning meta-judge: did the judge fail it for the RIGHT reason?

This is the piece that turns rubric tuning into *reasoning-aligned* rubric
tuning, and it is the part of the method most likely to be dropped as
unnecessary. It is not.

Why a right verdict for the wrong reason is a defect
----------------------------------------------------
Two arguments, and the second is the one that decides it.

**Generalisation.** A rubric that rejects the right artefacts by accident stops
rejecting them as soon as the inputs shift. Label accuracy cannot see the
difference between a rubric that has understood the criterion and one that has
learned a correlation, because both produce the same labels on the data you
have. Reason agreement can.

**Deployment.** In Phase III the judge's rejection reason is handed to the
generator as its revision instruction. A wrong reason does not merely fail to
help - it actively steers the next draft towards fixing something that was
never broken, while the real defect survives every retry. The retry budget is
spent, the artefact is dropped, and the pass-rate-versus-k curve flattens for a
reason that no label-only metric can explain.

Why only agreed-fails
---------------------
The meta-judge runs ONLY where judge and human both said FAIL. That is not an
optimisation.

* Both said PASS: there is no reason to compare. Neither party wrote one.
* They disagree on the label: the disagreement is already counted, by
  specificity or recall. Asking whether the reasons match too would count one
  error twice, and would push the optimiser to over-weight cases that are
  already penalised.

Agreed-fails are the only place where a shared label can still hide divergent
reasoning, so they are the only place worth looking.
"""

from __future__ import annotations

import json
import logging

from .prompts import META_JUDGE_SCHEMA, META_JUDGE_SYSTEM, meta_judge_prompt
from .runtime import Runtime

log = logging.getLogger(__name__)

AGREEMENT = "RATIONALE_AGREEMENT"
MISMATCH = "RATIONALE_MISMATCH"


class MetaJudge:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def agrees(
        self,
        artefact: str,
        judge_reason: str,
        judge_mode: str | None,
        human_rationale: str,
        human_mode: str | None,
    ) -> bool:
        completion = self.runtime.call(
            "meta_judge",
            meta_judge_prompt(
                artefact, judge_reason, judge_mode, human_rationale, human_mode
            ),
            system=META_JUDGE_SYSTEM,
            json_schema=META_JUDGE_SCHEMA,
        )
        return self._parse(completion.text)

    def _parse(self, text: str) -> bool:
        """Unreadable output counts as AGREEMENT.

        The opposite of the judge's fail-closed default, and deliberately so.
        The two failures cost different things.

        A broken judge that defaults to PASS ships bad artefacts to users. A
        broken meta-judge that defaults to MISMATCH pushes every unparseable
        response into the focus set, and the reflector then rewrites the rubric
        to fix disagreements that were never observed. Tuning would be driven by
        parser noise, and the resulting rubric would look tuned and be worse.

        Failing towards "no signal" is the safe direction for an optimiser's
        input. Failing towards "no signal" is the unsafe direction for a gate.
        Same word, opposite defaults, and the reason is which way the error
        propagates.
        """
        try:
            data = json.loads(text)
            verdict = str(data.get("verdict", "")).strip().upper()
        except (json.JSONDecodeError, AttributeError):
            log.warning("meta-judge returned unparseable output: %.200s", text)
            return True

        if verdict not in (AGREEMENT, MISMATCH):
            log.warning("meta-judge returned unknown verdict %r", verdict)
            return True
        return verdict == AGREEMENT

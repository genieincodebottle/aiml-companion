"""The deterministic stub 'brain' - schema-valid outputs without an LLM.

This is what makes stub mode a faithful stand-in rather than a mock. Given the same
(schema, system, user) a live model would receive, it produces a schema-valid object
by reading simple, explicit signals out of the prompt text (the squad embeds compact
``FINDINGS:`` blocks the brain keys off). The result: the full platform - gateway,
policy, runtime, HITL, evals, tracing - runs end-to-end, offline and free, and the
three seed alerts reach their known ground-truth dispositions.

Signals are intentionally transparent: `sanctioned`/`sanctions_hit` -> file a SAR,
`pep`/`pep_hit` (without sanctions) -> escalate, otherwise -> close.
"""

from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from ...models.schemas import (
    CritiqueResult,
    EvidenceItem,
    GraphFinding,
    JudgeVerdict,
    NarrativeDraft,
    ScreeningResult,
    TriageResult,
)

T = TypeVar("T", bound=BaseModel)


def _has(text: str, *needles: str) -> bool:
    low = text.lower()
    return any(n in low for n in needles)


def synthesize(schema: Type[T], system: str, user: str) -> T:
    """Return a schema-valid object derived from the prompt's signals.

    We key off `user` (the data the agent gathered), never `system` (the instructions),
    so instruction words like 'unsupported' or field names like 'sanctioned_in_network'
    cannot be mistaken for evidence. The agents pass explicit value signals for the
    load-bearing decisions (e.g. 'sanctions=true').
    """
    text = user

    if schema is TriageResult:
        actionable = not _has(text, "consistent retail", "normal retail", "below threshold")
        high = _has(text, "structuring", "sanction", "layering", "pep")
        return TriageResult(  # type: ignore[return-value]
            severity="high" if high else ("low" if not actionable else "medium"),
            is_actionable=actionable,
            rationale="Signals in the alert and network indicate elevated risk."
            if high
            else "Activity is consistent with the entity's expected profile.",
        )

    if schema is GraphFinding:
        patterns: list[str] = []
        related: list[str] = []
        # Key off the tool's sanctioned list, never an entity name: ALRT-005's customer
        # shares a name with the sanctioned shell and must not be flagged for it.
        if _has(text, '"sanctioned_in_network": ["'):
            patterns.append("reconvergence on a sanctioned shell company")
            related.append("E900")
        if _has(text, "fan-out", "fan out", "structuring"):
            patterns.append("fan-out structuring to newly opened accounts")
        if _has(text, "pep", "layering"):
            patterns.append("layering through a PEP-controlled entity")
        return GraphFinding(  # type: ignore[return-value]
            related_entities=related,
            suspicious_patterns=patterns,
            risk_note="Network structure is consistent with money movement designed to obscure origin."
            if patterns
            else "No suspicious structure evident in the network.",
        )

    if schema is ScreeningResult:
        # Key off the tool's actual boolean values, not field names.
        sanctions = _has(text, 'sanctions_hit": true', "ofac-sdn")
        pep = _has(text, 'pep_hit": true', "pep-register")
        lists: list[str] = []
        if sanctions:
            lists.append("OFAC-SDN")
        if pep:
            lists.append("PEP-Register")
        return ScreeningResult(  # type: ignore[return-value]
            sanctions_hit=sanctions,
            pep_hit=pep,
            matched_lists=lists,
            note="Watchlist evidence supports the reported matches." if lists else "No watchlist matches found.",
        )

    if schema is NarrativeDraft:
        # Read the explicit, evidence-derived signals the narrative agent emitted.
        sanctions = _has(text, "sanctions=true")
        pep = _has(text, "pep=true")
        structuring = _has(text, "structuring=true")
        if sanctions:
            action = "file_sar"
        elif pep:
            action = "escalate"
        else:
            action = "close"
        claims: list[str] = []
        evidence: list[EvidenceItem] = []
        if sanctions:
            claims.append("Funds reconverge on Meridian Holdings Ltd, an OFAC-sanctioned shell company.")
            evidence.append(EvidenceItem(source="graph.expand", detail="E900 flagged sanctioned in network"))
            evidence.append(EvidenceItem(source="screening.check", detail="E900 matches OFAC-SDN"))
        if structuring:
            claims.append("The subject fanned out sub-threshold transfers to newly opened mule accounts.")
            evidence.append(EvidenceItem(source="graph.expand", detail="fan-out to E101/E102/E103"))
        if pep and not sanctions:
            claims.append("Transfers were layered through a PEP-controlled entity across jurisdictions.")
            evidence.append(EvidenceItem(source="screening.check", detail="entity matches PEP-Register"))
        if not claims:
            claims.append("Activity is consistent with the entity's established retail profile.")
            evidence.append(EvidenceItem(source="alert.read", detail="amount consistent with history"))
        return NarrativeDraft(  # type: ignore[return-value]
            summary="Investigation of the alerted entity and its transaction network.",
            claims=claims,
            evidence=evidence,
            recommended_action=action,  # type: ignore[arg-type]
        )

    if schema is CritiqueResult:
        # The critic approves when the structural citation check (passed in as an
        # explicit signal by the critic agent) confirms every claim is cited.
        cited = _has(text, "every_claim_cited=true")
        return CritiqueResult(  # type: ignore[return-value]
            approved=cited,
            every_claim_cited=cited,
            issues=[] if cited else ["one or more claims lack a cited evidence item"],
            note="All claims are backed by cited evidence." if cited else "Citation gap detected.",
        )

    if schema is JudgeVerdict:
        # Read the EXPLICIT structural signals the eval computes, the same way
        # the critic branch above reads `every_claim_cited=`.
        #
        # This used to keyword-match the narrative prose for "uncited",
        # "unsupported", "mismatch". That is a confession detector, not a
        # grounding check: a narrative is ungrounded because it LACKS citations,
        # not because it announces that it lacks them. Measured on the old code:
        #
        #   claims=['A is laundering money','B is a shell'], evidence=[]  -> 1.0
        #   claims=['definitely fraud'],                     evidence=[]  -> 1.0
        #   claims=['uncited claim']                                      -> 0.6
        #
        # Only the narrative that used the magic word scored below perfect, so
        # the offline promotion gate could not fail on grounding and reduced to
        # "disposition accuracy == 100%". An eval that cannot return a bad score
        # is not measuring anything.
        grounded = _has(text, "uncited_claims=0") and not _has(text, "evidence_count=0")
        sound = not _has(text, "disposition_sound=false", "disposition_matches_expected=false")
        if not _has(text, "uncited_claims="):
            # No structural signal supplied: fall back to the old prose scan so
            # ad-hoc callers still get something, but never award a perfect
            # score on evidence that was never checked.
            grounded = not _has(text, "uncited", "unsupported", "grounded=false")
            sound = not _has(text, "mismatch", "wrong disposition")
            # 0.7, not 0.8: the default MC_EVAL_THRESHOLD is 0.8 and the gate
            # tests `>=`, so 0.8 would have cleared the very gate this branch
            # exists to withhold.
            score = 0.7 if (grounded and sound) else (0.5 if grounded or sound else 0.2)
            return JudgeVerdict(  # type: ignore[return-value]
                score=score,
                grounded=grounded,
                disposition_sound=sound,
                rationale="No structural signals supplied; scored from prose only, "
                "capped below the promotion threshold.",
            )
        score = 1.0 if (grounded and sound) else (0.6 if grounded or sound else 0.2)
        return JudgeVerdict(  # type: ignore[return-value]
            score=score,
            grounded=grounded,
            disposition_sound=sound,
            rationale="Narrative is fully grounded and the disposition follows from the evidence."
            if score >= 0.99
            else "Investigation has gaps in grounding or disposition support.",
        )

    # Fallback: construct a default instance if the schema is unrecognised.
    return schema()  # type: ignore[call-arg]

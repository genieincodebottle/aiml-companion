"""The guardrail engine: three enforcement points, one object.

    INGEST TIME     guard.check_document(text, doc_id)
                    Protects the integrity of the graph. The most important of
                    the three, because what passes here becomes persistent
                    shared state that every future query traverses.

    QUERY TIME      guard.check_question(question, caller)
                    Protects one answer, the system prompt, and the budget.

    RESPONSE TIME   guard.check_answer(answer, retrieval)
                    Protects the user from a fabricated citation, an invented
                    supplier, or a number the model adjusted on its way out.

Everything is configured from `configs/base.yaml` under `guardrails`, so a
deployment can tighten or relax each control without editing code, and so the
settings are visible in a file a reviewer can read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .audit import AuditLog
from .injection import ScanResult, scan_document, scan_question, wrap_untrusted
from .limits import Budget, BudgetExceeded, InputLimits, RateLimiter
from .pii import PIIResult, scan_and_redact
from .validate import ValidationResult, validate_answer

__all__ = [
    "GuardrailEngine", "GuardrailViolation", "GuardrailDecision",
    "AuditLog", "Budget", "BudgetExceeded", "InputLimits", "RateLimiter",
    "scan_document", "scan_question", "scan_and_redact", "validate_answer",
    "wrap_untrusted", "ScanResult", "PIIResult", "ValidationResult",
]


class GuardrailViolation(Exception):
    """A guardrail refused the request. Carries a reason fit to show a user."""

    def __init__(self, message: str, *, kind: str, detail: Any = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.detail = detail


@dataclass
class GuardrailDecision:
    """What the guardrails did, returned alongside the result rather than
    hidden. The UI renders this, because a guardrail the user cannot see is a
    guardrail they cannot trust."""

    allowed: bool = True
    text: str = ""
    checks: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append({"check": name, "status": status, "detail": detail})

    @property
    def passed(self) -> list[str]:
        return [c["check"] for c in self.checks if c["status"] == "pass"]

    @property
    def flagged(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["status"] in {"warn", "block"}]

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "checks": self.checks}


class GuardrailEngine:
    def __init__(self, config: Any) -> None:
        settings: dict[str, Any] = {}
        try:
            settings = config.section("guardrails")
        except (KeyError, TypeError):
            settings = {}

        self.enabled = settings.get("enabled", True)
        self.block_on_injection = settings.get("block_on_injection", True)
        self.redact_pii = settings.get("redact_pii", True)
        self.block_on_secrets = settings.get("block_on_secrets", True)
        self.validate_output = settings.get("validate_output", True)

        self.limits = InputLimits(
            max_question_chars=settings.get("max_question_chars", 2000),
            max_document_chars=settings.get("max_document_chars", 200_000),
        )
        self.budget = Budget(
            max_llm_calls=settings.get("max_llm_calls_per_request", 12),
            max_input_tokens=settings.get("max_input_tokens_per_request", 250_000),
            max_usd=settings.get("max_usd_per_request", 0.50),
        )
        self.rate_limiter = RateLimiter(
            max_requests=settings.get("rate_limit_requests", 30),
            window_seconds=settings.get("rate_limit_window_seconds", 60),
        )
        self.audit = AuditLog(
            Path(config.root) / settings.get("audit_log", "artifacts/audit.jsonl")
        )

    # ------------------------------------------------------------ ingest time
    def check_document(self, text: str, doc_id: str) -> GuardrailDecision:
        """Runs before the extractor sees a document.

        Order matters: size, then secrets, then injection, then PII redaction.
        Cheapest and most decisive first, so an oversized or credential-bearing
        document never reaches the more expensive checks.
        """
        decision = GuardrailDecision(text=text)
        if not self.enabled:
            decision.record("guardrails", "skipped", "disabled in config")
            return decision

        try:
            self.limits.check_document(text, doc_id)
            decision.record("document_size", "pass", f"{len(text)} chars")
        except ValueError as exc:
            decision.allowed = False
            decision.record("document_size", "block", str(exc))
            self.audit.write("ingest_blocked", doc_id=doc_id, reason="size", blocked=True)
            raise GuardrailViolation(str(exc), kind="document_size")

        pii = scan_and_redact(text)
        if pii.has_secrets and self.block_on_secrets:
            kinds = sorted({f.kind for f in pii.secrets})
            decision.allowed = False
            decision.record("secrets", "block", f"detected: {', '.join(kinds)}")
            self.audit.write("ingest_blocked", doc_id=doc_id, reason="secret",
                             kinds=kinds, blocked=True)
            raise GuardrailViolation(
                f"{doc_id} contains what looks like a credential ({', '.join(kinds)}). "
                "Ingestion stopped. The correct response is to rotate that "
                "credential and remove it from the source document - not to "
                "redact it and carry on, which would remove the only signal "
                "that it was ever exposed.",
                kind="secret_in_document", detail=kinds,
            )
        decision.record("secrets", "pass" if not pii.has_secrets else "warn",
                        pii.summary())

        scan = scan_document(pii.text if self.redact_pii else text, doc_id)
        if scan.blocked and self.block_on_injection:
            decision.allowed = False
            decision.record("prompt_injection", "block", scan.summary())
            self.audit.write("ingest_blocked", doc_id=doc_id, reason="injection",
                             detections=[d.group for d in scan.detections],
                             excerpts=[d.excerpt[:200] for d in scan.detections],
                             blocked=True)
            raise GuardrailViolation(
                f"{doc_id} contains text that reads as an instruction to the "
                f"extractor ({scan.summary()}). Ingestion stopped.\n\n"
                "This matters more in GraphRAG than in ordinary RAG: a poisoned "
                "document here does not corrupt one answer, it writes an edge "
                "into shared persistent state that every future traversal can "
                "reach, for every user, carrying a real citation.",
                kind="prompt_injection",
                detail=[{"group": d.group, "excerpt": d.excerpt} for d in scan.detections],
            )
        if scan.needs_review:
            decision.record("prompt_injection", "warn", scan.summary())
            self.audit.write("ingest_flagged", doc_id=doc_id,
                             detections=[d.group for d in scan.detections],
                             warnings=True)
        else:
            decision.record("prompt_injection", "pass", "no patterns matched")

        decision.text = scan.cleaned
        if self.redact_pii and pii.findings:
            decision.record("pii_redaction", "warn",
                            f"redacted {pii.summary()} before extraction")
        else:
            decision.record("pii_redaction", "pass", "nothing to redact")
        return decision

    # ------------------------------------------------------------- query time
    def check_question(self, question: str, caller: str = "local") -> GuardrailDecision:
        decision = GuardrailDecision(text=question)
        if not self.enabled:
            decision.record("guardrails", "skipped", "disabled in config")
            return decision

        allowed, retry_after = self.rate_limiter.check(caller)
        if not allowed:
            decision.allowed = False
            decision.record("rate_limit", "block", f"retry in {retry_after:.0f}s")
            self.audit.write("question_blocked", caller=caller, reason="rate_limit",
                             blocked=True)
            raise GuardrailViolation(
                f"Rate limit reached. Try again in {retry_after:.0f} seconds.",
                kind="rate_limit",
            )
        decision.record("rate_limit", "pass", f"caller={caller}")

        try:
            self.limits.check_question(question)
            decision.record("input_length", "pass", f"{len(question.strip())} chars")
        except ValueError as exc:
            decision.allowed = False
            decision.record("input_length", "block", str(exc))
            raise GuardrailViolation(str(exc), kind="input_length")

        scan = scan_question(question)
        if scan.blocked and self.block_on_injection:
            decision.allowed = False
            decision.record("prompt_injection", "block", scan.summary())
            self.audit.write("question_blocked", caller=caller, reason="injection",
                             detections=[d.group for d in scan.detections],
                             blocked=True)
            raise GuardrailViolation(
                "That question contains an instruction aimed at the model "
                f"rather than a question about the supply chain ({scan.summary()}).",
                kind="prompt_injection",
                detail=[{"group": d.group, "excerpt": d.excerpt} for d in scan.detections],
            )
        decision.record("prompt_injection", "pass", "no patterns matched")

        pii = scan_and_redact(question)
        if pii.has_secrets:
            decision.text = pii.text
            decision.record("secrets", "warn", "redacted before sending to the model")
        else:
            decision.record("secrets", "pass", "none detected")

        decision.text = pii.text if self.redact_pii else question
        return decision

    # ---------------------------------------------------------- response time
    def check_answer(self, answer_text: str, *, context: str,
                     available_documents: Iterable[str],
                     graph_entity_names: Iterable[str] | None = None,
                     graph_fact_ids: Iterable[str] | None = None,
                     ) -> tuple[GuardrailDecision, ValidationResult]:
        decision = GuardrailDecision(text=answer_text)
        result = ValidationResult()
        if not self.enabled or not self.validate_output:
            decision.record("output_validation", "skipped", "disabled in config")
            return decision, result

        result = validate_answer(
            answer_text, context=context,
            available_documents=available_documents,
            graph_entity_names=graph_entity_names,
            known_graph_fact_ids=graph_fact_ids,
        )
        for warning in result.warnings:
            decision.record(warning.kind,
                            "block" if warning.severity == "error" else "warn",
                            warning.detail)
        if result.clean:
            decision.record("output_validation", "pass",
                            f"{len(result.cited_documents)} citations verified")

        # Not raised. A failed output check means "show this to the user with
        # the warning attached", not "hide the answer". Suppressing the answer
        # would also suppress the evidence that the system misbehaved, and the
        # user is the one best placed to judge.
        decision.allowed = True
        if not result.ok:
            self.audit.write("answer_flagged", warnings=True,
                             kinds=[w.kind for w in result.warnings])
        return decision, result

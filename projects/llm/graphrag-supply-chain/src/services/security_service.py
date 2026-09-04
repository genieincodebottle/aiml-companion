"""Guardrail inspection service.

Exposes the guardrails as an inspectable subsystem rather than an invisible
one. A control nobody can watch firing is a control that silently stops working
after a refactor, and nobody notices until it matters.

See `src/services/qa.py` for the layering rationale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..guardrails import GuardrailEngine
from ..guardrails.injection import scan_document, scan_question
from ..guardrails.pii import scan_and_redact


class SecurityService:
    def __init__(self, *, config: Config, guard: GuardrailEngine) -> None:
        self.config = config
        self.guard = guard

    def policy(self) -> dict[str, Any]:
        """The ACTIVE policy, as loaded - not as written in the YAML.

        The distinction matters: a setting that failed to parse falls back to a
        default, and reading the file would not tell you that happened.
        """
        guard = self.guard
        return {
            "enabled": guard.enabled,
            "block_on_injection": guard.block_on_injection,
            "block_on_secrets": guard.block_on_secrets,
            "redact_pii": guard.redact_pii,
            "validate_output": guard.validate_output,
            "max_question_chars": guard.limits.max_question_chars,
            "max_document_chars": guard.limits.max_document_chars,
            "rate_limit": {
                "requests": guard.rate_limiter.max_requests,
                "window_seconds": guard.rate_limiter.window,
            },
            "budget_per_request": {
                "max_llm_calls": guard.budget.max_llm_calls,
                "max_input_tokens": guard.budget.max_input_tokens,
                "max_usd": guard.budget.max_usd,
            },
            "enforcement_points": [
                {"stage": "ingest", "protects": "the integrity of the graph",
                 "note": "The important one. What passes here becomes persistent "
                         "shared state every future traversal can reach."},
                {"stage": "query", "protects": "one answer, the system prompt, the budget",
                 "note": "Rate limit, length cap, injection scan, secret redaction."},
                {"stage": "response", "protects": "the user from fabricated citations, "
                                                  "invented entities and adjusted numbers",
                 "note": "Deterministic checks. No judge - a model that finds an "
                         "invented supplier plausible when writing finds it "
                         "plausible when grading."},
            ],
        }

    def scan(self, text: str, *, as_document: bool = True) -> dict[str, Any]:
        """Run the scanners over arbitrary text.

        `as_document` changes the answer, and that is the point. The
        graph-poisoning patterns apply only to text destined for the extractor:
        a user may legitimately ask "which suppliers should we add a second
        source for", while the same words in an ingested document are an
        instruction aimed at the thing that writes to the database.
        """
        result = scan_document(text) if as_document else scan_question(text)
        pii = scan_and_redact(text)
        return {
            "blocked": result.blocked,
            "needs_review": result.needs_review,
            "summary": result.summary(),
            "detections": [
                {"group": d.group, "severity": d.severity, "pattern": d.pattern,
                 "excerpt": d.excerpt}
                for d in result.detections
            ],
            "pii": {
                "summary": pii.summary(),
                "has_secrets": pii.has_secrets,
                "findings": [{"kind": f.kind, "excerpt": f.excerpt,
                              "is_secret": f.is_secret} for f in pii.findings],
                "redacted_preview": pii.text[:600],
            },
        }

    def audit(self, *, limit: int = 100, event: str | None = None) -> dict[str, Any]:
        return {"events": self.guard.audit.tail(limit=limit, event=event),
                "stats": self.guard.audit.stats(),
                "path": str(self.guard.audit.path)}

    def adversarial_sample(self) -> dict[str, Any] | None:
        """The poisoned document, so the UI can demonstrate the block live.

        Stored outside `data/documents/` and never ingested. Keeping an attack
        sample where the ingestion pipeline would pick it up is how a
        demonstration becomes an incident.
        """
        path = (Path(self.config.root) / "data" / "adversarial"
                / "POISONED-SUPPLIER-RESPONSE.md")
        if not path.exists():
            return None
        return {"doc_id": path.stem, "path": path.name,
                "text": path.read_text(encoding="utf-8")}

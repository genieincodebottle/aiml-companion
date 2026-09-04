"""PII and secret detection, applied before text leaves the process.

Two different problems that get conflated and should not be:

  PII      Personal data that must not be sent to a third-party model without
           a basis for doing so. In this domain it arrives incidentally - an
           auditor's name and email in a report footer, a plant manager's phone
           number in a supplier profile.

  SECRETS  Credentials that must not be sent anywhere, ever. These end up in
           corpora more often than anyone expects, usually because someone
           pasted a connection string into a runbook that later got indexed.

Why redact rather than block? Because in this domain the PII is almost always
incidental to the answer. An audit report is useful because of its findings,
not because of the auditor's email address. Blocking the document would lose
the findings to protect a detail nobody asked for. Redaction keeps the value
and drops the exposure.

Secrets are different and are always blocked, never redacted through: if a
credential is in your corpus, the correct response is to rotate it, and a
pipeline that quietly redacts and carries on removes the signal that would have
told you.

WHAT THIS IS NOT
================
This is regex-based, and regex-based PII detection has a real false-negative
rate. A production deployment handling regulated personal data should use a
dedicated recogniser (Microsoft Presidio, or a cloud DLP API) with a trained
NER model behind it, and should treat this module as a cheap first pass rather
than a control it relies on. That trade-off is discussed in
docs/production-notes.md. What is here is honest about its limits and covers
the structured identifiers, which are the ones a regex is genuinely good at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Structured identifiers. Regex is well suited to these because they have a
# defined shape - which is exactly why the free-text cases below are weaker.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    "phone_intl": re.compile(r"(?<!\w)\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\w)"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # Payment card with a Luhn check, because a bare 16-digit pattern matches
    # order numbers, batch codes and part quantities all day long.
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
}

# Credentials. Blocked, never redacted-and-forgotten.
_SECRETS: dict[str, re.Pattern[str]] = {
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bolt_url_with_password": re.compile(r"\b(?:bolt|neo4j)(?:\+s)?://[^\s:]+:[^\s@]+@"),
    "generic_assignment": re.compile(
        r"\b(?:password|passwd|secret|api[_-]?key|token)\s*[:=]\s*"
        r"['\"]?[A-Za-z0-9_\-]{12,}['\"]?", re.IGNORECASE),
}


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for char in reversed(digits):
        value = ord(char) - 48
        if alt:
            value *= 2
            if value > 9:
                value -= 9
        total += value
        alt = not alt
    return total % 10 == 0


@dataclass
class PIIFinding:
    kind: str
    excerpt: str
    is_secret: bool = False


@dataclass
class PIIResult:
    text: str                                   # redacted text
    findings: list[PIIFinding] = field(default_factory=list)

    @property
    def secrets(self) -> list[PIIFinding]:
        return [f for f in self.findings if f.is_secret]

    @property
    def has_secrets(self) -> bool:
        return bool(self.secrets)

    def summary(self) -> str:
        if not self.findings:
            return "clean"
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.kind] = counts.get(finding.kind, 0) + 1
        return ", ".join(f"{k}x{v}" for k, v in sorted(counts.items()))


def _mask(value: str, kind: str) -> str:
    """Redact but keep the shape.

    `[EMAIL_REDACTED]` rather than deleting the span, because deletion changes
    the sentence and can invert its meaning. Keeping a labelled placeholder also
    lets the model say "an email address was redacted here" instead of
    hallucinating a name to fill the gap.
    """
    return f"[{kind.upper()}_REDACTED]"


def scan_and_redact(text: str) -> PIIResult:
    findings: list[PIIFinding] = []
    redacted = text

    for kind, pattern in _SECRETS.items():
        for match in pattern.finditer(text):
            findings.append(
                PIIFinding(kind=kind, excerpt=_truncate(match.group(0)), is_secret=True)
            )
        redacted = pattern.sub(f"[{kind.upper()}_REDACTED]", redacted)

    for kind, pattern in _PATTERNS.items():
        def _replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            if kind == "credit_card":
                digits = re.sub(r"\D", "", raw)
                # Without the Luhn check this pattern flags every long numeric
                # string in an engineering document.
                if not (13 <= len(digits) <= 19 and _luhn(digits)):
                    return raw
            if kind == "ipv4" and not _plausible_ip(raw):
                return raw
            findings.append(PIIFinding(kind=kind, excerpt=_truncate(raw)))
            return _mask(raw, kind)

        redacted = pattern.sub(_replace, redacted)

    return PIIResult(text=redacted, findings=findings)


def _plausible_ip(value: str) -> bool:
    """Reject version numbers and measurements that look like IPs.

    "1.2.3.4" in a document is far more likely to be a clause reference than an
    address, and every octet above 255 is definitely not an address.
    """
    parts = value.split(".")
    if len(parts) != 4:
        return False
    try:
        octets = [int(p) for p in parts]
    except ValueError:
        return False
    if any(o > 255 for o in octets):
        return False
    return any(o > 9 for o in octets)


def _truncate(value: str, keep: int = 4) -> str:
    """Never log the full secret. Logging a credential to prove you found a
    credential is a bug that has caused real incidents."""
    value = value.strip()
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}...{'*' * 6}"

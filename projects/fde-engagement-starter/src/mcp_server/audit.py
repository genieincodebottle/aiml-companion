"""Append-only audit log.

YOU IMPLEMENT THIS. See tests/test_audit.py for the contract.

An audit log is the artifact that makes an AI system deployable in a regulated
environment. It is also the thing most prototypes bolt on last, at which point it
records only the calls somebody remembered to instrument, which is worse than
having none because it looks complete.

The rule this file enforces: no tool call completes without a record. Success,
failure, and refusal all produce one. If the audit write fails, the tool call
fails. That ordering is deliberate and it is the part people get wrong: a system
that performs the action and then fails to log it has done something nobody can
account for, which in a bank or a hospital is the worse outcome.

The contract the tests enforce:

1. Records are append-only. Nothing rewrites or truncates an existing line.
2. One JSON object per line (JSONL), so a partial write damages one record rather
   than the file.
3. Every record carries: timestamp (UTC, ISO 8601), principal, tool, outcome
   (`success` | `denied` | `error`), and a stable request id.
4. Arguments are recorded, but PII-bearing fields are redacted before the write.
   An audit log that leaks the data it was protecting is a finding, not a control.
5. Writing the record is not optional and not best-effort.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Outcome = Literal["success", "denied", "error"]

# Fields redacted before an argument dict is written. Extend this when the
# customer's data has more, and say in your ADR how you decided what counts.
REDACT_KEYS = frozenset(
    {"address", "recipient_name", "phone", "email", "customer_id", "cust_id"}
)


@dataclass(frozen=True)
class AuditRecord:
    """One line of the audit log."""

    request_id: str
    timestamp: str
    principal: str
    tool: str
    outcome: Outcome
    arguments: dict[str, Any]
    detail: str = ""


def redact(arguments: dict[str, Any]) -> dict[str, Any]:
    """Replace PII-bearing values with a typed placeholder.

    Keep the KEY and the fact that a value was present. `{"address": "[REDACTED]"}`
    is auditable; dropping the key entirely means a reviewer cannot tell whether
    the call included an address at all.

    Implementer notes:
        - Nested dicts count. A flat pass over top-level keys is the usual bug.
        - Do not mutate the caller's dict.
    """
    raise NotImplementedError("Implement redact. See tests/test_audit.py.")


def write_record(record: AuditRecord, path: Path | None = None) -> None:
    """Append one record to the audit log.

    Args:
        record: the record to write.
        path: override for the log location, defaults to settings.audit_log_path.

    Raises:
        OSError: if the record cannot be written. Let this propagate. The caller
            must fail the tool call rather than proceed unlogged.

    Implementer notes:
        - Open in append mode, write one line, flush. Do not hold the file open
          across calls; a long-running process that crashes should not lose the
          buffered tail.
        - `json.dumps(..., ensure_ascii=False)` keeps the log readable.
        - Redact BEFORE serialising, not after.
    """
    raise NotImplementedError("Implement write_record. See tests/test_audit.py.")


def read_records(path: Path | None = None) -> list[AuditRecord]:
    """Read the log back. Used by the tests and by your demo.

    Being able to show a customer the audit trail for the thing you just did, live
    in a demo, is worth more than any slide about compliance.
    """
    raise NotImplementedError("Implement read_records. See tests/test_audit.py.")

"""RUBRIC: no tool call completes without an audit record.

These fail until you implement src/mcp_server/audit.py.

The tests check three properties a customer's compliance team will check:
the log is append-only, it records refusals as well as successes, and it does not
leak the personal data it exists to protect.
"""
from __future__ import annotations

import json

import pytest

from src.mcp_server.audit import AuditRecord, read_records, redact, write_record


def _record(**overrides) -> AuditRecord:
    base = dict(
        request_id="req-1",
        timestamp="2026-08-14T10:00:00Z",
        principal="reader",
        tool="list_shipments",
        outcome="success",
        arguments={"limit": 10},
        detail="",
    )
    base.update(overrides)
    return AuditRecord(**base)


def test_redact_replaces_pii_but_keeps_the_key(tmp_path):
    """Dropping the key hides that the call included an address at all."""
    out = redact({"address": "12 Main St", "limit": 10})
    assert "address" in out, "keep the key so a reviewer can see a value was present"
    assert out["address"] != "12 Main St"
    assert out["limit"] == 10, "non-PII values pass through unchanged"


def test_redact_reaches_nested_values():
    out = redact({"filter": {"recipient_name": "A. Patel"}, "limit": 5})
    assert out["filter"]["recipient_name"] != "A. Patel", (
        "a flat pass over top-level keys is the usual bug"
    )


def test_redact_does_not_mutate_the_caller_dict():
    original = {"address": "12 Main St"}
    redact(original)
    assert original["address"] == "12 Main St"


def test_write_then_read_round_trips(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_record(_record(), path)
    records = read_records(path)
    assert len(records) == 1
    assert records[0].tool == "list_shipments"


def test_log_is_append_only(tmp_path):
    """Three writes produce three lines. Nothing truncates."""
    path = tmp_path / "audit.jsonl"
    for i in range(3):
        write_record(_record(request_id=f"req-{i}"), path)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3, "the log was overwritten rather than appended"
    assert len(read_records(path)) == 3


def test_each_line_is_independently_parseable(tmp_path):
    """JSONL, so a partial write damages one record rather than the file."""
    path = tmp_path / "audit.jsonl"
    write_record(_record(request_id="a"), path)
    write_record(_record(request_id="b"), path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)


def test_records_carry_the_required_fields(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_record(_record(), path)
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    for field in ("request_id", "timestamp", "principal", "tool", "outcome"):
        assert field in payload, f"audit records must carry {field}"


def test_denials_are_recorded(tmp_path):
    """The interesting half of an audit log.

    A log containing only successes cannot answer "did anything try and fail",
    which is the question an incident review opens with.
    """
    path = tmp_path / "audit.jsonl"
    write_record(_record(outcome="denied", detail="missing shipments:write"), path)
    records = read_records(path)
    assert records[0].outcome == "denied"
    assert records[0].detail


def test_pii_never_reaches_the_file(tmp_path):
    """The control must not leak what it protects."""
    path = tmp_path / "audit.jsonl"
    write_record(_record(arguments={"address": "12 Main St, Unit 4"}), path)
    assert "12 Main St" not in path.read_text(encoding="utf-8")


def test_write_failure_propagates(tmp_path):
    """If the record cannot be written, the caller must find out.

    Swallowing this is how a system ends up performing actions nobody can account
    for. Ordering matters: audit first, act second.
    """
    unwritable = tmp_path / "no-such-dir" / "audit.jsonl"
    with pytest.raises(OSError):
        write_record(_record(), unwritable)

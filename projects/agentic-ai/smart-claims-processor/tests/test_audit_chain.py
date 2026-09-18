"""Audit log tamper evidence and retention.

Hashes used to be per entry only: deleting a whole line left every remaining
hash valid, and retention_days was never acted on.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.security import audit_log


@pytest.fixture
def audit_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path))
    audit_log._last_hash_cache.clear()
    yield tmp_path
    audit_log._last_hash_cache.clear()


def _write_three():
    for i in range(3):
        audit_log.log_agent_action(claim_id=f"CLM-{i}", agent_name="intake_agent", action="test")


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_an_untouched_log_verifies(audit_dir):
    _write_three()
    assert audit_log.verify_audit_chain(audit_dir) == []


def test_deleting_an_entry_breaks_the_chain(audit_dir):
    _write_three()
    path = next(audit_dir.glob("audit_*.ndjson"))
    lines = _lines(path)
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    problems = audit_log.verify_audit_chain(audit_dir)
    assert any("prev_hash does not match" in p for p in problems)


def test_editing_an_entry_breaks_its_hash(audit_dir):
    _write_three()
    path = next(audit_dir.glob("audit_*.ndjson"))
    lines = _lines(path)
    entry = json.loads(lines[1])
    entry["action"] = "something_else"
    lines[1] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert any("does not match its hash" in p for p in audit_log.verify_audit_chain(audit_dir))


def test_retention_deletes_only_expired_days_and_records_it(audit_dir):
    now = datetime.now(timezone.utc)
    old_day = (now - timedelta(days=2600)).strftime("%Y-%m-%d")
    kept_day = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    (audit_dir / f"audit_{old_day}.ndjson").write_text("{}\n", encoding="utf-8")
    (audit_dir / f"audit_{kept_day}.ndjson").write_text("{}\n", encoding="utf-8")

    deleted = audit_log.enforce_retention(now=now)

    assert deleted == [f"audit_{old_day}.ndjson"]
    assert (audit_dir / f"audit_{kept_day}.ndjson").exists()
    todays = _lines(next(p for p in audit_dir.glob("audit_*.ndjson") if p.name.endswith(now.strftime("%Y-%m-%d") + ".ndjson")))
    assert json.loads(todays[-1])["event_type"] == "RETENTION_PURGE"

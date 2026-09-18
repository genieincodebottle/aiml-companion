"""
Immutable audit log for insurance compliance (7-year retention).

Every agent action, HITL decision, and final outcome is recorded with:
- SHA-256 hash of the entry, chained to the previous entry's hash
  (prev_hash), so editing, deleting or reordering a line breaks the chain
  and verify_audit_chain() reports where
- Timestamp (UTC)
- Claim ID and agent name
- Input/output snapshots (PII-masked)
- Cost attribution

Log format: newline-delimited JSON (NDJSON), one file per UTC day. The first
entry of each day links to the last entry of the previous day's file.

Retention: enforce_retention() deletes daily files older than
security.audit_log.retention_days (7 years) and records the purge. It runs at
API startup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.config import get_security_config

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


GENESIS_HASH = "0" * 64

# Claims run in parallel threads; reading the last hash and appending the next
# entry must be one step, or two writers chain to the same parent.
_write_lock = threading.Lock()
_last_hash_cache: dict[str, str] = {}


def _hash_entry(entry: dict) -> str:
    """Hash of everything in the entry except the hash itself (prev_hash included)."""
    body = {k: v for k, v in entry.items() if k != "hash"}
    serialized = json.dumps(body, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _log_dir() -> Path:
    cfg = get_security_config()
    base = Path(cfg["audit_log"]["path"])
    base.mkdir(parents=True, exist_ok=True)
    return base


def _get_log_path(claim_id: str) -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _log_dir() / f"audit_{date_str}.ndjson"


def _last_hash_in(path: Path) -> Optional[str]:
    """Hash of the last well-formed entry in a file, or None."""
    if not path.exists():
        return None
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line).get("hash") or last
            except json.JSONDecodeError:
                continue
    return last


def _previous_file_hash(path: Path) -> str:
    """Last hash of the newest daily file before `path`, or the genesis hash."""
    earlier = sorted(p for p in path.parent.glob("audit_*.ndjson") if p.name < path.name)
    for candidate in reversed(earlier):
        found = _last_hash_in(candidate)
        if found:
            return found
    return GENESIS_HASH


def _write_entry(claim_id: str, entry: dict) -> str:
    """Chain, hash, append to the daily log file, return the entry's SHA-256 hash."""
    path = _get_log_path(claim_id)
    key = str(path)
    with _write_lock:
        prev = _last_hash_cache.get(key) or _last_hash_in(path) or _previous_file_hash(path)
        entry["prev_hash"] = prev
        entry_hash = _hash_entry(entry)
        entry["hash"] = entry_hash
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            _last_hash_cache[key] = entry_hash
        except Exception as e:
            logger.error(f"AUDIT LOG WRITE FAILED for {claim_id}: {e}")
    return entry_hash


def verify_audit_chain(base: Optional[Path] = None) -> list[str]:
    """Walk every daily file oldest-first and report chain breaks.

    Returns a list of problems (empty = intact). The first surviving file may
    start from a purged day, so its first link is not checked against a file
    that no longer exists; every other link is.
    """
    base = Path(base) if base else _log_dir()
    problems: list[str] = []
    expected_prev: Optional[str] = None
    for path in sorted(base.glob("audit_*.ndjson")):
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    problems.append(f"{path.name}:{lineno} is not valid JSON")
                    expected_prev = None
                    continue
                if "prev_hash" not in entry:
                    # Written before chaining existed; start the chain after it.
                    expected_prev = entry.get("hash")
                    continue
                if _hash_entry(entry) != entry.get("hash"):
                    problems.append(f"{path.name}:{lineno} content does not match its hash (edited)")
                if expected_prev is not None and entry["prev_hash"] != expected_prev:
                    problems.append(
                        f"{path.name}:{lineno} prev_hash does not match the previous entry "
                        "(an entry before it was deleted, inserted or reordered)"
                    )
                expected_prev = entry.get("hash")
    return problems


def enforce_retention(now: Optional[datetime] = None) -> list[str]:
    """Delete daily audit files older than security.audit_log.retention_days.

    Returns the deleted file names and records the purge in today's log, so
    the deletion itself is auditable.
    """
    cfg = get_security_config()["audit_log"]
    retention_days = int(cfg.get("retention_days", 2555))
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    deleted = []
    for path in sorted(_log_dir().glob("audit_*.ndjson")):
        day = path.stem.replace("audit_", "")
        if day < cutoff:
            path.unlink()
            _last_hash_cache.pop(str(path), None)
            deleted.append(path.name)
    if deleted:
        logger.info("Audit retention: deleted %d file(s) older than %s", len(deleted), cutoff)
        _write_entry("SYSTEM", {
            "timestamp": _now_iso(),
            "claim_id": "SYSTEM",
            "event_type": "RETENTION_PURGE",
            "retention_days": retention_days,
            "cutoff_date": cutoff,
            "deleted_files": deleted,
        })
    return deleted


def log_agent_action(
    claim_id: str,
    agent_name: str,
    action: str,
    input_summary: Optional[dict] = None,
    output_summary: Optional[dict] = None,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    duration_ms: int = 0,
    error: Optional[str] = None,
) -> str:
    """Record a single agent action. Returns the SHA-256 hash of the entry."""
    return _write_entry(claim_id, {
        "timestamp": _now_iso(),
        "claim_id": claim_id,
        "agent": agent_name,
        "action": action,
        "input": input_summary or {},
        "output": output_summary or {},
        "tokens_used": tokens_used,
        "cost_usd": cost_usd,
        "duration_ms": duration_ms,
        "error": error,
    })


def log_hitl_event(
    claim_id: str,
    event: str,
    priority: str,
    triggers: list[str],
    reviewer_id: Optional[str] = None,
    human_decision: Optional[str] = None,
    human_notes: Optional[str] = None,
    override_ai: bool = False,
) -> str:
    """Record HITL queue events and human decisions."""
    return _write_entry(claim_id, {
        "timestamp": _now_iso(),
        "claim_id": claim_id,
        "event_type": "HITL",
        "hitl_event": event,
        "priority": priority,
        "triggers": triggers,
        "reviewer_id": reviewer_id,
        "human_decision": human_decision,
        "human_notes": human_notes,
        "override_ai": override_ai,
    })


def log_final_decision(
    claim_id: str,
    decision: str,
    amount_usd: float,
    total_tokens: int,
    total_cost_usd: float,
    evaluation_score: Optional[float] = None,
    human_reviewed: bool = False,
) -> str:
    """Record the final claim decision for compliance audit trail."""
    return _write_entry(claim_id, {
        "timestamp": _now_iso(),
        "claim_id": claim_id,
        "event_type": "FINAL_DECISION",
        "decision": decision,
        "settlement_amount_usd": amount_usd,
        "total_tokens_used": total_tokens,
        "total_cost_usd": total_cost_usd,
        "evaluation_score": evaluation_score,
        "human_reviewed": human_reviewed,
    })


def get_claim_audit_trail(claim_id: str, days_back: int = 30) -> list[dict]:
    """Retrieve all audit entries for a claim ID. For review and compliance queries."""
    cfg = get_security_config()
    base = Path(cfg["audit_log"]["path"])
    entries = []
    if not base.exists():
        return entries

    for log_file in sorted(base.glob("audit_*.ndjson"), reverse=True)[:days_back]:
        try:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("claim_id") == claim_id:
                            entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

    return sorted(entries, key=lambda x: x.get("timestamp", ""))

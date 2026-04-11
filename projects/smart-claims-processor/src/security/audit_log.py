"""
Immutable audit log for insurance compliance (7-year retention).

Every agent action, HITL decision, and final outcome is recorded with:
- SHA-256 hash of the entry (tamper detection)
- Timestamp (UTC)
- Claim ID and agent name
- Input/output snapshots (PII-masked)
- Cost attribution

Log format: newline-delimited JSON (NDJSON) for easy streaming and parsing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import get_security_config

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_entry(entry: dict) -> str:
    serialized = json.dumps(entry, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _get_log_path(claim_id: str) -> Path:
    cfg = get_security_config()
    base = Path(cfg["audit_log"]["path"])
    base.mkdir(parents=True, exist_ok=True)
    # One file per day per claim for easy retention management
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return base / f"audit_{date_str}.ndjson"


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
    """
    Record a single agent action. Returns the SHA-256 hash of the entry.
    Entry is appended atomically to the daily audit log file.
    """
    entry = {
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
    }
    entry_hash = _hash_entry(entry)
    entry["hash"] = entry_hash

    log_path = _get_log_path(claim_id)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        # Audit failures must not crash the pipeline - but always alert
        logger.error(f"AUDIT LOG WRITE FAILED for {claim_id}: {e}")

    return entry_hash


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
    entry = {
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
    }
    entry_hash = _hash_entry(entry)
    entry["hash"] = entry_hash

    log_path = _get_log_path(claim_id)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.error(f"AUDIT LOG WRITE FAILED (HITL) for {claim_id}: {e}")

    return entry_hash


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
    entry = {
        "timestamp": _now_iso(),
        "claim_id": claim_id,
        "event_type": "FINAL_DECISION",
        "decision": decision,
        "settlement_amount_usd": amount_usd,
        "total_tokens_used": total_tokens,
        "total_cost_usd": total_cost_usd,
        "evaluation_score": evaluation_score,
        "human_reviewed": human_reviewed,
    }
    entry_hash = _hash_entry(entry)
    entry["hash"] = entry_hash

    log_path = _get_log_path(claim_id)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.error(f"AUDIT LOG WRITE FAILED (FINAL) for {claim_id}: {e}")

    return entry_hash


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

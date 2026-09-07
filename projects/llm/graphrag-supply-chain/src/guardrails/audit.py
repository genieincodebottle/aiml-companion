"""Append-only audit log.

Every question asked, every guardrail decision, every ingestion. Written as
JSON lines, one event per line, never rewritten.

WHY THIS EXISTS
===============
Not for compliance theatre. For three concrete things you cannot do without it:

  1. Answer "why did the system say that?" three weeks later. The retrieval,
     the strategy, the documents and the guardrail verdicts are all recorded, so
     a disputed answer can be reconstructed rather than argued about.
  2. Detect a poisoning attempt in hindsight. The injection scanner is a
     heuristic and will miss things. When a bad edge is eventually noticed, the
     log is what lets you find which ingestion introduced it and what else that
     run touched.
  3. Notice guardrails firing. A guardrail nobody monitors is a guardrail that
     silently stops working after a refactor.

WHY JSON LINES AND NOT A DATABASE
=================================
Because an audit log's most important property is that writing to it cannot
fail in a way that takes down the thing it is auditing, and its second most
important property is that it can be read by anything. A file opened in append
mode satisfies both. A production system ships these lines to a log pipeline;
the format does not change.

WHAT IS DELIBERATELY NOT LOGGED
===============================
Full document text, full answers, and anything the PII scanner flagged. The log
records that a secret was detected and its kind, never the secret. An audit log
that accumulates the sensitive data it was written to protect is a liability,
and this is a mistake that gets made constantly.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Appends from several threads (FastAPI workers) must not interleave
        # mid-line. A single line under the pipe buffer size is atomic on POSIX
        # but not guaranteed on Windows, so the lock is not optional here.
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "pid": os.getpid(),
            **fields,
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return record

    def tail(self, limit: int = 100, event: str | None = None) -> list[dict[str, Any]]:
        """Read recent events. Used by the API and the UI's guardrail panel."""
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        out: list[dict[str, Any]] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A partially written line at the tail is expected under
                # concurrent append. Skip it rather than failing the read.
                continue
            if event and record.get("event") != event:
                continue
            out.append(record)
            if len(out) >= limit:
                break
        return out

    def stats(self) -> dict[str, Any]:
        events = self.tail(limit=5000)
        by_event: dict[str, int] = {}
        blocked = 0
        warned = 0
        for record in events:
            by_event[record.get("event", "?")] = by_event.get(record.get("event", "?"), 0) + 1
            if record.get("blocked"):
                blocked += 1
            if record.get("warnings"):
                warned += 1
        return {
            "events_read": len(events),
            "by_event": by_event,
            "blocked": blocked,
            "with_warnings": warned,
        }

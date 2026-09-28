"""Durable, cross-repo memory - exposed to the worker as tools.

The Claude Agent SDK gives a worker *session* continuity; it does NOT give the
organization durable memory. That is something you architect, and here it is a tool
surface backed by a JSON file, scoped per job:

  * `record_memory(note, tags)`  - a worker writes a gotcha it learned (e.g. "adding
                                    the timezone import was required").
  * `recall_memory(query)`       - a LATER repo's worker retrieves those gotchas, so
                                    the fleet gets smarter as it works instead of
                                    re-learning the same lesson on every repository.

Persisting to disk (not hidden in-process state) is deliberate: every memory write is
an auditable event, which is what you want for a digital employee.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path


class MemoryStore:
    """One store per job. Notes are shared across all repos in that job."""

    def __init__(self, data_dir: Path, job_id: str) -> None:
        self._dir = data_dir / "memory"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{job_id}.json"
        # One store is shared by every concurrently-migrating repo in the job, and
        # record() runs on worker threads - guard the list + file flush.
        self._lock = threading.Lock()
        self._notes: list[dict] = []
        if self._path.exists():
            try:
                self._notes = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._notes = []

    def record(self, worktree=None, note: str = "", repo: str = "", tags: str = "") -> dict:
        entry = {"note": note, "repo": repo, "tags": tags, "ts": time.time()}
        with self._lock:
            self._notes.append(entry)
            self._flush()
            return {"recorded": True, "total_notes": len(self._notes)}

    def recall(self, worktree=None, query: str = "", top_k: int = 5) -> dict:
        q = {t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2}
        scored = []
        with self._lock:
            notes = list(self._notes)
        for n in notes:
            doc = {t for t in re.split(r"[^a-z0-9]+", f"{n['note']} {n['tags']}".lower()) if len(t) > 2}
            scored.append((len(q & doc), n))
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [n for score, n in scored if score > 0][:top_k]
        return {"query": query, "results": hits, "count": len(hits)}

    def _flush(self) -> None:
        try:
            self._path.write_text(json.dumps(self._notes, indent=2), encoding="utf-8")
        except OSError:  # pragma: no cover - memory must never break the run
            pass

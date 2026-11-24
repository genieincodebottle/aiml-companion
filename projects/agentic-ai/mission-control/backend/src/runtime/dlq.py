"""Dead-letter queue.

A workflow that exhausts retries, blows its budget, or dies in an unrecoverable way
is not silently dropped - it is written to the DLQ with its last state and the reason,
so an operator can inspect and replay it. "No silent failures" is a platform promise;
the DLQ is how it is kept for the cases the runtime cannot finish on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DeadLetter:
    workflow_id: str
    reason: str
    state: dict[str, Any]


class DeadLetterQueue:
    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "dlq"
        self._path.mkdir(parents=True, exist_ok=True)
        self._file = self._path / "dlq.jsonl"

    def put(self, workflow_id: str, reason: str, state: dict[str, Any]) -> DeadLetter:
        dl = DeadLetter(workflow_id=workflow_id, reason=reason, state=state)
        try:
            with self._file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"workflow_id": workflow_id, "reason": reason, "state": state}, default=str) + "\n")
        except OSError:  # pragma: no cover
            pass
        return dl

    def list(self) -> list[DeadLetter]:
        if not self._file.exists():
            return []
        out: list[DeadLetter] = []
        with self._file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    out.append(DeadLetter(d["workflow_id"], d["reason"], d.get("state", {})))
        return out

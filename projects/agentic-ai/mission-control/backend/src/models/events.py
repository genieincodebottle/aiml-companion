"""The event log - the source of truth for durable execution.

The runtime is event-sourced: a workflow's state is the fold of its event stream,
never a mutated object. This is what makes execution *durable* - to resume after a
crash we replay events up to the last checkpoint; to audit a case we read the log;
to debug we diff two event streams. Every plane (gateway, policy, runtime, HITL)
appends here, and every event is also an observability signal.

An `Event` is deliberately a flat, JSON-serializable record so the log is a plain
JSONL file that survives process death and is trivially shippable to Kafka later.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    # workflow lifecycle
    WORKFLOW_STARTED = "workflow_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    CHECKPOINT_WRITTEN = "checkpoint_written"
    WORKFLOW_RESUMED = "workflow_resumed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    # gateway
    GATEWAY_CALL = "gateway_call"
    CACHE_HIT = "cache_hit"
    BUDGET_EXCEEDED = "budget_exceeded"
    PROVIDER_FAILOVER = "provider_failover"
    # policy + guardrails
    POLICY_ALLOWED = "policy_allowed"
    POLICY_DENIED = "policy_denied"
    INJECTION_BLOCKED = "injection_blocked"
    # human-in-the-loop
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    MAKER_CHECKER_BLOCKED = "maker_checker_blocked"
    # reliability
    SAGA_COMPENSATED = "saga_compensated"
    DEAD_LETTERED = "dead_lettered"
    RETRY = "retry"


@dataclass
class Event:
    """One immutable fact in a workflow's history."""

    workflow_id: str
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    seq: int = 0

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d, default=str)


class EventLog:
    """Append-only event store for one workflow, mirrored to JSONL on disk.

    In production this is Kafka + an event-store DB; here a JSONL file is the
    identical contract (append, then fold) and lets the whole system survive a
    restart with zero external dependencies.
    """

    def __init__(self, workflow_id: str, data_dir: Path) -> None:
        self.workflow_id = workflow_id
        self.events: list[Event] = []
        self._sink = data_dir / "events"
        self._sink.mkdir(parents=True, exist_ok=True)
        self._path = self._sink / f"{workflow_id}.jsonl"

    def append(self, type: EventType, **payload: Any) -> Event:
        evt = Event(workflow_id=self.workflow_id, type=type, payload=payload, seq=len(self.events))
        self.events.append(evt)
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(evt.to_json() + "\n")
        except OSError:  # pragma: no cover - persistence must never break the run
            pass
        return evt

    def replay(self) -> list[Event]:
        """Load a prior workflow's events from disk (durable resume)."""
        if not self._path.exists():
            return []
        loaded: list[Event] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                loaded.append(
                    Event(
                        workflow_id=d["workflow_id"],
                        type=EventType(d["type"]),
                        payload=d.get("payload", {}),
                        ts=d.get("ts", 0.0),
                        seq=d.get("seq", 0),
                    )
                )
        self.events = loaded
        return loaded

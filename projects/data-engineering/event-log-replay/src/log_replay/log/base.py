"""The Log interface and the Consumer that sits on top of it.

Both backends (the file log and the Kafka log) implement `LogBackend`, and the
sinks only ever see a `Consumer`. That is what lets the identical sink code run
against both.

Kafka semantics kept on purpose
  * a partition is an append-only sequence, an offset is an index into it
  * records with the same key land in the same partition, in order
  * the group's committed offsets are stored apart from the data
    (like __consumer_offsets), and `commit` is explicit
  * `poll` advances the in-memory position, `commit` makes it durable, and a
    new consumer after a crash starts from the committed offset, not the position
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Record:
    partition: int
    offset: int
    key: str
    value: dict


def partition_for(key: str, n_partitions: int) -> int:
    """Stable key -> partition. Python's built-in hash() is salted per process,
    so the same key would land on a different partition on every run and the
    log would not be reproducible. crc32 is the same on every OS."""
    return zlib.crc32(key.encode("utf-8")) % n_partitions


class LogBackend(Protocol):
    n_partitions: int

    def append(self, partition: int, key: str, value: dict) -> int: ...
    def end_offset(self, partition: int) -> int: ...
    def read(self, partition: int, offset: int, n: int) -> list[Record]: ...
    def committed(self, group: str) -> dict[int, int]: ...
    def commit(self, group: str, offsets: dict[int, int]) -> None: ...
    def reset_group(self, group: str) -> None: ...


class Consumer:
    """One consumer process. Crashing means throwing this object away."""

    def __init__(self, log: LogBackend, group: str):
        self.log = log
        self.group = group
        committed = log.committed(group)
        self.position = {p: committed.get(p, 0) for p in range(log.n_partitions)}

    def seek(self, partition: int, offset: int) -> None:
        self.position[partition] = offset

    def poll(self, max_records: int) -> list[Record]:
        """Round robin across partitions, one record at a time, so a batch mixes
        partitions the way a real fetch does. Order inside a partition is kept."""
        out: list[Record] = []
        live = True
        while len(out) < max_records and live:
            live = False
            for p in range(self.log.n_partitions):
                if len(out) >= max_records:
                    break
                got = self.log.read(p, self.position[p], 1)
                if got:
                    out.extend(got)
                    self.position[p] += 1
                    live = True
        return out

    def commit(self) -> None:
        """Commit the current position of every partition (offset of the NEXT record)."""
        self.log.commit(self.group, dict(self.position))

"""File-backed log with Kafka semantics. No broker, no network.

    data/log/<topic>/partition-N.jsonl   one JSON record per line, offset = line index
    data/log/__consumer_offsets/<group>.jsonl
                                         append-only commit log per group, last
                                         line per partition wins

The commit log is append-only, like the compacted __consumer_offsets topic, so
a commit is one small write and a crash can never leave half a state file. One
file per group keeps parallel runs of different strategies from sharing a file.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .base import Record


class FileLog:
    def __init__(self, root: Path, topic: str, n_partitions: int, create: bool = False):
        self.n_partitions = n_partitions
        self.topic = topic
        self.dir = Path(root) / topic
        self.offsets_dir = Path(root) / "__consumer_offsets"
        if create:
            self.dir.mkdir(parents=True, exist_ok=True)
            for p in range(n_partitions):
                self._path(p).write_text("", encoding="utf-8", newline="\n")
            shutil.rmtree(self.offsets_dir, ignore_errors=True)
        self._parts: list[list[Record]] = [self._load(p) for p in range(n_partitions)]
        self._handles: dict = {}
        self._committed: dict[str, dict[int, int]] = {}

    def _path(self, p: int) -> Path:
        return self.dir / f"partition-{p}.jsonl"

    def _group_path(self, group: str) -> Path:
        return self.offsets_dir / f"{group}.jsonl"

    def _load(self, p: int) -> list[Record]:
        recs = []
        with open(self._path(p), encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                row = json.loads(line)
                recs.append(Record(p, i, row["key"], row["value"]))
        return recs

    # ------------------------------------------------------------ producing
    def append(self, partition: int, key: str, value: dict) -> int:
        fh = self._handles.get(partition)
        if fh is None:
            fh = open(self._path(partition), "a", encoding="utf-8", newline="\n")
            self._handles[partition] = fh
        offset = len(self._parts[partition])
        fh.write(json.dumps({"key": key, "value": value}, sort_keys=True) + "\n")
        self._parts[partition].append(Record(partition, offset, key, value))
        return offset

    def flush(self) -> None:
        for fh in self._handles.values():
            fh.close()
        self._handles = {}

    # -------------------------------------------------------------- reading
    def end_offset(self, partition: int) -> int:
        return len(self._parts[partition])

    def read(self, partition: int, offset: int, n: int) -> list[Record]:
        return self._parts[partition][offset: offset + n]

    # ------------------------------------------------------- group offsets
    def committed(self, group: str) -> dict[int, int]:
        if group not in self._committed:
            out: dict[int, int] = {}
            if self._group_path(group).exists():
                with open(self._group_path(group), encoding="utf-8") as fh:
                    for line in fh:
                        row = json.loads(line)
                        out[row["partition"]] = row["offset"]
            self._committed[group] = out
        return dict(self._committed[group])

    def commit(self, group: str, offsets: dict[int, int]) -> None:
        self.committed(group)  # load what is on disk before the first write
        lines = "".join(json.dumps({"partition": p, "offset": o}) + "\n"
                        for p, o in sorted(offsets.items()))
        fh = self._handles.get(("offsets", group))
        if fh is None:
            self.offsets_dir.mkdir(parents=True, exist_ok=True)
            fh = open(self._group_path(group), "a", encoding="utf-8", newline="\n")
            self._handles[("offsets", group)] = fh
        # One write call, flushed before returning: the commit either happened or it did not.
        fh.write(lines)
        fh.flush()
        self._committed[group].update(offsets)

    def reset_group(self, group: str) -> None:
        """Rewind to the earliest offset, like `kafka-consumer-groups --reset-offsets --to-earliest`."""
        self.commit(group, {p: 0 for p in range(self.n_partitions)})

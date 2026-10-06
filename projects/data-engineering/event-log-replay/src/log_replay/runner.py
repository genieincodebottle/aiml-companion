"""The consume loop, with crashes.

A crash is modelled the way Kafka sees it. The consumer object (its in-memory
position, its half-done batch) is thrown away and a new one starts from the
committed offset. The sink (the database) survives, because it is a separate
system. That asymmetry is the source of every delivery problem in this project.
"""
from __future__ import annotations

from dataclasses import dataclass

from .faults import Position
from .log.base import Consumer, LogBackend
from .sinks.strategies import Crash, Sink


@dataclass
class RunStats:
    crashes: int = 0
    polls: int = 0
    records_handled: int = 0     # records a sink began processing, repeats included
    redelivered: int = 0         # of those, records the sink had already been handed once


def run_consumer(log: LogBackend, sink: Sink, group: str, batch_size: int,
                 crashes: list[Position]) -> RunStats:
    pending = set(crashes)
    seen: set[Position] = set()
    stats = RunStats()
    while True:
        consumer = Consumer(log, group)      # a restarted process
        sink.on_start(consumer)
        try:
            while True:
                batch = consumer.poll(batch_size)
                if not batch:
                    return stats
                stats.polls += 1
                crash_at = next((i for i, r in enumerate(batch)
                                 if (r.partition, r.offset) in pending), None)
                handled = batch if crash_at is None else batch[: crash_at + 1]
                for r in handled:
                    pos = (r.partition, r.offset)
                    if pos in seen:
                        stats.redelivered += 1
                    seen.add(pos)
                stats.records_handled += len(handled)
                try:
                    sink.handle(consumer, batch, crash_at)
                except Crash:
                    r = batch[crash_at]
                    pending.discard((r.partition, r.offset))
                    stats.crashes += 1
                    raise
        except Crash:
            continue

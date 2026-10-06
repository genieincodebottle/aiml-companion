"""The seeded consumer crash schedule.

A crash position is a (partition, offset) pair, and the process dies the first
time its batch reaches that record. Each position fires once. Keying crashes to
log positions, not to "every Nth call", means every strategy and every batch
size sees exactly the same crashes, so their scoreboards are comparable.
"""
from __future__ import annotations

import random

Position = tuple[int, int]


def build_schedule(partition_lengths: list[int], every: int, jitter: float, seed: int,
                   pin: Position | None = None) -> list[Position]:
    """About one crash per `every` records in each partition, with +- jitter.

    `pin` replaces the nearest scheduled crash in its partition, so the featured
    account's own event is guaranteed to be the one the consumer dies on."""
    if every <= 0:
        return []
    rng = random.Random(seed + 1)
    spread = int(every * jitter)
    positions: list[Position] = []
    for partition, length in enumerate(partition_lengths):
        offset = rng.randint(every // 2, every)
        while offset < length:
            positions.append((partition, offset))
            offset += every + rng.randint(-spread, spread)
    if pin is not None:
        same = [x for x in positions if x[0] == pin[0]]
        if same:
            nearest = min(same, key=lambda x: abs(x[1] - pin[1]))
            positions.remove(nearest)
        positions.append(pin)
    return sorted(set(positions))

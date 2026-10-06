"""The four ways to write consumed events into the sink.

Each sink gets a polled batch and `crash_at`, the index of the record the
process dies on (or None). Where the crash lands relative to the write and the
offset commit is the whole difference between the strategies.

    naive               write, commit offset              crash after the write  -> duplicates
    commit_first        commit offset, write              crash before the write -> lost events
    dedup_separate_txn  txn 1 write, txn 2 record ids     crash between the txns -> duplicates
    atomic              write + ids + offset, one txn     crash before COMMIT    -> rolled back

A crash on record i means records 0..i of the batch were handled, so the
unfinished work is the prefix `batch[: i + 1]`. This is the same convention for
every strategy, which is what keeps the scoreboards comparable.
"""
from __future__ import annotations

from ..log.base import Consumer
from .sql import (FRESH, PROFILE, PROFILE_POLICIES, WALLET, apply_deltas, apply_profiles,
                  load_batch, mark_processed, select_fresh, store_offsets)


class Crash(Exception):
    """The consumer process died. Everything not committed is gone with it."""

    def __init__(self, index: int):
        super().__init__(f"consumer crashed on batch index {index}")
        self.index = index


class Sink:
    name = "base"

    def __init__(self, con, profile_policy: str):
        assert profile_policy in PROFILE_POLICIES, profile_policy
        self.con = con
        self.policy = profile_policy

    def on_start(self, consumer: Consumer) -> None:
        """Called each time a (re)started consumer comes up."""

    def handle(self, consumer: Consumer, batch, crash_at: int | None) -> None:
        raise NotImplementedError

    def rewind(self, log, group: str) -> None:
        """Send the consumer back to offset zero. The offsets live in the log's group store."""
        log.reset_group(group)

    def _write(self, records) -> None:
        """Plain writes. Each statement is its own implicit transaction."""
        load_batch(self.con, records)
        apply_deltas(self.con, WALLET)
        apply_profiles(self.con, PROFILE, self.policy)


class NaiveSink(Sink):
    """Write the batch, then commit. Gives at-least-once and nothing else."""
    name = "naive"

    def handle(self, consumer, batch, crash_at):
        done = batch if crash_at is None else batch[: crash_at + 1]
        self._write(done)
        if crash_at is not None:
            raise Crash(crash_at)  # the write happened, the commit did not
        consumer.commit()


class CommitFirstSink(Sink):
    """Commit, then write. At-most-once: a crash after the commit loses the rest of the batch."""
    name = "commit_first"

    def handle(self, consumer, batch, crash_at):
        consumer.commit()
        if crash_at is not None:
            # dies before writing record crash_at; it and everything after it is never written
            self._write(batch[:crash_at])
            raise Crash(crash_at)
        self._write(batch)


class DedupSeparateTxnSink(Sink):
    """Dedup on event_id, but the processed-id insert is its OWN transaction.

    Transaction 1 applies the effects of ids not yet recorded. Transaction 2
    records the ids. Producer retries are caught, because the earlier copy's id
    is already recorded. A crash between the two leaves the effects committed
    and the ids missing, so the redelivered batch passes the dedup check and is
    applied again. Swap the order and the same crash loses events instead.
    Two transactions can only choose which way to be wrong."""
    name = "dedup_separate_txn"

    def handle(self, consumer, batch, crash_at):
        con = self.con
        done = batch if crash_at is None else batch[: crash_at + 1]
        load_batch(con, done)
        con.execute("BEGIN")
        select_fresh(con)
        apply_deltas(con, FRESH)
        apply_profiles(con, PROFILE, self.policy)
        con.execute("COMMIT")
        if crash_at is not None:
            raise Crash(crash_at)  # effects are durable, the ids are not
        con.execute("BEGIN")
        mark_processed(con)
        con.execute("COMMIT")
        consumer.commit()


class AtomicSink(Sink):
    """Effect, dedup record and consumer offset in ONE transaction.

    The offset is stored in the sink, not committed to the log, so there is no
    second system to disagree with. On start the consumer seeks to the offset
    the sink says it reached. A crash before COMMIT rolls all three back, so the
    batch is redelivered into a sink that has no trace of it."""
    name = "atomic"

    def on_start(self, consumer):
        rows = self.con.execute(
            "SELECT log_partition, next_offset FROM consumer_offsets").fetchall()
        for partition, next_offset in rows:
            consumer.seek(partition, next_offset)

    def rewind(self, log, group):
        self.con.execute("UPDATE consumer_offsets SET next_offset = 0")

    def stage(self, con) -> None:
        select_fresh(con)

    def mark(self, con) -> None:
        mark_processed(con)

    def handle(self, consumer, batch, crash_at):
        con = self.con
        done = batch if crash_at is None else batch[: crash_at + 1]
        load_batch(con, done)
        con.execute("BEGIN")
        try:
            self.stage(con)
            apply_deltas(con, FRESH)
            apply_profiles(con, PROFILE, self.policy)
            self.mark(con)
            store_offsets(con)
            if crash_at is not None:
                raise Crash(crash_at)  # all of the above is uncommitted and dies with the process
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise


class AtomicOffsetOnlySink(AtomicSink):
    """Atomic offsets WITHOUT the event_id table. Ablation, not a recommendation.

    Offset and effect still commit together, so a crash never double-applies or
    loses a record. But the offset says nothing about a producer retry, which is
    a second record with the same event_id at a later offset. That one is applied
    twice. Exactly-once processing of the log is not exactly-once effect of the event."""
    name = "atomic_offset_only"

    def stage(self, con) -> None:
        con.execute(f"CREATE OR REPLACE TEMP TABLE fresh AS {WALLET}")

    def mark(self, con) -> None:
        """No event ids recorded."""


SINKS = {
    "naive": NaiveSink,
    "commit_first": CommitFirstSink,
    "dedup_separate_txn": DedupSeparateTxnSink,
    "atomic": AtomicSink,
    "atomic_offset_only": AtomicOffsetOnlySink,
}

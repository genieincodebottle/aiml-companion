"""Kafka backend (Redpanda or Apache Kafka) behind the same LogBackend interface.

Needs `pip install -r requirements-kafka.txt` and a broker (docker-compose.yml).
The sinks run unchanged against it. Three details keep the numbers equal to the
file backend.

  * Records are produced to an explicit partition chosen by `partition_for`, so
    the layout matches the file log. Kafka's own partitioner would hash with
    murmur2 and shuffle which account lands where.
  * Auto commit is OFF everywhere. The only commits are the ones the sink asks for.
  * Reads use assign + seek, not subscribe. A group rebalance would move
    partitions between readers mid-run and make the crash schedule meaningless.
"""
from __future__ import annotations

import json
import time

from confluent_kafka import Consumer as KafkaConsumer
from confluent_kafka import Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic

from .base import Record

FETCH = 2000          # records read ahead per partition
TIMEOUT = 20.0        # seconds before a broker call is treated as failed


class KafkaLog:
    def __init__(self, bootstrap: str, topic: str, n_partitions: int):
        self.bootstrap = bootstrap
        self.topic = topic
        self.n_partitions = n_partitions
        self._admin_reader = self._consumer("log-replay-meta")
        self._readers: dict[int, KafkaConsumer] = {}
        self._next: dict[int, int] = {}
        self._groups: dict[str, KafkaConsumer] = {}
        self._producer: Producer | None = None
        self._cache: dict[int, tuple[int, list[Record]]] = {}
        self._errors: list[str] = []

    def _consumer(self, group: str) -> KafkaConsumer:
        return KafkaConsumer({
            "bootstrap.servers": self.bootstrap, "group.id": group,
            "enable.auto.commit": False, "auto.offset.reset": "earliest",
        })

    # ------------------------------------------------------------ producing
    def recreate_topic(self) -> None:
        admin = AdminClient({"bootstrap.servers": self.bootstrap})
        if self.topic in admin.list_topics(timeout=TIMEOUT).topics:
            for fut in admin.delete_topics([self.topic], operation_timeout=30).values():
                fut.result()
            deadline = time.time() + TIMEOUT
            while self.topic in admin.list_topics(timeout=TIMEOUT).topics and time.time() < deadline:
                time.sleep(0.5)
        for fut in admin.create_topics(
                [NewTopic(self.topic, num_partitions=self.n_partitions, replication_factor=1)]).values():
            fut.result()
        self._cache.clear()

    def append(self, partition: int, key: str, value: dict) -> int:
        if self._producer is None:
            self._producer = Producer({"bootstrap.servers": self.bootstrap,
                                       "enable.idempotence": True, "linger.ms": 20})
        self._producer.produce(self.topic, key=key.encode(), value=json.dumps(value, sort_keys=True).encode(),
                               partition=partition, on_delivery=self._on_delivery)
        self._producer.poll(0)
        return -1  # the broker assigns the offset; it is known only after the flush

    def _on_delivery(self, err, _msg) -> None:
        if err is not None:
            self._errors.append(str(err))

    def flush(self) -> None:
        if self._producer is not None:
            self._producer.flush(TIMEOUT * 3)
        if self._errors:
            raise RuntimeError(f"{len(self._errors)} produce errors, first: {self._errors[0]}")

    # -------------------------------------------------------------- reading
    def end_offset(self, partition: int) -> int:
        _, high = self._admin_reader.get_watermark_offsets(
            TopicPartition(self.topic, partition), timeout=TIMEOUT)
        return high

    def _reader(self, partition: int) -> KafkaConsumer:
        """One assigned reader per partition, so reads on one partition never see another's messages."""
        if partition not in self._readers:
            c = self._consumer("log-replay-reader")
            c.assign([TopicPartition(self.topic, partition, 0)])
            self._readers[partition] = c
            self._next[partition] = 0
        return self._readers[partition]

    def _fetch(self, partition: int, offset: int) -> list[Record]:
        want = min(FETCH, self.end_offset(partition) - offset)
        if want <= 0:
            return []
        reader = self._reader(partition)
        if self._next[partition] != offset:
            reader.seek(TopicPartition(self.topic, partition, offset))
        out: list[Record] = []
        deadline = time.time() + TIMEOUT
        while len(out) < want and time.time() < deadline:
            for m in reader.consume(num_messages=want - len(out), timeout=1.0):
                if m.error():
                    raise RuntimeError(str(m.error()))
                out.append(Record(partition, m.offset(), m.key().decode(), json.loads(m.value())))
        if len(out) < want:
            raise RuntimeError(f"partition {partition}: read {len(out)} of {want} records before the timeout")
        self._next[partition] = offset + len(out)
        return out

    def read(self, partition: int, offset: int, n: int) -> list[Record]:
        start, recs = self._cache.get(partition, (0, []))
        if not (start <= offset < start + len(recs)):
            recs = self._fetch(partition, offset)
            start = offset
            self._cache[partition] = (start, recs)
        i = offset - start
        return recs[i: i + n]

    # ------------------------------------------------------- group offsets
    def _group(self, group: str) -> KafkaConsumer:
        if group not in self._groups:
            self._groups[group] = self._consumer(group)
        return self._groups[group]

    def committed(self, group: str) -> dict[int, int]:
        tps = [TopicPartition(self.topic, p) for p in range(self.n_partitions)]
        out = {}
        for tp in self._group(group).committed(tps, timeout=TIMEOUT):
            if tp.offset >= 0:
                out[tp.partition] = tp.offset
        return out

    def commit(self, group: str, offsets: dict[int, int]) -> None:
        tps = [TopicPartition(self.topic, p, o) for p, o in sorted(offsets.items())]
        self._group(group).commit(offsets=tps, asynchronous=False)

    def reset_group(self, group: str) -> None:
        self.commit(group, {p: 0 for p in range(self.n_partitions)})

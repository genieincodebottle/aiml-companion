"""The sink schema and the SQL every strategy shares.

Strategies differ in WHEN they run this SQL, in how many transactions, and
whether they dedup first. They never differ in the arithmetic, so a difference
in the scoreboard is always a difference in delivery handling, not in sums.

Everything is set-based. DuckDB is a columnar engine, so one statement over a
batch costs about the same as one statement over a single row, and a
row-at-a-time consumer would spend its time in statement overhead.
"""
from __future__ import annotations

import duckdb

SCHEMA = """
CREATE TABLE balances (
    account_id    INTEGER PRIMARY KEY,
    balance_cents BIGINT NOT NULL
);
CREATE TABLE profiles (
    account_id        INTEGER PRIMARY KEY,
    tier              VARCHAR NOT NULL,
    daily_limit_cents BIGINT NOT NULL,
    event_time        BIGINT NOT NULL,
    version           INTEGER NOT NULL
);
-- effect-level dedup record: one row per event_id already applied
CREATE TABLE processed_events (
    event_id      VARCHAR PRIMARY KEY,
    log_partition INTEGER NOT NULL,
    log_offset    BIGINT NOT NULL
);
-- consumer position stored WITH the data, so offset and effect commit together
CREATE TABLE consumer_offsets (
    log_partition INTEGER PRIMARY KEY,
    next_offset   BIGINT NOT NULL
);
-- Instrumentation only. One row per time a wallet event changed a balance. No
-- strategy reads it to decide anything; the scorer reads it to say which events
-- were lost or applied twice, because a summed balance cannot say that.
CREATE TABLE effect_audit (event_id VARCHAR NOT NULL);

-- The polled batch, one row per record. Rewritten before every write.
CREATE TEMP TABLE batch (
    seq               BIGINT,
    log_partition     INTEGER,
    log_offset        BIGINT,
    event_id          VARCHAR,
    type              VARCHAR,
    account_id        INTEGER,
    delta_cents       BIGINT,
    tier              VARCHAR,
    daily_limit_cents BIGINT,
    event_time        BIGINT,
    version           INTEGER
);
"""

PROFILE_POLICIES = ("last_arrived_wins", "time_only_guard", "guarded_upsert")

WALLET = "SELECT event_id, account_id, delta_cents FROM batch WHERE type = 'wallet_txn'"
PROFILE = "SELECT * FROM batch WHERE type = 'profile_updated'"
FRESH = "SELECT * FROM fresh"

_COLS = 11


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    # Batches are tens of rows. Thread hand-off costs more than it saves, and one
    # thread also makes the engine's behaviour the same on every machine.
    con.execute("SET threads = 1")
    con.execute(SCHEMA)
    return con


def delta_of(value: dict) -> int:
    sign = 1 if value["kind"] == "deposit" else -1
    return sign * value["amount_cents"]


def load_batch(con, records) -> None:
    """Replace the `batch` table with the polled records.

    `seq` is arrival order inside the batch. Two updates for one account always
    sit in one partition, so seq order is the order the log delivered them in."""
    con.execute("DELETE FROM batch")
    if not records:
        return
    cols: list[list] = [[] for _ in range(_COLS)]
    for seq, r in enumerate(records):
        v = r.value
        wallet = v["type"] == "wallet_txn"
        row = (seq, r.partition, r.offset, v["event_id"], v["type"], v["account_id"],
               delta_of(v) if wallet else None,
               None if wallet else v["tier"], None if wallet else v["daily_limit_cents"],
               None if wallet else v["event_time"], None if wallet else v["version"])
        for col, x in zip(cols, row):
            col.append(x)
    # One list parameter per column, zipped by unnest. A VALUES list with one `?`
    # per cell takes four times as long to prepare at batch sizes in the hundreds.
    con.execute("INSERT INTO batch SELECT " + ",".join(["unnest(?)"] * _COLS), cols)


def apply_deltas(con, src: str) -> None:
    """Add every delta row in `src` to its balance, and audit each application.

    This is the statement that cannot be made idempotent. `balance + delta`
    run twice is wrong however the row is keyed. Only NOT running it twice,
    decided in the same transaction, fixes that."""
    con.execute(f"INSERT INTO effect_audit SELECT event_id FROM ({src})")
    con.execute(
        f"""INSERT INTO balances
            SELECT account_id, SUM(delta_cents) FROM ({src}) GROUP BY account_id
            ON CONFLICT (account_id) DO UPDATE
            SET balance_cents = balances.balance_cents + excluded.balance_cents""")


# Which update wins inside one batch, per policy. Across batches the ON CONFLICT
# guard decides. Using the same rule in both places keeps batch size from
# changing the answer.
_BATCH_WINNER = {
    "last_arrived_wins": "seq DESC",
    "time_only_guard": "event_time DESC, seq ASC",
    "guarded_upsert": "event_time DESC, version DESC",
}
_UPSERT_GUARD = {
    "last_arrived_wins": "",
    "time_only_guard": "WHERE excluded.event_time > profiles.event_time",
    "guarded_upsert": ("WHERE (excluded.event_time, excluded.version) "
                       "> (profiles.event_time, profiles.version)"),
}


def apply_profiles(con, src: str, policy: str) -> None:
    """Upsert the state rows in `src` (columns of `batch`) under a profile policy."""
    winner, guard = _BATCH_WINNER[policy], _UPSERT_GUARD[policy]
    con.execute(
        f"""INSERT INTO profiles
            SELECT account_id, tier, daily_limit_cents, event_time, version
            FROM ({src})
            QUALIFY row_number() OVER (PARTITION BY account_id ORDER BY {winner}) = 1
            ON CONFLICT (account_id) DO UPDATE SET
                tier = excluded.tier, daily_limit_cents = excluded.daily_limit_cents,
                event_time = excluded.event_time, version = excluded.version
            {guard}""")


def select_fresh(con) -> None:
    """Stage the wallet events that have not been applied yet.

    NOT IN drops ids already recorded in the sink, row_number drops the
    producer's own retry when both copies sit in this batch."""
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE fresh AS
            SELECT * FROM ({WALLET})
            WHERE event_id NOT IN (SELECT event_id FROM processed_events)
            QUALIFY row_number() OVER (PARTITION BY event_id) = 1""")


def mark_processed(con) -> None:
    con.execute(
        """INSERT INTO processed_events
           SELECT event_id, log_partition, log_offset FROM batch
           QUALIFY row_number() OVER (PARTITION BY event_id ORDER BY seq) = 1
           ON CONFLICT DO NOTHING""")


def store_offsets(con) -> None:
    con.execute(
        """INSERT INTO consumer_offsets
           SELECT log_partition, MAX(log_offset) + 1 FROM batch GROUP BY log_partition
           ON CONFLICT (log_partition) DO UPDATE SET next_offset = excluded.next_offset""")

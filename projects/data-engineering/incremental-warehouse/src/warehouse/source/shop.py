"""The OLTP source. A SQLite database that a simulated shop mutates day by day.

The pipeline gets a connection to this database and nothing else. It never sees
the event list or the answer key.

`advance_to(day)` puts the database in the state it had at the end of that day.
Running forward applies one batch. Asking for an earlier day, or for a day after
a break was injected, rebuilds from scratch by replaying the batches. A real
OLTP table keeps only current rows, so a real backfill cannot do this. Here it
lets the repo prove rerun and backfill against the exact state a daily run saw.
"""
import sqlite3

SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    region      TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE order_lines (
    order_id         INTEGER NOT NULL,
    line_no          INTEGER NOT NULL,
    sku              TEXT NOT NULL,
    quantity         INTEGER NOT NULL,
    unit_price_cents INTEGER NOT NULL,
    PRIMARY KEY (order_id, line_no)
);
CREATE INDEX idx_orders_updated ON orders (updated_at);
CREATE INDEX idx_customers_updated ON customers (updated_at);
"""

BREAK_KINDS = ("rename", "nulls", "volume")


class Shop:
    def __init__(self, path, days, break_options=None):
        """`days` is the batch list from `generate_world`. `path` may be ':memory:'."""
        self.path = str(path)
        self.days = days
        self.break_options = break_options or {}
        self.conn = None
        self.current_day = 0
        self.tainted = False
        self._reset()

    def _reset(self):
        if self.conn is not None:
            self.conn.close()
        self.conn = sqlite3.connect(self.path)
        for (name,) in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
            self.conn.execute(f"DROP TABLE {name}")
        self.conn.executescript(SCHEMA)
        self.current_day = 0
        self.tainted = False

    def advance_to(self, day, break_kind=None):
        """Bring the database to the end of `day`, optionally with a break injected on it."""
        if day == self.current_day and not break_kind and not self.tainted:
            return
        if day < self.current_day or self.tainted or day == self.current_day:
            self._reset()
        while self.current_day < day:
            nxt = self.current_day + 1
            self._apply(nxt, break_kind if nxt == day else None)
            self.current_day = nxt
        self.tainted = bool(break_kind)

    def _apply(self, day, break_kind):
        batch = self.days[day - 1]
        events = batch["events"]
        if break_kind == "volume":
            # An upstream job wrote only a fraction of the day's rows.
            keep = self.break_options["volume_keep_every"]
            events = [e for i, e in enumerate(events) if i % keep == 0]
        with self.conn:
            for ev in events:
                self._apply_event(ev)
            if break_kind == "rename":
                # A migration renamed a column. SELECT * now returns a new name.
                self.conn.execute("ALTER TABLE orders RENAME COLUMN status TO order_status")
            elif break_kind == "nulls":
                # A bad deploy stopped writing customer_id on part of the day's rows.
                mod = self.break_options["null_modulus"]
                self.conn.execute(
                    "UPDATE orders SET customer_id = NULL "
                    "WHERE updated_at >= ? AND order_id % ? < 2",
                    (batch["date"], mod))

    def _apply_event(self, ev):
        c, op = self.conn, ev["op"]
        if op == "insert_customer":
            c.execute("INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
                      (ev["customer_id"], ev["name"], ev["email"], ev["region"], ev["ts"]))
        elif op == "update_customer":
            if "region" in ev:
                c.execute("UPDATE customers SET region = ?, updated_at = ? WHERE customer_id = ?",
                          (ev["region"], ev["ts"], ev["customer_id"]))
            else:
                c.execute("UPDATE customers SET email = ?, updated_at = ? WHERE customer_id = ?",
                          (ev["email"], ev["ts"], ev["customer_id"]))
        elif op == "insert_order":
            c.execute("INSERT INTO orders VALUES (?, ?, 'placed', ?, ?)",
                      (ev["order_id"], ev["customer_id"], ev["ts"], ev["ts"]))
            c.executemany(
                "INSERT INTO order_lines VALUES (?, ?, ?, ?, ?)",
                [(ev["order_id"], l["line_no"], l["sku"], l["quantity"], l["unit_price_cents"])
                 for l in ev["lines"]])
        elif op == "update_order":
            c.execute("UPDATE orders SET status = ?, updated_at = ? WHERE order_id = ?",
                      (ev["status"], ev["ts"], ev["order_id"]))
        elif op == "delete_order":
            c.execute("DELETE FROM order_lines WHERE order_id = ?", (ev["order_id"],))
            c.execute("DELETE FROM orders WHERE order_id = ?", (ev["order_id"],))
        else:
            raise ValueError(f"unknown event op {op!r}")

    def counts(self):
        """Row counts per table, for the volume table in the report."""
        return {t: self.conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("customers", "orders", "order_lines")}

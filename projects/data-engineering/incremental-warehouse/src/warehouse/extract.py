"""Incremental extraction from the OLTP source.

The window for a logical date is `[midnight - lookback, next midnight)` on
`updated_at`. A strict watermark is the same query with lookback 0. The
lookback is the fix for late commits, and it makes the extract re-read some
rows it already landed. Staging removes those repeats by `(key, updated_at)`.

Extracts use `SELECT *` on purpose. If the source renames a column, the new name
is landed as it arrives and the contract check in `check_raw` catches it. An
explicit column list would crash the extract instead and hide what changed.
"""
from datetime import datetime, timedelta

import pyarrow as pa

_TYPES = {"INTEGER": pa.int64(), "TEXT": pa.string()}


def window(run_date, lookback_minutes):
    """(lo, hi) as sortable text. `hi` is exclusive."""
    midnight = datetime(run_date.year, run_date.month, run_date.day)
    fmt = "%Y-%m-%d %H:%M:%S"
    return ((midnight - timedelta(minutes=lookback_minutes)).strftime(fmt),
            (midnight + timedelta(days=1)).strftime(fmt))


def fetch(conn, sql, params, table):
    """Run a query and return an Arrow table typed from the source column types."""
    declared = {r[1]: r[2] for r in conn.execute(f"PRAGMA table_info({table})")}
    cur = conn.execute(sql, params)
    names = [d[0] for d in cur.description]
    rows = cur.fetchall()
    columns = {n: pa.array([r[i] for r in rows], type=_TYPES[declared[n]])
               for i, n in enumerate(names)}
    return pa.table(columns)


def extract_orders(conn, run_date, opts):
    lo, hi = window(run_date, opts.lookback_minutes)
    out = {
        "orders": fetch(conn,
                        "SELECT * FROM orders WHERE updated_at >= ? AND updated_at < ? "
                        "ORDER BY order_id", (lo, hi), "orders"),
        "order_lines": fetch(conn,
                             "SELECT l.* FROM order_lines l JOIN orders o USING (order_id) "
                             "WHERE o.updated_at >= ? AND o.updated_at < ? "
                             "ORDER BY l.order_id, l.line_no", (lo, hi), "order_lines"),
    }
    if opts.detect_deletes:
        # Every key that exists right now. A key missing from this list, that was
        # present before, was hard-deleted. updated_at can never show that.
        out["order_keys"] = fetch(conn, "SELECT order_id FROM orders ORDER BY order_id",
                                  (), "orders")
    return out


def extract_customers(conn, run_date, opts):
    lo, hi = window(run_date, opts.lookback_minutes)
    return {"customers": fetch(conn,
                               "SELECT * FROM customers WHERE updated_at >= ? "
                               "AND updated_at < ? ORDER BY customer_id", (lo, hi),
                               "customers")}

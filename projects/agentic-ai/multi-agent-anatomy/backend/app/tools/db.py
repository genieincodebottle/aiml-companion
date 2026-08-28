"""Seeded SQLite. Real rows, real queries, no mocked tool responses.

Two tenants exist on purpose. The cross-tenant test in tests/test_tenant_scoping.py
needs a second retailer's data to actually be present, otherwise it proves
nothing: an empty result from an empty table is not isolation.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("ANATOMY_DB", Path(__file__).resolve().parents[2] / "data" / "anatomy.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    tier        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    order_id     TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    customer_id  TEXT NOT NULL,
    item         TEXT NOT NULL,
    amount_usd   REAL NOT NULL,
    currency     TEXT NOT NULL DEFAULT 'USD',
    status       TEXT NOT NULL,
    placed_at    TEXT NOT NULL,
    delivered_at TEXT
);
CREATE TABLE IF NOT EXISTS shipments (
    shipment_id  TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    order_id     TEXT NOT NULL,
    courier      TEXT NOT NULL,
    tracking     TEXT NOT NULL,
    status       TEXT NOT NULL,
    eta          TEXT,
    last_scan_at TEXT
);
CREATE TABLE IF NOT EXISTS saga_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL,
    step        TEXT NOT NULL,
    direction   TEXT NOT NULL,
    payload     TEXT NOT NULL,
    at          TEXT NOT NULL
);
"""

SEED_CUSTOMERS = [
    ("cust-1001", "tenant-northwind", "Priya Raman", "priya@example.com", "gold"),
    ("cust-1002", "tenant-northwind", "Sam Okafor", "sam@example.com", "standard"),
    # A second retailer. Its rows must never appear in tenant-northwind results.
    ("cust-9001", "tenant-contoso", "Lee Nakamura", "lee@example.com", "gold"),
]

SEED_ORDERS = [
    ("ORD-4412", "tenant-northwind", "cust-1001", "Aeron desk chair", 743.00, "USD",
     "delivered", "2026-07-02T09:14:00Z", "2026-07-08T16:02:00Z"),
    ("ORD-4488", "tenant-northwind", "cust-1001", "Standing desk mat", 89.50, "USD",
     "in_transit", "2026-07-28T11:40:00Z", None),
    ("ORD-4501", "tenant-northwind", "cust-1002", "Monitor arm", 129.00, "USD",
     "processing", "2026-08-01T08:05:00Z", None),
    ("ORD-7777", "tenant-contoso", "cust-9001", "Espresso grinder", 410.00, "USD",
     "delivered", "2026-06-11T10:00:00Z", "2026-06-15T12:30:00Z"),
]

SEED_SHIPMENTS = [
    ("SHP-8801", "tenant-northwind", "ORD-4412", "Meridian Express", "MX-99120041",
     "delivered", "2026-07-08", "2026-07-08T16:02:00Z"),
    ("SHP-8842", "tenant-northwind", "ORD-4488", "Meridian Express", "MX-99120552",
     "in_transit", "2026-08-06", "2026-08-03T22:10:00Z"),
    ("SHP-9901", "tenant-contoso", "ORD-7777", "Northline", "NL-2210", "delivered",
     "2026-06-15", "2026-06-15T12:30:00Z"),
]


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(force: bool = False) -> None:
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        if force:
            for table in ("customers", "orders", "shipments", "saga_log"):
                conn.execute(f"DELETE FROM {table}")
        if not conn.execute("SELECT 1 FROM customers LIMIT 1").fetchone():
            conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", SEED_CUSTOMERS)
            conn.executemany(
                "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?)", SEED_ORDERS
            )
            conn.executemany(
                "INSERT INTO shipments VALUES (?,?,?,?,?,?,?,?)", SEED_SHIPMENTS
            )
        conn.commit()
    finally:
        conn.close()

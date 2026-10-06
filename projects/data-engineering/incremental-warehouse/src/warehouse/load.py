"""Partition-level loading of the fact tables.

A fact table is split by `order_date`. A run recomputes the partitions it
touched and replaces them, never appends. A partition can be touched by more than
the day it is named after, because an order changes status, is deleted, or its
customer moves for days after it was placed.
"""


def plan_touched_dates(conn, partition_date, detect_deletes, scd_type):
    """Order dates this run must recompute. Returns a sorted list of dates.

    Three sources, each one a way an old partition goes stale.
      1. orders that arrived in this run's extract (new, or changed)
      2. orders already in the fact that staging now marks as deleted
      3. orders of customers whose dimension rows changed in this run
    """
    conn.execute("CREATE OR REPLACE TABLE build.touched_dates (order_date DATE)")
    conn.execute(
        "INSERT INTO build.touched_dates "
        "SELECT DISTINCT CAST(created_at AS TIMESTAMP)::DATE FROM raw_orders "
        "WHERE extract_date = ?", [partition_date])
    if detect_deletes and _has(conn, "fct_order"):
        conn.execute(
            "INSERT INTO build.touched_dates "
            "SELECT DISTINCT f.order_date FROM build.fct_order f "
            "JOIN build.stg_orders s USING (order_id) WHERE s.is_deleted")
    first_change = _changed_dimension_rows(conn)
    if first_change:
        conn.execute("CREATE OR REPLACE TEMP TABLE changed_customers "
                     "(customer_id BIGINT, valid_from DATE)")
        conn.executemany("INSERT INTO changed_customers VALUES (?, ?)", first_change)
        # Type 2 only changes orders from the new version's start date onwards.
        # Type 1 rewrites the customer's whole history.
        cutoff = "o.order_date >= c.valid_from" if scd_type == 2 else "TRUE"
        conn.execute(
            "INSERT INTO build.touched_dates "
            "SELECT DISTINCT o.order_date FROM build.stg_orders o "
            f"JOIN changed_customers c USING (customer_id) WHERE {cutoff}")
    conn.execute("CREATE OR REPLACE TABLE build.touched_dates AS "
                 "SELECT DISTINCT order_date FROM build.touched_dates ORDER BY order_date")
    return [d for (d,) in conn.execute("SELECT order_date FROM build.touched_dates").fetchall()]


def _has(conn, table):
    return conn.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'build' AND table_name = ?", [table]).fetchone()[0] == 1


def _changed_dimension_rows(conn):
    """(customer_id, earliest changed valid_from) for dimension rows new since publish."""
    published = conn.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'published' AND table_name = 'dim_customer_scd2'").fetchone()[0]
    previous = ("SELECT customer_id, region, valid_from FROM published.dim_customer_scd2"
                if published else
                "SELECT NULL::BIGINT AS customer_id, NULL::VARCHAR AS region, "
                "NULL::DATE AS valid_from WHERE FALSE")
    return conn.execute(
        "SELECT customer_id, min(valid_from) FROM ("
        "  SELECT customer_id, region, valid_from FROM build.dim_customer_scd2"
        f"  EXCEPT {previous}) GROUP BY customer_id ORDER BY customer_id").fetchall()


def load_partitions(conn, table, model_sql, mode, partition_date):
    """Replace the touched partitions of one fact.

    `overwrite` deletes the touched order dates and inserts the recomputed rows,
    so a rerun gives the same table. `append` is the naive variant. It inserts
    the rows of the orders in this run's extract and deletes nothing, so every
    version of an order stays and a rerun adds the whole day again.
    """
    conn.execute(f"CREATE OR REPLACE TEMP TABLE new_{table} AS {model_sql}")
    if not _has(conn, table):
        conn.execute(f"CREATE TABLE build.{table} AS SELECT * FROM new_{table} WHERE FALSE")
    touched = "SELECT order_date FROM build.touched_dates"
    if mode == "overwrite":
        conn.execute(f"DELETE FROM build.{table} WHERE order_date IN ({touched})")
        conn.execute(f"INSERT INTO build.{table} SELECT * FROM new_{table} "
                     f"WHERE order_date IN ({touched})")
    else:
        conn.execute(f"INSERT INTO build.{table} SELECT * FROM new_{table} WHERE order_id IN "
                     "(SELECT order_id FROM raw_orders WHERE extract_date = ?)", [partition_date])
    return conn.execute(f"SELECT count(*) FROM build.{table}").fetchone()[0]

"""Raw storage. One partition per table per extract date, replaced as a whole.

`ParquetStore` is the lake. `MemoryStore` keeps the same partitions in memory
for tests and for the notebook, which may not write to disk. Both expose the
same views to DuckDB (`raw_orders`, `raw_order_lines`, `raw_customers`,
`raw_order_keys`) with an `extract_date` column, so the SQL models cannot tell
them apart.

Two properties matter.

* A partition is written to a temp file and renamed over the old one. A rerun
  of the same date replaces the same path and cannot leave half a file.
* Rows are written in the order the extract returned them (sorted by key), so
  the same input gives the same bytes.
"""
import hashlib
import io
import os
import shutil
from datetime import date

import pyarrow.parquet as pq

TABLES = ("orders", "order_lines", "customers", "order_keys")


def _part_name(d):
    return f"extract_date={d.isoformat()}"


class ParquetStore:
    def __init__(self, root):
        self.root = root

    def _dir(self, table, d):
        return self.root / table / _part_name(d)

    def file(self, table, d):
        return self._dir(table, d) / "part.parquet"

    def land(self, table, d, arrow_table):
        folder = self._dir(table, d)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "part.parquet.tmp"
        pq.write_table(arrow_table, tmp)
        os.replace(tmp, folder / "part.parquet")

    def partitions(self, table):
        base = self.root / table
        if not base.exists():
            return []
        return sorted(date.fromisoformat(p.name.split("=")[1]) for p in base.iterdir())

    def read(self, table, d):
        return pq.read_table(self.file(table, d))

    def row_counts(self, table):
        return {d: pq.ParquetFile(self.file(table, d)).metadata.num_rows
                for d in self.partitions(table)}

    def digest(self, table, d):
        return hashlib.sha256(self.file(table, d).read_bytes()).hexdigest()

    def quarantined(self):
        base = self.root / "_quarantine"
        return sorted(f"{p.parent.name}/{p.name}" for p in base.rglob("extract_date=*"))             if base.exists() else []

    def quarantine(self, d, tables):
        """Move a failed partition out of the lake so no later run reads it."""
        for table in tables:
            src = self._dir(table, d)
            if not src.exists():
                continue
            dst = self.root / "_quarantine" / table / _part_name(d)
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    def register(self, conn):
        for table in TABLES:
            if not self.partitions(table):
                continue
            glob = (self.root / table).as_posix() + "/extract_date=*/part.parquet"
            conn.execute(
                f"CREATE OR REPLACE TEMP VIEW raw_{table} AS "
                f"SELECT * FROM read_parquet('{glob}', hive_partitioning = true, "
                "union_by_name = true)")


class MemoryStore:
    def __init__(self):
        self.parts = {}     # (table, date) -> pyarrow Table
        self.quarantined_parts = {}

    def land(self, table, d, arrow_table):
        self.parts[(table, d)] = arrow_table

    def partitions(self, table):
        return sorted(d for (t, d) in self.parts if t == table)

    def read(self, table, d):
        return self.parts[(table, d)]

    def row_counts(self, table):
        return {d: self.parts[(table, d)].num_rows for d in self.partitions(table)}

    def digest(self, table, d):
        """sha256 of the bytes this partition would have as a Parquet file."""
        buf = io.BytesIO()
        pq.write_table(self.parts[(table, d)], buf)
        return hashlib.sha256(buf.getvalue()).hexdigest()

    def quarantined(self):
        return sorted(f"{t}/extract_date={d.isoformat()}" for (t, d) in self.quarantined_parts)

    def quarantine(self, d, tables):
        for table in tables:
            if (table, d) in self.parts:
                self.quarantined_parts[(table, d)] = self.parts.pop((table, d))

    def register(self, conn):
        for table in TABLES:
            selects = []
            for d in self.partitions(table):
                name = f"mem_{table}_{d.isoformat().replace('-', '')}"
                conn.register(name, self.parts[(table, d)])
                selects.append(f"SELECT *, DATE '{d.isoformat()}' AS extract_date FROM {name}")
            if selects:
                conn.execute(f"CREATE OR REPLACE TEMP VIEW raw_{table} AS "
                             + " UNION ALL BY NAME ".join(selects))

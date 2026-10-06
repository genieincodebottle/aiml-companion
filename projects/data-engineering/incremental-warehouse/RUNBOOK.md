# Runbook

For whoever is on call for this pipeline and did not build it.

## How a run works

One DAG run per logical date. The logical date is the day of data the run
processes, not the day it runs. Every task receives it as `run_date`.

```
extract_orders, extract_customers
  >> land_raw >> check_raw >> prepare_build
  >> stg_orders, stg_order_lines, stg_customer_versions
  >> dim_customer_scd2 >> plan_partitions
  >> fct_order, fct_order_line >> check_marts >> publish
```

`check_raw` and `check_marts` stop the run. Everything after a failed task shows
as `skipped`, and `published` keeps the last good tables. Nothing needs undoing.

Look at a run with `python run.py runs --scenario <folder under data/> --date <date>`,
or query `ops.runs` and `ops.check_results` in `data/<scenario>/warehouse.duckdb`.

## Rerun and backfill

Rerunning a date is always safe. It replaces the raw partition for that date and
rebuilds the order dates it touches. Run the dates in order, oldest first.

```
python run.py rerun --date 2025-06-16
python run.py backfill --start 2025-06-10 --end 2025-06-14
```

Do not run a later date before an earlier one that failed. The extract window
for a date is that date plus a 180 minute lookback, so a skipped day is a gap in
the raw lake, and `reconcile_by_date` will fail until the gap is filled.

## Transient failures

A task that raises `TransientError` is retried twice, after 0.05 and 0.1
seconds. `attempts` above 1 in `ops.runs` means a retry happened and nothing
else needs doing. A task `failed` with attempts 3 means the source stayed
unreachable. Fix the connection, then rerun the date.

## Checks that stop a run

Each check below is in `ops.check_results` with a detail string.

### check_raw, contract_orders, contract_order_lines, contract_customers

Meaning. The columns landed from the source differ from `checks.contract` in
`conf/config.yaml`. Someone renamed, added or dropped a column.

First query. Compare `missing` and `unexpected` in the detail, then look at the
source schema with `PRAGMA table_info(orders)`.

Fix and rerun. Agree the change with the source owner. Either have it reverted,
or update the contract and the staging SQL together. Then
`python run.py rerun --date <date>`. The failed partition was moved to
`data/<scenario>/raw/_quarantine/`, so later runs never read it.

### check_raw, nulls_orders, nulls_order_lines, nulls_customers

Meaning. A required column has more nulls than `checks.max_null_rate`. The
writer to the source stopped filling it.

First query.
```sql
SELECT count(*), count(customer_id) FROM raw_orders WHERE extract_date = '<date>';
```

Fix and rerun. Have the writer fixed and the rows repaired in the source, then
rerun the date.

### check_raw, volume

Meaning. Today's order rows are outside 0.5x to 3x the median of the previous
seven partitions. Rows are missing (an upstream job did not write) or doubled.

First query.
```sql
SELECT extract_date, count(*) FROM raw_orders GROUP BY 1 ORDER BY 1 DESC LIMIT 10;
```

Fix and rerun. Check the upstream job. Once the source holds the full day, rerun
the date. If the quiet day is real, widen `volume_min_ratio` for that date only
and say so in the ticket.

### check_raw, freshness

Meaning. The newest `updated_at` is more than `freshness_hours` before the end
of the logical day. The extract ran before the source caught up.

First query. `SELECT max(updated_at) FROM orders` on the source.

Fix and rerun. Wait for the source, then rerun the date.

### check_marts, grain_fct_order, grain_fct_order_line, grain_dim_customer

Meaning. A key appears twice. Either `load_mode` was changed to `append`, or a
partition overwrite was interrupted outside the pipeline.

First query.
```sql
SELECT order_id, count(*) FROM build.fct_order GROUP BY 1 HAVING count(*) > 1 LIMIT 10;
```

Fix and rerun. Confirm `pipeline.load_mode` is `overwrite`, then rerun the date.
`published` was not touched.

### check_marts, ri_fct_order_customer

Meaning. An order has no customer dimension row for its order date. The
customer extract is behind the order extract, or `customer_id` is null.

First query.
```sql
SELECT f.order_id, f.customer_id, f.order_date FROM build.fct_order f
LEFT JOIN build.dim_customer_scd2 d USING (customer_sk) WHERE d.customer_sk IS NULL LIMIT 10;
```

Fix and rerun. Rerun the date once the customer rows have landed.

### check_marts, lines_sum_to_order_total

Meaning. An order total differs from the sum of its lines. Lines of one order
landed in different runs.

First query. Compare `fct_order.order_total_cents` with `sum(line_total_cents)`
for the order in `build.fct_order_line`.

Fix and rerun. Rerun the date that extracted the order.

### check_marts, reconcile_order_count, reconcile_revenue, reconcile_by_date

Meaning. The warehouse disagrees with a control total taken from the source.
The usual causes are a late commit older than the lookback, a skipped date, or
a hard delete that the key snapshot did not see. `reconcile_by_date` names the
first dates that differ.

First query. Run the same count on both sides for the first date in the detail.

```sql
-- source (SQLite)
SELECT count(*) FROM orders WHERE substr(created_at, 1, 10) = '<date>';
-- warehouse (DuckDB)
SELECT count(*) FROM build.fct_order WHERE order_date = '<date>';
```

Fix and rerun. For a skipped date run `python run.py backfill --start <date>
--end <date>`. For a late commit older than `pipeline.lookback_minutes`, raise
the lookback above the real lateness, then rerun every date from the first
differing date.

## What a failed run leaves behind

| Place | State |
|---|---|
| `published` tables | unchanged, still the last good run |
| `build` schema | scratch, rebuilt from `published` on the next run |
| raw partition for the date | quarantined if `check_raw` failed, otherwise kept |
| `ops.runs`, `ops.check_results` | the full record, nothing is deleted |

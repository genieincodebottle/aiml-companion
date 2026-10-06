-- One row per order line, the latest extract of it.
--   dbt equivalent: models/staging/stg_order_lines.sql
SELECT
    order_id,
    line_no,
    sku,
    quantity,
    unit_price_cents,
    quantity * unit_price_cents AS line_total_cents
FROM raw_order_lines
QUALIFY row_number() OVER (PARTITION BY order_id, line_no ORDER BY extract_date DESC) = 1
ORDER BY order_id, line_no

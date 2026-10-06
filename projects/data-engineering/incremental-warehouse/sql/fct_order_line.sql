-- Grain is one row per order line, loaded by order_date partition.
--   dbt equivalent: models/marts/fct_order_line.sql (incremental, insert_overwrite)
SELECT
    l.order_id,
    l.line_no,
    o.order_date,
    l.sku,
    l.quantity,
    l.unit_price_cents,
    l.line_total_cents
FROM {{ ref('stg_order_lines') }} AS l
JOIN {{ ref('stg_orders') }} AS o USING (order_id)
WHERE NOT o.is_deleted
ORDER BY l.order_id, l.line_no

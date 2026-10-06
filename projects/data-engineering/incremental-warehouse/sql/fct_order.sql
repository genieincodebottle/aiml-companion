-- Grain is one row per order, loaded by order_date partition.
--   dbt equivalent: models/marts/fct_order.sql (incremental, insert_overwrite by order_date)
WITH totals AS (
    SELECT order_id, count(*) AS line_count, sum(line_total_cents) AS order_total_cents
    FROM {{ ref('stg_order_lines') }}
    GROUP BY order_id
)
SELECT
    o.order_id,
    o.order_date,
    c.customer_sk,
    o.customer_id,
    c.region,
    o.status,
    coalesce(t.line_count, 0) AS line_count,
    coalesce(t.order_total_cents, 0) AS order_total_cents,
    CASE WHEN o.status IN ({{ revenue_statuses }}) THEN coalesce(t.order_total_cents, 0) ELSE 0 END
        AS revenue_cents
FROM {{ ref('stg_orders') }} AS o
LEFT JOIN totals AS t USING (order_id)
{% if scd_type == 2 -%}
-- The region the customer lived in on the day of the order.
LEFT JOIN {{ ref('dim_customer_scd2') }} AS c
    ON c.customer_id = o.customer_id AND o.order_date >= c.valid_from AND o.order_date < c.valid_to
{%- else -%}
-- Type 1 shortcut. Today's region, applied to every past order.
LEFT JOIN {{ ref('dim_customer_scd2') }} AS c
    ON c.customer_id = o.customer_id AND c.is_current
{%- endif %}
WHERE NOT o.is_deleted
ORDER BY o.order_id

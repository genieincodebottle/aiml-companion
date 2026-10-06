-- One row per order, the latest version, with hard deletes marked.
--   dbt equivalent: models/staging/stg_orders.sql (materialized = table)
WITH versions AS (
    -- The lookback re-reads rows already landed. Keep one copy of each
    -- (order_id, updated_at), so a repeated row cannot be counted twice.
    SELECT *
    FROM raw_orders
    QUALIFY row_number() OVER (PARTITION BY order_id, updated_at ORDER BY extract_date) = 1
),
latest AS (
    SELECT *
    FROM versions
    QUALIFY row_number() OVER (PARTITION BY order_id ORDER BY updated_at DESC, extract_date DESC) = 1
){% if detect_deletes %},
live_keys AS (
    -- The newest key snapshot is the set of orders that exist in the source now.
    SELECT order_id
    FROM raw_order_keys
    WHERE extract_date = (SELECT max(extract_date) FROM raw_order_keys)
){% endif %}
SELECT
    l.order_id,
    l.customer_id,
    l.status,
    CAST(l.created_at AS TIMESTAMP)::DATE AS order_date,
    l.updated_at,
    {% if detect_deletes %}k.order_id IS NULL{% else %}FALSE{% endif %} AS is_deleted
FROM latest AS l
{% if detect_deletes %}LEFT JOIN live_keys AS k USING (order_id){% endif %}
ORDER BY l.order_id

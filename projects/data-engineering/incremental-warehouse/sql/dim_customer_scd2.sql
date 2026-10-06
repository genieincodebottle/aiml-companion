-- Type 2 history of the customer region.
--   dbt equivalent: a snapshot on region, or this model over the raw versions.
-- A new row opens only when the region changes. An email change makes a new
-- version upstream and no new row here.
WITH flagged AS (
    SELECT
        *,
        region IS DISTINCT FROM lag(region) OVER (PARTITION BY customer_id ORDER BY updated_at)
            AS region_changed
    FROM {{ ref('stg_customer_versions') }}
),
islands AS (
    SELECT
        customer_id,
        region,
        version_date,
        sum(CASE WHEN region_changed THEN 1 ELSE 0 END)
            OVER (PARTITION BY customer_id ORDER BY updated_at) AS version_no
    FROM flagged
),
spans AS (
    SELECT customer_id, version_no, any_value(region) AS region, min(version_date) AS valid_from
    FROM islands
    GROUP BY customer_id, version_no
)
SELECT
    -- Stable across rebuilds. row_number() would renumber every customer
    -- whenever a new one arrives and orphan the keys stored in the fact.
    customer_id * 1000 + version_no AS customer_sk,
    customer_id,
    region,
    valid_from,
    coalesce(lead(valid_from) OVER (PARTITION BY customer_id ORDER BY valid_from), DATE '9999-12-31')
        AS valid_to,
    lead(valid_from) OVER (PARTITION BY customer_id ORDER BY valid_from) IS NULL AS is_current
FROM spans
ORDER BY customer_id, valid_from

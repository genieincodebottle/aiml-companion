-- Every version of every customer, repeats removed. The dimension is built from this.
--   dbt equivalent: models/staging/stg_customer_versions.sql
SELECT
    customer_id,
    region,
    updated_at,
    CAST(updated_at AS TIMESTAMP)::DATE AS version_date
FROM raw_customers
QUALIFY row_number() OVER (PARTITION BY customer_id, updated_at ORDER BY extract_date) = 1
ORDER BY customer_id, updated_at

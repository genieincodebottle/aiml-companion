-- Revenue on the line fact must equal revenue recomputed from staging. The
-- formula is written out again here on purpose, so a change to the model's
-- formula is not copied into its own check.
select
    f.revenue as fact_revenue,
    s.revenue as staging_revenue
from (select sum(line_revenue) as revenue from {{ ref('fct_order_line') }}) as f
cross join (
    select sum(quantity * unit_price - discount) as revenue from {{ ref('stg_order_lines') }}
) as s
where f.revenue is distinct from s.revenue

{{ config(tags=['added']) }}
-- Added after the matrix showed a break nobody caught. A fact with the right
-- grain has exactly one row for each staging row it describes. Fewer rows means
-- a join dropped some, more means a join fanned out or the grain changed.
select 'fct_order_line' as fact, f.n as fact_rows, s.n as staging_rows
from (select count(*) as n from {{ ref('fct_order_line') }}) as f
cross join (select count(*) as n from {{ ref('stg_order_lines') }}) as s
where f.n <> s.n

union all

select 'fct_order' as fact, f.n as fact_rows, s.n as staging_rows
from (select count(*) as n from {{ ref('fct_order') }}) as f
cross join (select count(*) as n from {{ ref('stg_orders') }}) as s
where f.n <> s.n

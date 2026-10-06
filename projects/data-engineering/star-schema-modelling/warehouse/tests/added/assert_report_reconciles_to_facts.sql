{{ config(tags=['added']) }}
-- Added after the matrix showed a break nobody caught. A report must add up to
-- the fact it is built from. Both reports are checked against the fact of the
-- right grain, so shipping read from the line grain shows up here.
select 'line_revenue' as measure, r.total as report_total, f.total as fact_total
from (select sum(line_revenue) as total from {{ ref('rpt_revenue_by_region_category_day') }}) as r
cross join (select sum(line_revenue) as total from {{ ref('fct_order_line') }}) as f
where r.total is distinct from f.total

union all

select 'shipping_revenue' as measure, r.total as report_total, f.total as fact_total
from (select sum(shipping_revenue) as total from {{ ref('rpt_shipping_by_region_day') }}) as r
cross join (select sum(shipping_cost) as total from {{ ref('fct_order') }}) as f
where r.total is distinct from f.total

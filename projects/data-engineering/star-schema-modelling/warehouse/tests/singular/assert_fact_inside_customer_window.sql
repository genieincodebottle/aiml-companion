-- The business rule behind the Type 2 dimension. Whatever customer version a
-- fact points at must have been valid when the order was placed. No generic
-- test can say this, because it compares a fact column to a dimension window.
select 'fct_order_line' as fact, f.order_id, f.line_no
from {{ ref('fct_order_line') }} as f
inner join {{ ref('dim_customer') }} as c on c.customer_sk = f.customer_sk
where not (f.order_ts >= c.valid_from and f.order_ts < c.valid_to)

union all

select 'fct_order' as fact, f.order_id, null as line_no
from {{ ref('fct_order') }} as f
inner join {{ ref('dim_customer') }} as c on c.customer_sk = f.customer_sk
where not (f.order_ts >= c.valid_from and f.order_ts < c.valid_to)

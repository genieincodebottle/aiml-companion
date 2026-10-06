-- Shipping charged, by the customer's region at order time. Read from the
-- order grain fact, where one row is one order.
select
    f.order_date,
    c.region,
    count(*) as orders,
    sum(f.shipping_cost) as shipping_revenue
from {{ ref('fct_order') }} as f
left join {{ ref('dim_customer') }} as c on c.customer_sk = f.customer_sk
group by 1, 2

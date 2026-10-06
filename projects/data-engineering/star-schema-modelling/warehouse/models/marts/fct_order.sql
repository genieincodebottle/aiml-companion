-- Grain: one row per order. Shipping is charged once per order, so it lives
-- here and nowhere else. Put it on the line fact and every multi-line order
-- counts it once per line.
select
    o.order_id,
    c.customer_sk,
    o.order_ts,
    cast(o.order_ts as date) as order_date,
    o.status,
    o.shipping_cost
from {{ ref('stg_orders') }} as o
left join {{ ref('dim_customer') }} as c
    on c.customer_id = o.customer_id
    and o.order_ts >= c.valid_from
    and o.order_ts < c.valid_to

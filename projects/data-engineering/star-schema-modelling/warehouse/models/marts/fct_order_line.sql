-- Grain: one row per order line, (order_id, line_no).
-- customer_sk is the customer version in force when the order was placed, found
-- by the validity window, never by is_current. LEFT join so a missing version
-- shows up as a null that the not_null test catches, not as a vanished row.
select
    l.order_id,
    l.line_no,
    c.customer_sk,
    p.product_sk,
    o.order_ts,
    cast(o.order_ts as date) as order_date,
    l.quantity,
    l.unit_price,
    l.discount,
    l.quantity * l.unit_price - l.discount as line_revenue
from {{ ref('stg_order_lines') }} as l
inner join {{ ref('stg_orders') }} as o on o.order_id = l.order_id
left join {{ ref('dim_customer') }} as c
    on c.customer_id = o.customer_id
    and o.order_ts >= c.valid_from
    and o.order_ts < c.valid_to
left join {{ ref('dim_product') }} as p on p.product_id = l.product_id

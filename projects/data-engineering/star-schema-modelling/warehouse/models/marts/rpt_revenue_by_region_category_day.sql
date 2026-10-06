-- Revenue by the region the customer was in when they ordered.
select
    f.order_date,
    c.region,
    p.category,
    sum(f.quantity) as units,
    count(*) as order_lines,
    sum(f.line_revenue) as line_revenue
from {{ ref('fct_order_line') }} as f
left join {{ ref('dim_customer') }} as c on c.customer_sk = f.customer_sk
left join {{ ref('dim_product') }} as p on p.product_sk = f.product_sk
group by 1, 2, 3

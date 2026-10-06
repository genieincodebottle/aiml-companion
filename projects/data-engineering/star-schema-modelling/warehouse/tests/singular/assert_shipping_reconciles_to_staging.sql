-- Shipping on the order fact must equal shipping in staging.
select
    f.shipping as fact_shipping,
    s.shipping as staging_shipping
from (select sum(shipping_cost) as shipping from {{ ref('fct_order') }}) as f
cross join (select sum(shipping_cost) as shipping from {{ ref('stg_orders') }}) as s
where f.shipping is distinct from s.shipping

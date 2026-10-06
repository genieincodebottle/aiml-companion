select
    cast(order_id as integer) as order_id,
    cast(customer_id as integer) as customer_id,
    cast(order_ts as timestamp) as order_ts,
    status,
    cast(shipping_cost as decimal(12, 2)) as shipping_cost
from {{ source('raw', 'orders') }}

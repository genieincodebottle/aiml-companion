-- The upstream retry re-sent chunks of this file, so the same (order_id, line_no)
-- can arrive twice. The natural key is the grain, so staging keeps one copy.
with typed as (
    select
        cast(order_id as integer) as order_id,
        cast(line_no as integer) as line_no,
        cast(product_id as integer) as product_id,
        cast(quantity as integer) as quantity,
        cast(unit_price as decimal(12, 2)) as unit_price,
        cast(discount as decimal(12, 2)) as discount
    from {{ source('raw', 'order_lines') }}
),

ranked as (
    select
        *,
        row_number() over (partition by order_id, line_no order by product_id) as copy_rank
    from typed
)

select
    order_id,
    line_no,
    product_id,
    quantity,
    unit_price,
    discount
from ranked
where copy_rank = 1

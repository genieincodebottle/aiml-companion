select
    cast(product_id as integer) as product_id,
    name,
    category
from {{ source('raw', 'products') }}

select
    cast(product_id as integer) as product_id,
    category as corrected_category,
    cast(corrected_at as timestamp) as corrected_at
from {{ source('raw', 'product_corrections') }}

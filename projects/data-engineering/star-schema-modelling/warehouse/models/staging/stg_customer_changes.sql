-- Staging casts and renames. It does not filter, join or correct anything.
select
    cast(customer_id as integer) as customer_id,
    name,
    email,
    region,
    cast(updated_at as timestamp) as updated_at
from {{ source('raw', 'customer_changes') }}

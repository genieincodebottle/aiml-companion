-- Type 1 product dimension. A category correction fixes a data entry error, so
-- it replaces the old value and restates history. Nobody wants last quarter's
-- revenue left under a category that was never right.
select
    row_number() over (order by p.product_id) as product_sk,
    p.product_id,
    p.name,
    coalesce(c.corrected_category, p.category) as category
from {{ ref('stg_products') }} as p
left join {{ ref('stg_product_corrections') }} as c on c.product_id = p.product_id

{{ config(tags=['added']) }}
-- Added after the matrix showed a break nobody caught. Every category correction
-- from the source must be what dim_product holds.
select d.product_id, d.category as dim_category, c.corrected_category
from {{ ref('dim_product') }} as d
inner join {{ ref('stg_product_corrections') }} as c on c.product_id = d.product_id
where d.category <> c.corrected_category

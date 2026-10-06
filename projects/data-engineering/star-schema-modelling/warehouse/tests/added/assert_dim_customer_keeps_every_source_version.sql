{{ config(tags=['added']) }}
-- Added after the matrix showed a break nobody caught. A Type 2 dimension keeps
-- one row for every version the source reported. A dimension overwritten in
-- place still has one current row and windows that contain every order, so only
-- a count against the source shows that history is gone.
select d.n as dim_rows, s.n as source_versions
from (select count(*) as n from {{ ref('dim_customer') }}) as d
cross join (select count(*) as n from {{ ref('stg_customer_changes') }}) as s
where d.n <> s.n

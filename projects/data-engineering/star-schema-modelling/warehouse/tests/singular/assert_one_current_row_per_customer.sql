-- Every customer has exactly one current version. Zero means a customer who
-- vanishes from "now" reports, two means they are counted twice.
select
    customer_id,
    sum(case when is_current then 1 else 0 end) as current_rows
from {{ ref('dim_customer') }}
group by customer_id
having sum(case when is_current then 1 else 0 end) <> 1

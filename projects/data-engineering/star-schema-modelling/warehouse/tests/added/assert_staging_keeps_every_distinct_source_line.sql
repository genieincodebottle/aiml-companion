{{ config(tags=['added']) }}
-- Added after the matrix showed a break nobody caught. Reconciling a fact to
-- staging proves nothing when staging itself is wrong. This test goes back to
-- the raw table and counts the distinct (order_id, line_no) keys it holds.
select s.n as staging_rows, r.n as distinct_source_lines
from (select count(*) as n from {{ ref('stg_order_lines') }}) as s
cross join (
    select count(*) as n
    from (select distinct order_id, line_no from {{ source('raw', 'order_lines') }})
) as r
where s.n <> r.n

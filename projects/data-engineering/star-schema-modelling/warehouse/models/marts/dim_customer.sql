-- Type 2 customer dimension. One row per version the source reported.
--
-- valid_from is inclusive and valid_to is EXCLUSIVE, and valid_to is the next
-- version's valid_from. Facts join with  ts >= valid_from and ts < valid_to,
-- so an order stamped exactly on a change lands in exactly one version.
--
-- Version 1 is back-dated to 1900-01-01. Some customers were synced after their
-- first order, so the source's first timestamp is later than the order. The
-- order proves the customer existed, and the first known region is the best
-- evidence for where they were, so version 1 covers all earlier time. An
-- "unknown member" row would drop the region from revenue that has one.
with versions as (
    select
        customer_id,
        name,
        email,
        region,
        updated_at,
        row_number() over (partition by customer_id order by updated_at) as version_no,
        lead(updated_at) over (partition by customer_id order by updated_at) as next_updated_at
    from {{ ref('stg_customer_changes') }}
)

select
    row_number() over (order by customer_id, updated_at) as customer_sk,
    customer_id,
    name,
    email,
    region,
    version_no,
    case when version_no = 1 then cast('1900-01-01' as timestamp) else updated_at end as valid_from,
    coalesce(next_updated_at, cast('9999-12-31' as timestamp)) as valid_to,
    next_updated_at is null as is_current
from versions

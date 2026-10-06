"""The ways to break the model, as data.

Each variant is a list of edits to the SHIPPED model text. They are applied to a
copy of the dbt project, so the real models are never touched. The same edits
feed the `naive` table (the realistic wrong answer for each planted trap) and the
`break` mutation matrix (which tests notice the wrong answer).

Pure standard library, because the notebook embeds this file as it is.
"""

JOIN_WINDOW = ("    and o.order_ts >= c.valid_from\n"
               "    and o.order_ts < c.valid_to")
FCT_ORDER = "models/marts/fct_order.sql"
FCT_LINE = "models/marts/fct_order_line.sql"
DIM_CUSTOMER = "models/marts/dim_customer.sql"
STG_LINES = "models/staging/stg_order_lines.sql"
BACKDATE = ("case when version_no = 1 then cast('1900-01-01' as timestamp) "
            "else updated_at end as valid_from")

TYPE1_CUSTOMER = """-- mutation: Type 1. One row per customer holding the latest region.
select
    row_number() over (order by customer_id) as customer_sk,
    customer_id,
    name,
    email,
    region,
    1 as version_no,
    cast('1900-01-01' as timestamp) as valid_from,
    cast('9999-12-31' as timestamp) as valid_to,
    true as is_current
from (
    select
        *,
        row_number() over (partition by customer_id order by updated_at desc) as recency
    from {{ ref('stg_customer_changes') }}
)
where recency = 1
"""

WRONG_GRAIN_LINE = """-- mutation: one row per (order, category) instead of one row per order line.
select
    l.order_id,
    min(l.line_no) as line_no,
    c.customer_sk,
    min(p.product_sk) as product_sk,
    o.order_ts,
    cast(o.order_ts as date) as order_date,
    sum(l.quantity) as quantity,
    sum(l.quantity * l.unit_price - l.discount) as line_revenue
from {{ ref('stg_order_lines') }} as l
inner join {{ ref('stg_orders') }} as o on o.order_id = l.order_id
left join {{ ref('dim_customer') }} as c
    on c.customer_id = o.customer_id
    and o.order_ts >= c.valid_from
    and o.order_ts < c.valid_to
left join {{ ref('dim_product') }} as p on p.product_id = l.product_id
group by l.order_id, c.customer_sk, o.order_ts, p.category
"""


def _replace(path, old, new):
    return {"op": "replace", "path": path, "old": old, "new": new}


def _both_facts(old, new):
    return [_replace(FCT_ORDER, old, new), _replace(FCT_LINE, old, new)]


VARIANTS = [
    {"id": "m01_join_on_is_current", "trap": "1 relocation",
     "title": "Join facts to the customer dimension on is_current",
     "ops": _both_facts(JOIN_WINDOW, "    and c.is_current")},
    {"id": "m02_type1_customer", "trap": "1 relocation",
     "title": "Overwrite the customer region in place (Type 1)",
     "ops": [{"op": "write", "path": DIM_CUSTOMER, "text": TYPE1_CUSTOMER}]},
    {"id": "m03_shipping_on_line_fact", "trap": "2 mixed grain",
     "title": "Put order-level shipping on the line fact and sum it there",
     "ops": [
         _replace(FCT_LINE, "    l.discount,\n",
                  "    l.discount,\n    o.shipping_cost,\n"),
         _replace("models/marts/rpt_shipping_by_region_day.sql",
                  "from {{ ref('fct_order') }} as f",
                  "from {{ ref('fct_order_line') }} as f"),
         _replace("models/marts/rpt_shipping_by_region_day.sql",
                  "count(*) as orders", "count(distinct f.order_id) as orders")]},
    {"id": "m04_drop_dedup", "trap": "3 re-sent delivery",
     "title": "Remove the dedup from stg_order_lines",
     "ops": [_replace(STG_LINES, "where copy_rank = 1", "-- mutation: no filter on copy_rank")]},
    {"id": "m05_inner_join_late_customers", "trap": "4 late-arriving dimension",
     "title": "No back-dating, and inner join facts to the window",
     "ops": [_replace(DIM_CUSTOMER, BACKDATE, "updated_at as valid_from")]
            + _both_facts("left join {{ ref('dim_customer') }} as c",
                          "inner join {{ ref('dim_customer') }} as c")},
    {"id": "m06_no_backdate_left_join", "trap": "4 late-arriving dimension",
     "title": "No back-dating, facts keep the left join",
     "ops": [_replace(DIM_CUSTOMER, BACKDATE, "updated_at as valid_from")]},
    {"id": "m07_inclusive_window_join", "trap": "5 overlapping windows",
     "title": "Join with <= valid_to, so a boundary matches two versions",
     "ops": _both_facts("and o.order_ts < c.valid_to", "and o.order_ts <= c.valid_to")},
    {"id": "m08_dedup_on_wrong_key", "trap": "3 re-sent delivery",
     "title": "Dedup on order_id alone instead of (order_id, line_no)",
     "ops": [_replace(STG_LINES, "partition by order_id, line_no", "partition by order_id")]},
    {"id": "m09_float_money", "trap": "money type",
     "title": "Cast money to DOUBLE in staging",
     "ops": [
         _replace(STG_LINES, "cast(unit_price as decimal(12, 2))", "cast(unit_price as double)"),
         _replace(STG_LINES, "cast(discount as decimal(12, 2))", "cast(discount as double)"),
         _replace("models/staging/stg_orders.sql", "cast(shipping_cost as decimal(12, 2))",
                  "cast(shipping_cost as double)")]},
    {"id": "m10_wrong_grain", "trap": "grain",
     "title": "Collapse order lines to one row per order and category",
     "ops": [{"op": "write", "path": FCT_LINE, "text": WRONG_GRAIN_LINE}]},
    {"id": "m11_stale_is_current", "trap": "SCD2 maintenance",
     "title": "Never clear is_current on superseded versions",
     "ops": [_replace(DIM_CUSTOMER, "next_updated_at is null as is_current",
                      "true as is_current")]},
    {"id": "m12_skip_product_correction", "trap": "Type 1 correction",
     "title": "Ignore the product category corrections",
     "ops": [_replace("models/marts/dim_product.sql",
                      "coalesce(c.corrected_category, p.category) as category",
                      "p.category as category")]},
    {"id": "m13_timezone_shift", "trap": "date derivation",
     "title": "Derive order_date with a 6 hour offset",
     "ops": _both_facts("cast(o.order_ts as date) as order_date",
                        "cast(o.order_ts + interval 6 hour as date) as order_date")},
]

# The realistic wrong answer for each planted trap, in README order.
NAIVE_IDS = ["m01_join_on_is_current", "m02_type1_customer", "m03_shipping_on_line_fact",
             "m04_drop_dedup", "m05_inner_join_late_customers", "m07_inclusive_window_join"]


def by_id(variant_id):
    return next(v for v in VARIANTS if v["id"] == variant_id)


def patch_files(files, ops):
    """Return a copy of {path: text} with the edits applied.

    Every replace must match exactly once. A silent no-op would make a mutation
    look harmless when it never ran, so a miss raises."""
    out = dict(files)
    for op in ops:
        if op["op"] == "write":
            out[op["path"]] = op["text"]
            continue
        text = out[op["path"]]
        found = text.count(op["old"])
        if found != 1:
            raise ValueError(f"{op['path']}: expected 1 match for {op['old']!r}, found {found}")
        out[op["path"]] = text.replace(op["old"], op["new"])
    return out

"""A seeded 30-day history of a shop, and the answer key that goes with it.

Two things come out of `generate_world`.

* `days`, a list of event batches. The source simulator (`shop.py`) applies one
  batch per simulated day to a SQLite database. This is all the pipeline ever
  sees, and only through SQL against that database.
* `answer_key`, the true committed state after each day. Only `scoring.py` may
  read it. A test scans every other module for the symbol.

The generator plants two failures at counted sizes.

* Late commits. A change keeps the `updated_at` stamped when its transaction
  started (late evening of the previous day) but becomes visible the next day,
  after the extract for that window has already run. The generator picks only
  rows that were not touched on the two days around the change, so the late
  stamp can never be older than another version of the same row.
* Hard deletes. Revenue-counting orders disappear from the source with no
  trace. A `WHERE updated_at > watermark` query cannot see a missing row.

Standard library only, so the notebook builder can copy this file verbatim.
"""
import random
from datetime import datetime, timedelta

FINAL_STATUSES = ("delivered", "cancelled")


def _stamp(day_start, seconds):
    return (day_start + timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


class AnswerKey:
    """What the shop's data really was, as committed at the end of each day."""

    def __init__(self, cfg):
        self.revenue_statuses = tuple(cfg["revenue_statuses"])
        self.orders = {}          # order_id -> insert facts, see _register_order
        self.status_history = {}  # order_id -> [(stamp, visible_day, status)]
        self.deleted_on = {}      # order_id -> day the hard delete became visible
        self.customer_regions = {}  # customer_id -> [(stamp_date, visible_day, region)]
        self.late_events = []     # one dict per planted late commit
        self.deletes = []         # one dict per planted hard delete

    def _region_on(self, customer_id, order_date, as_of_day):
        """Region of the customer on a date, using only changes visible by as_of_day."""
        region = None
        versions = sorted(
            (v for v in self.customer_regions[customer_id] if v[1] <= as_of_day),
            key=lambda v: v[0])
        for stamp_date, _, reg in versions:
            if stamp_date <= order_date:
                region = reg
        return region

    def orders_as_of(self, day):
        """order_id -> dict(order_date, status, region, total_cents, revenue_cents)."""
        out = {}
        for oid, o in self.orders.items():
            if o["visible_day"] > day:
                continue
            gone = self.deleted_on.get(oid)
            if gone is not None and gone <= day:
                continue
            visible = [h for h in self.status_history[oid] if h[1] <= day]
            status = max(visible, key=lambda h: h[0])[2]
            counts = status in self.revenue_statuses
            out[oid] = {
                "order_date": o["order_date"],
                "customer_id": o["customer_id"],
                "status": status,
                "region": self._region_on(o["customer_id"], o["order_date"], day),
                "total_cents": o["total_cents"],
                "revenue_cents": o["total_cents"] if counts else 0,
            }
        return out

    def revenue_as_of(self, day):
        """(order_date, region) -> revenue in cents, for the state after `day`."""
        rev = {}
        for o in self.orders_as_of(day).values():
            if o["revenue_cents"]:
                key = (o["order_date"], o["region"])
                rev[key] = rev.get(key, 0) + o["revenue_cents"]
        return rev

    def dim_rows_as_of(self, day):
        """(customer_id, region, valid_from) with consecutive equal regions merged."""
        rows = set()
        for cid, versions in self.customer_regions.items():
            last = None
            for stamp_date, vis, region in sorted(
                    (v for v in versions if v[1] <= day), key=lambda v: v[0]):
                if region != last:
                    rows.add((cid, region, stamp_date))
                    last = region
        return rows


def generate_world(cfg):
    """Return (days, answer_key). `cfg` is the full config dict."""
    s = cfg["source"]
    rng = random.Random(cfg["seed"])
    start = datetime.strptime(s["start_date"], "%Y-%m-%d")
    key = AnswerKey(cfg)

    sku_prices = [rng.randint(*s["sku_price_cents"]) for _ in range(s["sku_count"])]
    customers = {}        # customer_id -> {"region", "last_touch"}
    orders = {}           # order_id -> {"status", "last_touch", "alive"}
    pool = []             # customer ids that can place orders today
    next_customer, next_order = 1, 1
    days = []

    for day in range(1, s["days"] + 1):
        day_start = start + timedelta(days=day - 1)
        day_str = day_start.strftime("%Y-%m-%d")
        events = []
        seq = [0]

        def add(ev):
            seq[0] += 1
            ev["seq"] = seq[0]
            events.append(ev)
            return ev

        def when(eligible_for_late):
            """(stamp, is_late). A late stamp falls before midnight of the previous day."""
            if eligible_for_late and day >= 2 and rng.random() < s["late_commit_rate"]:
                back = 1 + rng.randrange(s["late_max_minutes"] * 60)
                return _stamp(day_start, -back), True
            return _stamp(day_start, 3600 + rng.randrange(86400 - 3600)), False

        # 1. Hard deletes. Old, untouched, revenue-counting orders only.
        if day >= s["hard_delete_from_day"]:
            for _ in range(s["hard_deletes_per_day"]):
                eligible = sorted(
                    oid for oid, o in orders.items()
                    if o["alive"] and o["status"] in key.revenue_statuses
                    and o["last_touch"] <= day - 2)
                if not eligible:
                    continue
                oid = eligible[rng.randrange(len(eligible))]
                ts, _ = when(False)
                add({"op": "delete_order", "ts": ts, "late": False, "order_id": oid})
                orders[oid]["alive"] = False
                orders[oid]["last_touch"] = day
                key.deleted_on[oid] = day
                key.deletes.append({"order_id": oid, "day": day, "ts": ts})

        # 2. Status moves on open orders.
        for oid in sorted(orders):
            o = orders[oid]
            if not o["alive"] or o["status"] in FINAL_STATUSES or o["last_touch"] >= day:
                continue
            roll, new = rng.random(), None
            if o["status"] == "placed":
                if roll < s["p_placed_to_paid"]:
                    new = "paid"
                elif roll < s["p_placed_to_paid"] + s["p_placed_to_cancelled"]:
                    new = "cancelled"
            elif o["status"] == "paid" and roll < s["p_paid_to_shipped"]:
                new = "shipped"
            elif o["status"] == "shipped" and roll < s["p_shipped_to_delivered"]:
                new = "delivered"
            if new is None:
                continue
            ts, late = when(o["last_touch"] <= day - 2)
            add({"op": "update_order", "ts": ts, "late": late,
                 "order_id": oid, "status": new})
            o["status"], o["last_touch"] = new, day
            key.status_history[oid].append((ts, day, new))
            if late:
                key.late_events.append({"table": "orders", "key": oid, "ts": ts, "day": day})

        # 3. Customer changes. Only region changes matter to the answer key.
        for cid in list(pool):
            c = customers[cid]
            roll = rng.random()
            if roll < s["relocation_rate"]:
                others = [r for r in s["regions"] if r != c["region"]]
                region = others[rng.randrange(len(others))]
                ts, late = when(c["last_touch"] <= day - 2)
                add({"op": "update_customer", "ts": ts, "late": late,
                     "customer_id": cid, "region": region})
                c["region"], c["last_touch"] = region, day
                key.customer_regions[cid].append((ts[:10], day, region))
                if late:
                    key.late_events.append(
                        {"table": "customers", "key": cid, "ts": ts, "day": day})
            elif roll < s["relocation_rate"] + s["email_change_rate"]:
                ts, late = when(c["last_touch"] <= day - 2)
                add({"op": "update_customer", "ts": ts, "late": late,
                     "customer_id": cid, "email": f"c{cid}.v{day}@example.com"})
                c["last_touch"] = day
                if late:
                    key.late_events.append(
                        {"table": "customers", "key": cid, "ts": ts, "day": day})

        # 4. New customers. They can place orders from the next day.
        n_new = (s["initial_customers"] if day == 1
                 else rng.randint(*s["new_customers_per_day"]))
        new_ids = []
        for _ in range(n_new):
            cid, next_customer = next_customer, next_customer + 1
            region = s["regions"][rng.randrange(len(s["regions"]))]
            ts = _stamp(day_start, rng.randrange(3600))
            add({"op": "insert_customer", "ts": ts, "late": False, "customer_id": cid,
                 "name": f"Customer {cid}", "email": f"c{cid}@example.com",
                 "region": region})
            customers[cid] = {"region": region, "last_touch": day}
            key.customer_regions[cid] = [(ts[:10], day, region)]
            new_ids.append(cid)
        if day == 1:
            pool.extend(new_ids)

        # 5. New orders. Placed on a customer who existed at the start of the day.
        inserts = []
        for _ in range(rng.randint(*s["orders_per_day"])):
            cid = pool[rng.randrange(len(pool))]
            lines = []
            for line_no in range(1, rng.randint(*s["lines_per_order"]) + 1):
                sku = rng.randrange(s["sku_count"])
                lines.append({"line_no": line_no, "sku": f"SKU-{sku + 1:03d}",
                              "quantity": rng.randint(*s["quantity"]),
                              "unit_price_cents": sku_prices[sku]})
            ts, late = when(True)
            inserts.append(add({"op": "insert_order", "ts": ts, "late": late,
                                "customer_id": cid, "lines": lines}))

        # Late events are applied first (their transaction committed after
        # midnight), then the rest in stamp order. Order ids follow that order,
        # like an auto-increment column assigned at insert time.
        events.sort(key=lambda e: (0 if e["late"] else 1, e["ts"], e["seq"]))
        for ev in events:
            if ev["op"] != "insert_order":
                continue
            oid, next_order = next_order, next_order + 1
            ev["order_id"] = oid
            total = sum(l["quantity"] * l["unit_price_cents"] for l in ev["lines"])
            orders[oid] = {"status": "placed", "last_touch": day, "alive": True}
            key.orders[oid] = {"customer_id": ev["customer_id"],
                               "order_date": ev["ts"][:10], "visible_day": day,
                               "total_cents": total}
            key.status_history[oid] = [(ev["ts"], day, "placed")]
            if ev["late"]:
                key.late_events.append(
                    {"table": "orders", "key": oid, "ts": ev["ts"], "day": day})
        if day >= 2:
            pool.extend(new_ids)

        for ev in events:
            del ev["seq"]
        days.append({"day": day, "date": day_str, "events": events})

    return days, key


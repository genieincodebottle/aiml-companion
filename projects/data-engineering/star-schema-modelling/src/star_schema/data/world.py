"""The source system, simulated. Pure standard library, no file or database I/O.

WHY it is generated: on a real export nobody knows the true revenue by region at
the time of the order, so nobody can say how wrong a model is. Here the process
that creates the events also records what really happened (TRUE_REVENUE,
TRUE_SHIPPING, TRUE_LEDGER). The warehouse build never sees those names, and a
test enforces it. Only evaluation/answer_key.py reads them back.

WHY stdlib `random` and integer cents: the same seed gives the same bytes on
every OS and every library version. No float ever holds money in this file.

The notebook embeds this file as it is, so keep it dependency free.
"""
import datetime as dt
import random

FIRST_NAMES = ["Asha", "Ben", "Chen", "Dara", "Eli", "Farah", "Gus", "Hana", "Ivo",
               "Jin", "Kofi", "Lena", "Mateo", "Nia", "Omar", "Pia", "Quentin",
               "Rosa", "Sven", "Tara", "Uma", "Viktor", "Wen", "Xavi", "Yara", "Zane"]
LAST_NAMES = ["Adams", "Bose", "Cruz", "Diaz", "Evans", "Fox", "Gray", "Hale",
              "Iyer", "Jones", "Khan", "Lopez", "Mehta", "Nunez", "Okoye", "Park",
              "Reyes", "Singh", "Tran", "Usman", "Vega", "Wu", "Young", "Zhou"]
ADJECTIVES = ["Compact", "Classic", "Deluxe", "Eco", "Pro", "Smart", "Sturdy", "Lite"]
NOUNS = ["Lamp", "Kettle", "Racket", "Novel", "Drone", "Planter", "Puzzle", "Speaker",
         "Mat", "Bottle", "Tent", "Cable"]

CUSTOMER_HEADER = ["customer_id", "name", "email", "region", "updated_at"]
ORDER_HEADER = ["order_id", "customer_id", "order_ts", "status", "shipping_cost"]
LINE_HEADER = ["order_id", "line_no", "product_id", "quantity", "unit_price", "discount"]
PRODUCT_HEADER = ["product_id", "name", "category"]
CORRECTION_HEADER = ["product_id", "category", "corrected_at"]

FIRST_ORDER_ID = 100001


def money(cents):
    return f"{cents // 100}.{cents % 100:02d}"


def stamp(t):
    return t.strftime("%Y-%m-%d %H:%M:%S")


def _region_at(history, ts):
    """Region in force at ts. history[0] is the first known region and applies to
    every earlier time, even when the source stamped that row late."""
    region = history[0][1]
    for changed_at, new_region in history[1:]:
        if changed_at <= ts:
            region = new_region
    return region


def _build_customers(cfg, rng, start):
    d, t, sc = cfg["data"], cfg["traps"], cfg["showcase"]
    n, horizon = d["n_customers"], d["n_days"] * 86400
    ids = list(range(1, n + 1))
    others = [c for c in ids if c != sc["customer_id"]]
    rng.shuffle(others)
    n_late = round(t["late_customer_share"] * n)
    n_twice = round(t["relocate_twice_share"] * n)
    n_once = round(t["relocate_once_share"] * n) - 1   # the showcase customer is one mover
    late = set(others[:n_late])
    twice = set(others[n_late:n_late + n_twice])
    once = set(others[n_late + n_twice:n_late + n_twice + n_once])

    histories, names = {}, {}
    for cid in ids:
        names[cid] = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        region = rng.choice(d["regions"])
        if cid == sc["customer_id"]:
            names[cid] = sc["name"]
            signup = start - dt.timedelta(days=30)
            moved_at = dt.datetime.fromisoformat(sc["moved_at"])
            histories[cid] = [(signup, sc["region_from"]), (moved_at, sc["region_to"])]
            continue
        if cid in late:
            first = start + dt.timedelta(
                seconds=rng.randrange(t["late_first_row_min_day"] * 86400,
                                      t["late_first_row_max_day"] * 86400))
        else:
            first = start - dt.timedelta(seconds=rng.randrange(86400, 60 * 86400))
        history = [(first, region)]
        if cid in once or cid in twice:
            move = start + dt.timedelta(
                seconds=rng.randrange(t["first_move_earliest_day"] * 86400,
                                      horizon - 30 * 86400))
            region = rng.choice([r for r in d["regions"] if r != region])
            history.append((move, region))
            if cid in twice:
                room = horizon - 5 * 86400 - int((move - start).total_seconds())
                move2 = move + dt.timedelta(
                    seconds=rng.randrange(t["second_move_min_gap_days"] * 86400, room))
                region = rng.choice([r for r in d["regions"] if r != region])
                history.append((move2, region))
        histories[cid] = history
    return ids, names, histories, late, once | {sc["customer_id"]}, twice


def _build_products(cfg, rng, start):
    d, t = cfg["data"], cfg["traps"]
    products = []
    for pid in range(1, d["n_products"] + 1):
        category = rng.choice(d["categories"])
        products.append({
            "product_id": pid,
            "name": f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} {pid}",
            "category": category,           # the correct category
            "listed_category": category,    # what products.csv says
            "price": rng.randrange(d["price_cents_min"], d["price_cents_max"]),
        })
    corrections = []
    for idx in sorted(rng.sample(range(len(products)), round(t["product_correction_share"] * len(products)))):
        p = products[idx]
        p["listed_category"] = rng.choice([c for c in d["categories"] if c != p["category"]])
        corrections.append([str(p["product_id"]), p["category"], stamp(start)])
    return products, corrections


def _build_orders(cfg, rng, start, histories, late, products, moves):
    d, t, sc = cfg["data"], cfg["traps"], cfg["showcase"]
    horizon = d["n_days"] * 86400
    sc_id = sc["customer_id"]
    moved_at = dt.datetime.fromisoformat(sc["moved_at"])
    before_span = int((moved_at - start).total_seconds())

    draws = []   # (customer_id, order_ts, kind)
    n_random = d["n_orders"] - t["boundary_orders"] - sc["orders_before"] - sc["orders_after"]
    for _ in range(n_random):
        draws.append((rng.randint(2, d["n_customers"]),
                      start + dt.timedelta(seconds=rng.randrange(horizon)), "random"))
    movers = [m for m in moves if m[0] != sc_id]
    for cid, changed_at in rng.sample(movers, t["boundary_orders"]):
        draws.append((cid, changed_at, "boundary"))
    for _ in range(sc["orders_before"]):
        draws.append((sc_id, start + dt.timedelta(seconds=rng.randrange(before_span)), "showcase"))
    for _ in range(sc["orders_after"]):
        draws.append((sc_id, moved_at + dt.timedelta(
            seconds=1 + rng.randrange(horizon - before_span - 1)), "showcase"))
    draws.sort(key=lambda x: (x[1], x[0], x[2]))

    orders = []
    for i, (cid, ts, kind) in enumerate(draws):
        n_lines = rng.choices(range(1, len(d["lines_per_order_weights"]) + 1),
                              d["lines_per_order_weights"])[0]
        lines = []
        for line_no, p_idx in enumerate(rng.sample(range(len(products)), n_lines), start=1):
            p = products[p_idx]
            qty = rng.choices([1, 2, 3, 4], d["quantity_weights"])[0]
            discount = 0
            if rng.randrange(100) < d["discount_line_share_pct"]:
                discount = qty * p["price"] * rng.choice(d["discount_pcts"]) // 100
            lines.append({"line_no": line_no, "product": p, "qty": qty, "discount": discount})
        orders.append({
            "order_id": FIRST_ORDER_ID + i, "customer_id": cid, "ts": ts, "kind": kind,
            "status": rng.choices(d["status_values"], d["status_weights"])[0],
            "shipping": rng.choices(d["shipping_cents"], d["shipping_weights"])[0],
            "lines": lines,
            "region": _region_at(histories[cid], ts),
        })
    return orders


def _resend_chunks(cfg, rng, n_lines):
    """Pick non-overlapping chunks of consecutive line rows that the upstream
    retry sends a second time, immediately after the first send."""
    t = cfg["traps"]
    target = round(t["resend_share"] * n_lines)
    covered, chunks, total = set(), [], 0
    while total < target:
        size = min(rng.randrange(t["resend_chunk_min"], t["resend_chunk_max"] + 1), target - total)
        first = rng.randrange(0, n_lines - size)
        span = range(first, first + size)
        if covered.intersection(span):
            continue
        covered.update(span)
        chunks.append((first, first + size))
        total += size
    return sorted(chunks)


def build_world(cfg):
    """Return {"raw": {table: (header, rows)}, "TRUE_REVENUE", "TRUE_SHIPPING", "TRUE_LEDGER"}.
    Raw rows are lists of strings, exactly what lands in the CSV files."""
    d = cfg["data"]
    rng = random.Random(d["seed"])
    start = dt.datetime.fromisoformat(d["start_date"])

    ids, names, histories, late, once, twice = _build_customers(cfg, rng, start)
    products, corrections = _build_products(cfg, rng, start)
    moves = sorted((cid, changed_at) for cid, h in histories.items() for changed_at, _ in h[1:])
    orders = _build_orders(cfg, rng, start, histories, late, products, moves)

    # ---- raw tables, as the OLTP export would write them ----
    changes = []
    for cid in ids:
        for changed_at, region in histories[cid]:
            changes.append([str(cid), names[cid], f"customer{cid}@example.com", region, stamp(changed_at)])
    changes.sort(key=lambda r: (r[4], int(r[0])))

    order_rows = [[str(o["order_id"]), str(o["customer_id"]), stamp(o["ts"]), o["status"],
                   money(o["shipping"])] for o in orders]
    line_rows, line_cents = [], []
    for o in orders:
        for ln in o["lines"]:
            p = ln["product"]
            line_rows.append([str(o["order_id"]), str(ln["line_no"]), str(p["product_id"]),
                              str(ln["qty"]), money(p["price"]), money(ln["discount"])])
            line_cents.append(ln["qty"] * p["price"] - ln["discount"])
    chunks = _resend_chunks(cfg, rng, len(line_rows))
    sent, ends = [], {last - 1: first for first, last in chunks}
    for i, row in enumerate(line_rows):
        sent.append(row)
        if i in ends:
            sent.extend(list(r) for r in line_rows[ends[i]:i + 1])
    product_rows = [[str(p["product_id"]), p["name"], p["listed_category"]] for p in products]

    # ---- the answer key, computed from the events, not from the raw tables ----
    by_cell, by_ship = {}, {}
    ledger = {k: 0 for k in (
        "late_orders", "late_order_lines", "late_revenue_cents",
        "boundary_order_lines", "boundary_revenue_cents", "boundary_shipping_cents",
        "relocated_orders", "relocated_lines", "relocated_revenue_cents",
        "shipping_excess_cents", "corrected_product_revenue_cents")}
    region_shift, resent_revenue = {}, 0
    current = {cid: h[-1][1] for cid, h in histories.items()}
    corrected_ids = {int(r[0]) for r in corrections}
    revenue_of = {}
    for o in orders:
        day = o["ts"].strftime("%Y-%m-%d")
        ship = by_ship.setdefault((day, o["region"]), [0, 0])
        ship[0] += 1
        ship[1] += o["shipping"]
        ledger["shipping_excess_cents"] += o["shipping"] * (len(o["lines"]) - 1)
        order_rev = 0
        for ln in o["lines"]:
            p = ln["product"]
            rev = ln["qty"] * p["price"] - ln["discount"]
            order_rev += rev
            cell = by_cell.setdefault((day, o["region"], p["category"]), [0, 0, 0])
            cell[0] += ln["qty"]
            cell[1] += 1
            cell[2] += rev
            if p["product_id"] in corrected_ids:
                ledger["corrected_product_revenue_cents"] += rev
        revenue_of[o["order_id"]] = order_rev
        if o["kind"] == "boundary":
            ledger["boundary_order_lines"] += len(o["lines"])
            ledger["boundary_revenue_cents"] += order_rev
            ledger["boundary_shipping_cents"] += o["shipping"]
        if o["customer_id"] in late and o["ts"] < histories[o["customer_id"]][0][0]:
            ledger["late_orders"] += 1
            ledger["late_order_lines"] += len(o["lines"])
            ledger["late_revenue_cents"] += order_rev
        if o["region"] != current[o["customer_id"]]:
            ledger["relocated_orders"] += 1
            ledger["relocated_lines"] += len(o["lines"])
            ledger["relocated_revenue_cents"] += order_rev
            region_shift[o["region"]] = region_shift.get(o["region"], 0) - order_rev
            region_shift[current[o["customer_id"]]] = region_shift.get(current[o["customer_id"]], 0) + order_rev
    for first, last in chunks:
        resent_revenue += sum(line_cents[first:last])

    sc = cfg["showcase"]
    showcase = [{
        "order_id": o["order_id"], "order_ts": stamp(o["ts"]),
        "amount_cents": revenue_of[o["order_id"]],
        "true_region": o["region"], "current_region": current[o["customer_id"]]}
        for o in orders if o["customer_id"] == sc["customer_id"]]

    ledger.update({
        "n_customers": len(ids), "n_products": len(products), "n_orders": len(orders),
        "n_lines": len(line_rows), "n_line_rows_sent": len(sent),
        "n_customer_change_rows": len(changes), "n_product_corrections": len(corrections),
        "customers_relocating_once": len(once), "customers_relocating_twice": len(twice),
        "move_events": len(moves), "late_customers": len(late),
        "boundary_orders": cfg["traps"]["boundary_orders"],
        "resent_rows": len(sent) - len(line_rows), "resent_chunks": len(chunks),
        "resent_revenue_cents": resent_revenue,
        "true_line_revenue_cents": sum(c[2] for c in by_cell.values()),
        "true_shipping_cents": sum(v[1] for v in by_ship.values()),
        "relocation_region_shift_cents": dict(sorted(region_shift.items())),
        "showcase_orders": showcase,
    })
    true_revenue = [[day, region, cat, v[0], v[1], v[2]] for (day, region, cat), v in sorted(by_cell.items())]
    true_shipping = [[day, region, v[0], v[1]] for (day, region), v in sorted(by_ship.items())]
    return {
        "raw": {
            "customer_changes": (CUSTOMER_HEADER, changes),
            "orders": (ORDER_HEADER, order_rows),
            "order_lines": (LINE_HEADER, sent),
            "products": (PRODUCT_HEADER, product_rows),
            "product_corrections": (CORRECTION_HEADER, corrections),
        },
        "TRUE_REVENUE": true_revenue,
        "TRUE_SHIPPING": true_shipping,
        "TRUE_LEDGER": ledger,
    }

"""Compare a built report to the true figures. Pure functions, no file access.

WHY exact Decimal arithmetic: the whole project argues that money must not be a
float. A scorer that rounded or used a tolerance would hide the float mutation,
so every difference here is exact, down to a fraction of a cent.

The notebook embeds this file as it is, so keep it dependency free.
"""
from decimal import Decimal


def cents_to_decimal(cents):
    return Decimal(cents).scaleb(-2)


def _label(value):
    return "(null)" if value is None else str(value)


def _summarise(true_total, report_total, diffs, extra):
    abs_error = sum((abs(d) for d in diffs.values()), Decimal(0))
    return {
        "true_total": true_total, "report_total": report_total,
        "net_error": report_total - true_total, "abs_error": abs_error,
        "pct_of_true": (abs_error / true_total * 100) if true_total else Decimal(0),
        **extra,
    }


def _group_net(rows_by_cell, position):
    out = {}
    for cell, value in rows_by_cell.items():
        out[cell[position]] = out.get(cell[position], Decimal(0)) + value
    return out


def _worst(net_by_group):
    if not net_by_group:
        return ("-", Decimal(0))
    name = max(sorted(net_by_group), key=lambda k: abs(net_by_group[k]))
    return (name, net_by_group[name])


def score_revenue(report_rows, true_rows):
    """report_rows: (order_date, region, category, units, order_lines, line_revenue).
    true_rows: [order_date, region, category, units, order_lines, revenue_cents]."""
    true = {(r[0], r[1], r[2]): (r[3], r[4], cents_to_decimal(r[5])) for r in true_rows}
    got = {}
    for day, region, cat, units, lines, revenue in report_rows:
        key = (str(day), _label(region), _label(cat))
        u, n, rev = got.get(key, (0, 0, Decimal(0)))
        got[key] = (u + int(units), n + int(lines), rev + Decimal(str(revenue)))
    keys = sorted(set(true) | set(got))
    zero = (0, 0, Decimal(0))
    diffs = {k: got.get(k, zero)[2] - true.get(k, zero)[2] for k in keys}
    wrong = [k for k in keys if got.get(k, zero) != true.get(k, zero)]
    by_region, by_category = _group_net(diffs, 1), _group_net(diffs, 2)
    summary = _summarise(
        sum((v[2] for v in true.values()), Decimal(0)),
        sum((v[2] for v in got.values()), Decimal(0)), diffs,
        {"cells_true": len(true), "cells_wrong": len(wrong),
         "cells_missing": len(set(true) - set(got)), "cells_extra": len(set(got) - set(true)),
         "units_error": sum(v[0] for v in got.values()) - sum(v[0] for v in true.values()),
         "lines_error": sum(v[1] for v in got.values()) - sum(v[1] for v in true.values()),
         "by_region": by_region, "by_category": by_category,
         "true_by_region": _group_net({k: v[2] for k, v in true.items()}, 1),
         "report_by_region": _group_net({k: v[2] for k, v in got.items()}, 1),
         "worst_region": _worst(by_region), "worst_category": _worst(by_category)})
    summary["exact"] = not wrong
    return summary


def score_shipping(report_rows, true_rows):
    """report_rows: (order_date, region, orders, shipping_revenue).
    true_rows: [order_date, region, orders, shipping_cents]."""
    true = {(r[0], r[1]): (r[2], cents_to_decimal(r[3])) for r in true_rows}
    got = {}
    for day, region, orders, shipping in report_rows:
        key = (str(day), _label(region))
        n, amount = got.get(key, (0, Decimal(0)))
        got[key] = (n + int(orders), amount + Decimal(str(shipping)))
    keys = sorted(set(true) | set(got))
    zero = (0, Decimal(0))
    diffs = {k: got.get(k, zero)[1] - true.get(k, zero)[1] for k in keys}
    wrong = [k for k in keys if got.get(k, zero) != true.get(k, zero)]
    by_region = _group_net(diffs, 1)
    summary = _summarise(
        sum((v[1] for v in true.values()), Decimal(0)),
        sum((v[1] for v in got.values()), Decimal(0)), diffs,
        {"cells_true": len(true), "cells_wrong": len(wrong),
         "by_region": by_region, "worst_region": _worst(by_region)})
    summary["exact"] = not wrong
    return summary

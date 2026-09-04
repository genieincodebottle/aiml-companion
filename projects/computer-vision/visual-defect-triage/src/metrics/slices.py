"""Per-slice accuracy, ranked by improvement ceiling rather than by error rate."""
from collections import defaultdict


def slice_report(records: list[dict], key: str) -> list[dict]:
    """records: [{"correct": bool, key: str}]

    'ceiling' is share x error rate, the amount overall accuracy would gain if
    this slice became perfect. It ranks the work, and it is frequently NOT the
    slice with the worst rate. Sorted by it here on purpose.
    """
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        buckets[r[key]].append(bool(r["correct"]))

    total = len(records)
    rows = []
    for name, hits in buckets.items():
        n = len(hits)
        acc = sum(hits) / n
        rows.append({
            "slice": name,
            "n": n,
            "share": n / total,
            "accuracy": acc,
            "ceiling": (n / total) * (1.0 - acc),
        })
    return sorted(rows, key=lambda r: -r["ceiling"])


def ceilings_sum_to_error_budget(rows: list[dict], overall_accuracy: float,
                                 tol: float = 1e-9) -> bool:
    """Every slice ceiling is a piece of the same error budget, so they must add
    up to 1 - overall accuracy. This is the identity that makes the ranking
    trustworthy, and it is asserted rather than assumed."""
    return abs(sum(r["ceiling"] for r in rows) - (1.0 - overall_accuracy)) < tol

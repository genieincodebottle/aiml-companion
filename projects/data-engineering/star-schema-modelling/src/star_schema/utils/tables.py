"""Plain ASCII tables. The console on Windows is often cp1252, so nothing here
prints a character outside ASCII."""
from __future__ import annotations

from decimal import Decimal


def money(cents: int | Decimal | None) -> str:
    """Format cents or a Decimal amount as 1,234.56 (no currency symbol)."""
    if cents is None:
        return "-"
    amount = Decimal(cents) / 100 if isinstance(cents, int) else Decimal(cents)
    return f"{amount:,.2f}"


def render(rows: list[list], header: list[str], align: str | None = None) -> str:
    """align is one char per column, l or r. Default is l for the first, r for the rest."""
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(header)]
    align = align or "l" + "r" * (len(header) - 1)

    def line(vals):
        return "  ".join(v.ljust(w) if a == "l" else v.rjust(w)
                         for v, w, a in zip(vals, widths, align))

    out = [line(header), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in cells]
    return "\n".join(out)

"""ASCII tables. No box-drawing characters, so a Windows cp1252 console never chokes."""
from __future__ import annotations


def money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    c = abs(int(cents))
    return f"{sign}{c // 100:,}.{c % 100:02d}"


def table(headers: list[str], rows: list[list], align: list[str] | None = None) -> str:
    """Left-align the first column, right-align the rest, unless `align` says otherwise ('l'/'r')."""
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(headers)]
    align = align or ["l"] + ["r"] * (len(headers) - 1)

    def fmt(row):
        return " | ".join(c.ljust(w) if a == "l" else c.rjust(w)
                          for c, w, a in zip(row, widths, align))

    rule = "-+-".join("-" * w for w in widths)
    return "\n".join([fmt(headers), rule] + [fmt(r) for r in cells])

"""RUBRIC: you can load their data without silently losing rows.

These fail until you implement src/ingest/loader.py.

The bar is not "it parses". The bar is that you can state, with a number, what
came in and what came out, and that the difference is accounted for rather than
discovered later by a customer.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingest.loader import IngestResult, load_shipments, unify

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "customer" / "data"

EXPORTS = [DATA / "shipments_2024.csv", DATA / "shipments_2025.csv"]


@pytest.mark.parametrize("path", EXPORTS, ids=lambda p: p.name)
def test_reads_every_source_row(path: Path):
    """rows_in must be the true source count: 400 in both files.

    Counting the source is the part people skip, and it is the only thing that
    makes the reconciliation claim meaningful.
    """
    result = load_shipments(path)
    assert isinstance(result, IngestResult)
    assert result.rows_in == 400, "count the source rows before parsing them"


@pytest.mark.parametrize("path", EXPORTS, ids=lambda p: p.name)
def test_nothing_disappears(path: Path):
    """The identity that makes silent loss impossible.

    Every source row is either a record or an explicitly rejected row. There is no
    third category, and "it threw a warning" is not one.
    """
    result = load_shipments(path)
    assert result.reconciles(), (
        f"{result.rows_in} in, {result.rows_out} out, {len(result.rejected)} rejected. "
        "Those numbers have to add up."
    )


def test_rejected_rows_are_explained():
    """A reject with no reason is a drop with extra steps.

    The 2025 export contains rows whose field count does not match the header. You
    may reject them or repair them, but if you reject them the record has to say
    why and keep the raw line so a human can look.
    """
    result = load_shipments(DATA / "shipments_2025.csv")
    if not result.rejected:
        pytest.skip("you repaired every row rather than rejecting any, which is allowed")
    for row in result.rejected:
        assert row.reason.strip(), "every rejection needs a specific reason"
        assert row.raw.strip(), "keep the raw line so a human can inspect it"
        assert row.line_number > 0


def test_embedded_delimiters_do_not_split_a_field():
    """Addresses like "12 Main St, Unit 4" contain the delimiter.

    Any record whose address field got truncated at the comma is a parsing bug that
    will show up as a wrong delivery address in front of a customer.
    """
    result = load_shipments(DATA / "shipments_2025.csv")
    addresses = [
        str(rec.get("dest_address", "")) for rec in result.records
    ]
    assert any("," in addr for addr in addresses), (
        "no address contains a comma, so quoted fields were split. "
        "Use the csv module rather than str.split."
    )


def test_nulls_are_normalised():
    """Nulls arrive as "", NULL, N/A, -, and the literal string nan.

    Five representations of absent is four too many. Pick one and convert.
    """
    result = load_shipments(DATA / "shipments_2025.csv")
    seen = {
        str(value).strip()
        for rec in result.records
        for value in rec.values()
    }
    leaked = {"NULL", "N/A", "nan", "-"} & seen
    assert not leaked, f"un-normalised null markers still present: {sorted(leaked)}"


def test_unify_preserves_every_record():
    """Merging the years must not lose anything either."""
    results = [load_shipments(p) for p in EXPORTS]
    merged = unify(results)
    assert len(merged) == sum(r.rows_out for r in results), (
        "unify dropped records. A concat that silently misaligns columns is the "
        "usual cause."
    )


def test_unify_makes_the_join_key_comparable():
    """The trap: 2024 stores 4521, 2025 stores "0004521".

    Read them naively and pandas coerces both to int, which joins rows that should
    not match. Read them defensively as strings and nothing matches at all. Both
    are wrong, which is why this needs a deliberate decision rather than a default.

    After unify, the same customer must be recognisable across both years.
    """
    results = [load_shipments(p) for p in EXPORTS]
    merged = unify(results)

    key = None
    for candidate in ("customer_id", "cust_id"):
        if merged and candidate in merged[0]:
            key = candidate
            break
    assert key, "unify must settle on one canonical customer column name"

    by_year: dict[str, set[str]] = {}
    for rec in merged:
        year = str(rec.get("source_year") or rec.get("year") or "")
        assert year, "keep the source year on each record so you can tell them apart"
        by_year.setdefault(year, set()).add(str(rec[key]))

    assert len(by_year) == 2, "both years should be present after unify"
    left, right = by_year.values()
    assert left & right, (
        "no customer appears in both years, so the id formats never reconciled"
    )

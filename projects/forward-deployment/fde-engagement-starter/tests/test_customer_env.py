"""Smoke test for the customer environment.

THIS FILE SHOULD PASS ON A FRESH CLONE. If it does not, fix your setup before
touching anything else: every other test assumes this environment works.

It also documents, in executable form, what you have been handed. Reading it is
faster than reading the CSVs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "customer" / "data"
POLICIES = DATA / "policies"
TICKETS = ROOT / "customer" / "tickets.jsonl"


def test_exports_exist():
    assert (DATA / "shipments_2024.csv").exists(), "run `make customer-seed`"
    assert (DATA / "shipments_2025.csv").exists(), "run `make customer-seed`"


def test_each_export_has_400_data_rows():
    """The number every ingest run has to reconcile against.

    If your loader returns fewer than 400 records plus rejects for either file,
    you dropped data.
    """
    for name in ("shipments_2024.csv", "shipments_2025.csv"):
        raw = (DATA / name).read_bytes().decode("utf-8", errors="replace")
        lines = [line for line in raw.splitlines() if line.strip()]
        assert len(lines) - 1 == 400, f"{name} should have 400 data rows"


def test_2025_export_starts_with_a_byte_order_mark():
    """A real defect, not a trick.

    Naive `open()` puts the BOM into the first column name, so `row["shipment_id"]`
    raises KeyError on a file that looks fine in a text editor.
    """
    assert (DATA / "shipments_2025.csv").read_bytes()[:3] == b"\xef\xbb\xbf"


def test_the_two_exports_disagree_about_schema():
    """Silent schema drift between years. This is the join bug waiting to happen."""
    # Decode only the header line. Reading either file whole with a strict codec
    # blows up on the latin-1 rows further down, which is itself instructive: be
    # careful what you assume about a file you have only looked at the top of.
    def header(name: str) -> str:
        first = (DATA / name).read_bytes().split(b"\n", 1)[0]
        return first.decode("utf-8-sig", errors="replace").strip()

    h24 = header("shipments_2024.csv")
    h25 = header("shipments_2025.csv")
    assert "customer_id" in h24
    assert "cust_id" in h25
    assert "customer_id" not in h25, "2025 renamed the column"
    assert len(h25.split(",")) > len(h24.split(",")), "2025 added a column"


def test_naive_pandas_read_does_not_survive_the_2025_export():
    """Proof that the shortcut fails, so you do not have to discover it in a demo.

    This test passes when the naive read FAILS. If pandas ever starts handling
    this file silently, that is worse news, not better.
    """
    pd = pytest.importorskip("pandas")
    with pytest.raises(Exception):
        pd.read_csv(DATA / "shipments_2025.csv")


def test_policy_corpus_present():
    docs = sorted(POLICIES.glob("*"))
    assert len(docs) == 5, "five policy documents"
    for doc in docs:
        assert len(doc.read_text(encoding="utf-8").split()) > 100


def test_tickets_are_readable_jsonl():
    lines = TICKETS.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 120
    first = json.loads(lines[0])
    assert {"ticket_id", "created_at", "channel", "text", "resolution"} <= set(first)


def test_tickets_contain_enough_material_for_a_golden_set():
    """The eval gate needs 30 cases across 5 failure modes.

    Nothing labels these tickets with a failure mode. Deriving the labels is the
    work, and it is the hour that makes the difference between an eval harness and
    a spreadsheet.
    """
    texts = [
        json.loads(line)["text"].lower()
        for line in TICKETS.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(texts) >= 30
    assert sum(1 for t in texts if "address" in t) >= 3

"""The generated source system is deterministic, and every trap is planted at the
size the config asks for."""
import csv
import re

import pytest

from conftest import temp_config
from star_schema.config import load_config
from star_schema.data.io import write_world
from star_schema.data.world import build_world

# Pinned so a change to the generator is a conscious one. The files are written
# with explicit "\n" line endings, so these are the same on Windows, macOS and Linux.
RAW_SHA256 = {
    "customer_changes.csv": "f0910643c69ba96f2cf63956284bab89dd92ef9a0953edc57eb21ea05cf1e19b",
    "order_lines.csv": "2ab0ee1d74ca2469156fb33a5db3d8611d9add67b6bed3c6d325b87ad27037cf",
    "orders.csv": "d95eb764fa8c89da95487f74bd93bc2af5c2628f4bbcfd1a96953ed9bb2f21d2",
    "product_corrections.csv": "a7baa0bb28c8bef52aea7bfcc79f55d1d3d6ddfc7d1cf413df383a39fffbf6a3",
    "products.csv": "ac05618a3ede1c414c63be158741fa483911725b8647989e8a7b2951c867cd09",
}


def read_csv(cfg, name):
    path = cfg["paths"]["raw_dir"]
    with open(f"{path}/{name}.csv", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_same_seed_gives_the_same_world():
    cfg = load_config()
    assert build_world(cfg) == build_world(cfg)


def test_raw_files_are_byte_identical_across_writes(cfg, raw_hashes, tmp_path):
    again = write_world(temp_config(tmp_path))
    assert again == raw_hashes
    assert raw_hashes == RAW_SHA256


def test_a_different_seed_changes_the_world():
    cfg = load_config()
    other = load_config()
    other["data"]["seed"] = 7
    assert build_world(cfg)["TRUE_LEDGER"] != build_world(other)["TRUE_LEDGER"]


def test_source_sizes(cfg, ledger, raw_hashes):
    assert ledger["n_customers"] == 1500 and ledger["n_products"] == 400
    assert ledger["n_orders"] == 12000
    assert ledger["n_lines"] == 29902
    assert len(read_csv(cfg, "customer_changes")) == ledger["n_customer_change_rows"] == 1950
    assert len(read_csv(cfg, "order_lines")) == ledger["n_line_rows_sent"] == 30500


def test_every_trap_is_planted_at_the_configured_size(cfg, ledger, raw_hashes):
    c = load_config()["traps"]
    assert ledger["late_customers"] == round(c["late_customer_share"] * 1500) == 60
    # The showcase customer is one of the 300 customers who move once.
    assert ledger["customers_relocating_once"] == round(c["relocate_once_share"] * 1500) == 300
    assert ledger["customers_relocating_twice"] == round(c["relocate_twice_share"] * 1500) == 75
    assert ledger["move_events"] == 300 + 2 * 75
    assert ledger["boundary_orders"] == c["boundary_orders"] == 60
    assert ledger["resent_rows"] == round(c["resend_share"] * ledger["n_lines"]) == 598
    assert ledger["n_product_corrections"] == round(c["product_correction_share"] * 400) == 12
    assert ledger["late_orders"] == 182
    assert ledger["relocated_orders"] == 1513


def test_resent_rows_are_exact_duplicates(cfg, ledger, raw_hashes):
    rows = read_csv(cfg, "order_lines")
    seen = {}
    duplicates = 0
    for row in rows:
        key = (row["order_id"], row["line_no"])
        if key in seen:
            assert seen[key] == row, "a re-sent row differs from its original"
            duplicates += 1
        seen[key] = row
    assert duplicates == ledger["resent_rows"]
    assert len(seen) == ledger["n_lines"]


def test_boundary_orders_sit_on_the_second_a_region_changed(cfg, ledger, raw_hashes):
    changes = {(r["customer_id"], r["updated_at"]) for r in read_csv(cfg, "customer_changes")}
    first_rows = {}
    for r in read_csv(cfg, "customer_changes"):
        first_rows[r["customer_id"]] = min(first_rows.get(r["customer_id"], r["updated_at"]), r["updated_at"])
    on_boundary = [o for o in read_csv(cfg, "orders")
                   if (o["customer_id"], o["order_ts"]) in changes
                   and first_rows[o["customer_id"]] != o["order_ts"]]
    assert len(on_boundary) == ledger["boundary_orders"] == 60


def test_late_customers_have_orders_before_their_first_row(cfg, ledger, raw_hashes):
    first = {}
    for r in read_csv(cfg, "customer_changes"):
        first[r["customer_id"]] = min(first.get(r["customer_id"], r["updated_at"]), r["updated_at"])
    early = [o for o in read_csv(cfg, "orders") if o["order_ts"] < first[o["customer_id"]]]
    assert len(early) == ledger["late_orders"] == 182
    assert len({o["customer_id"] for o in early}) <= ledger["late_customers"]


def test_the_showcase_customer_moves_and_orders_on_both_sides(cfg, ledger, raw_hashes):
    sc = load_config()["showcase"]
    orders = ledger["showcase_orders"]
    before = [o for o in orders if o["order_ts"] < sc["moved_at"]]
    after = [o for o in orders if o["order_ts"] >= sc["moved_at"]]
    assert len(before) == sc["orders_before"] and len(after) == sc["orders_after"]
    assert {o["true_region"] for o in before} == {sc["region_from"]}
    assert {o["true_region"] for o in after} == {sc["region_to"]}
    assert {o["current_region"] for o in orders} == {sc["region_to"]}


def test_money_is_written_as_exact_cents(cfg, raw_hashes):
    pattern = re.compile(r"^\d+\.\d{2}$")
    for row in read_csv(cfg, "order_lines"):
        assert pattern.match(row["unit_price"]) and pattern.match(row["discount"])
    for row in read_csv(cfg, "orders"):
        assert pattern.match(row["shipping_cost"])


def test_the_raw_files_hold_no_answer_key_columns(cfg, raw_hashes):
    for name in ("customer_changes", "orders", "order_lines", "products", "product_corrections"):
        header = " ".join(read_csv(cfg, name)[0])
        assert "true" not in header.lower()


@pytest.mark.parametrize("name", ["true_revenue.json", "true_shipping.json", "trap_ledger.json"])
def test_the_key_is_written_outside_the_raw_folder(cfg, raw_hashes, name):
    from pathlib import Path
    assert (Path(cfg["paths"]["key_dir"]) / name).exists()
    assert not (Path(cfg["paths"]["raw_dir"]) / name).exists()

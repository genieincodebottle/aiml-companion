"""Regression tests on the project's actual claims. If a lesson stops being true,
the README becomes wrong, so the lessons are asserted, not narrated.

The numbers below are the ones the README quotes. They come from the seeded
generator, so they are the same on every run."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from star_schema import variants as V
from star_schema.config import ROOT
from star_schema.pipelines import matrix as M
from star_schema.pipelines import report as R
from star_schema.verdict import classify, is_silent, tiers_firing

D = Decimal

# ---------------------------------------------------------------- the build
LAYER_ROWS = {
    "stg_customer_changes": 1950, "stg_orders": 12000, "stg_order_lines": 29902,
    "stg_products": 400, "dim_customer": 1950, "dim_product": 400,
    "fct_order": 12000, "fct_order_line": 29902,
    "rpt_revenue_by_region_category_day": 5372, "rpt_shipping_by_region_day": 905,
}


def test_the_correct_model_passes_every_test(baseline):
    assert baseline["run_ok"]
    assert len(baseline["tests"]) == 64
    assert [t["label"] for t in baseline["tests"] if t["status"] != "pass"] == []
    by_tier = {tier: sum(1 for t in baseline["tests"] if t["tier"] == tier)
               for tier in ("generic", "singular", "added")}
    assert by_tier == {"generic": 53, "singular": 4, "added": 7}


def test_the_correct_model_matches_the_answer_key_exactly(baseline):
    revenue, shipping = baseline["score"]["revenue"], baseline["score"]["shipping"]
    assert revenue["exact"] and shipping["exact"]
    assert revenue["abs_error"] == 0 and shipping["abs_error"] == 0
    assert revenue["true_total"] == D("4758558.74")
    assert shipping["true_total"] == D("62111.62")
    assert revenue["cells_true"] == 5372 and revenue["cells_wrong"] == 0
    assert shipping["cells_true"] == 905 and shipping["cells_wrong"] == 0


def test_row_counts_per_layer(baseline, ledger):
    assert baseline["rows"] == LAYER_ROWS
    # Grain. The line fact has one row per distinct (order_id, line_no), so the
    # 598 re-sent rows are gone, and the order fact has one row per order.
    assert baseline["rows"]["fct_order_line"] == ledger["n_lines"]
    assert baseline["rows"]["fct_order_line"] == ledger["n_line_rows_sent"] - ledger["resent_rows"]
    assert baseline["rows"]["fct_order"] == ledger["n_orders"]
    # Type 2 keeps one row per version the source reported.
    assert baseline["rows"]["dim_customer"] == ledger["n_customer_change_rows"]
    assert ledger["n_customer_change_rows"] == ledger["n_customers"] + ledger["move_events"]
    # Type 1 keeps one row per product.
    assert baseline["rows"]["dim_product"] == ledger["n_products"]


def test_money_columns_are_decimal(cfg, baseline):
    db = Path(cfg["paths"]["artifacts"]) / "work" / "baseline" / "warehouse.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    for table, column in (("fct_order_line", "line_revenue"), ("fct_order_line", "unit_price"),
                          ("fct_order_line", "discount"), ("fct_order", "shipping_cost"),
                          ("rpt_revenue_by_region_category_day", "line_revenue"),
                          ("rpt_shipping_by_region_day", "shipping_revenue")):
        kind = con.execute(f"select typeof({column}) from {table} limit 1").fetchone()[0]
        assert kind.startswith("DECIMAL"), (table, column, kind)
    con.close()


def test_the_shipped_models_never_mention_a_float_type():
    for path in (ROOT / "warehouse" / "models").rglob("*.sql"):
        text = path.read_text(encoding="utf-8").lower()
        assert not any(word in text for word in ("double", "float", "real)")), path.name


def test_a_second_build_gives_identical_tables(cfg, baseline, tmp_path):
    """Same seed, same SQL, same bytes. Hash every table in a second, separate build."""
    second_cfg = dict(cfg, paths=dict(cfg["paths"], artifacts=str(tmp_path / "second")))
    second = M.build_variant(None, second_cfg)
    assert second["rows"] == baseline["rows"]
    assert second["score"] == baseline["score"]

    def digest(artifacts):
        con = duckdb.connect(str(Path(artifacts) / "work" / "baseline" / "warehouse.duckdb"), read_only=True)
        out = {}
        for table in LAYER_ROWS:
            rows = con.execute(f"select * from {table} order by all").fetchall()
            out[table] = hashlib.sha256(repr(rows).encode()).hexdigest()
        con.close()
        return out

    assert digest(cfg["paths"]["artifacts"]) == digest(tmp_path / "second")


def test_the_shipped_dbt_project_is_never_edited(project_hash_before_and_after):
    _, before, after = project_hash_before_and_after
    assert before == after


# ------------------------------------------------------- the naive table
def _score(results, variant, measure):
    return results[variant]["score"][measure]


def test_relocation_moves_revenue_to_the_current_region(results, ledger):
    s = _score(results, "m01_join_on_is_current", "revenue")
    assert s["net_error"] == 0, "no revenue is lost, it only moves region"
    assert s["abs_error"] == D("834154.02")
    assert round(s["pct_of_true"], 2) == D("17.53")
    # The region errors equal what the generator says it moved.
    shift = {r: D(c).scaleb(-2) for r, c in ledger["relocation_region_shift_cents"].items()}
    assert s["by_region"] == shift
    assert s["by_region"]["North"] == D("26798.25") and s["by_region"]["West"] == D("-26734.95")
    assert ledger["relocated_revenue_cents"] == 59365828  # 593,658.28 sits in the wrong region
    assert not s["exact"]


def test_type_one_customer_overwrite_is_the_same_mistake(results):
    m01 = _score(results, "m01_join_on_is_current", "revenue")
    m02 = _score(results, "m02_type1_customer", "revenue")
    assert m02["abs_error"] == m01["abs_error"] and m02["by_region"] == m01["by_region"]


def test_shipping_on_the_line_fact_overstates_shipping(results, ledger):
    ship = _score(results, "m03_shipping_on_line_fact", "shipping")
    assert ship["net_error"] == D(ledger["shipping_excess_cents"]).scaleb(-2) == D("92163.41")
    assert round(ship["pct_of_true"], 2) == D("148.38")
    assert ship["report_total"] == D("154275.03")
    assert _score(results, "m03_shipping_on_line_fact", "revenue")["exact"], \
        "line revenue is untouched, only shipping is wrong"


def test_a_resent_chunk_inflates_revenue(results, ledger):
    s = _score(results, "m04_drop_dedup", "revenue")
    assert s["net_error"] == D(ledger["resent_revenue_cents"]).scaleb(-2) == D("93458.45")
    assert round(s["pct_of_true"], 2) == D("1.96")
    assert ledger["resent_rows"] == 598


def test_an_inner_join_silently_loses_late_customers(results, ledger):
    s = _score(results, "m05_inner_join_late_customers", "revenue")
    assert s["net_error"] == -D(ledger["late_revenue_cents"]).scaleb(-2) == D("-73341.63")
    assert round(s["pct_of_true"], 2) == D("1.54")
    assert _score(results, "m05_inner_join_late_customers", "shipping")["net_error"] == D("-932.51")


def test_an_inclusive_window_fans_out_the_boundary_orders(results, ledger):
    rev = _score(results, "m07_inclusive_window_join", "revenue")
    ship = _score(results, "m07_inclusive_window_join", "shipping")
    assert rev["net_error"] == D(ledger["boundary_revenue_cents"]).scaleb(-2) == D("24485.14")
    assert ship["net_error"] == D(ledger["boundary_shipping_cents"]).scaleb(-2) == D("314.54")
    assert results["m07_inclusive_window_join"]["rows"]["fct_order"] == 12000 + 60
    assert results["m07_inclusive_window_join"]["rows"]["fct_order_line"] == 29902 + 148


def test_the_relocated_customer_example(cfg, baseline, ledger):
    """The same eight orders, attributed by the window and by is_current."""
    db = Path(cfg["paths"]["artifacts"]) / "work" / "baseline" / "warehouse.duckdb"
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute("""
        select o.order_id, sum(l.line_revenue), d.region, cur.region
        from fct_order o
        join dim_customer d on d.customer_sk = o.customer_sk
        join dim_customer cur on cur.customer_id = d.customer_id and cur.is_current
        join fct_order_line l on l.order_id = o.order_id
        where d.customer_id = 1 group by 1, 3, 4 order by 1""").fetchall()
    con.close()
    truth = {o["order_id"]: o for o in ledger["showcase_orders"]}
    assert len(rows) == 8
    assert all(region == truth[oid]["true_region"] for oid, _, region, _ in rows)
    wrong = [(oid, amount) for oid, amount, _, current in rows if current != truth[oid]["true_region"]]
    assert [oid for oid, _ in wrong] == [102348, 103715, 104099, 104776]
    assert sum(amount for _, amount in wrong) == D("1841.38")


# ------------------------------------------------------ the mutation matrix
EXPECTED = {
    "m01_join_on_is_current": ("singular only", False, {"assert_fact_inside_customer_window": 5535}),
    "m02_type1_customer": ("added only", True, {"assert_dim_customer_keeps_every_source_version": 1}),
    "m03_shipping_on_line_fact": ("added only", True, {"assert_report_reconciles_to_facts": 1}),
    "m04_drop_dedup": ("generic", False, {
        "assert_staging_keeps_every_distinct_source_line": 1,
        "unique_combination_of_columns[fct_order_line](order_id,line_no)": 598,
        "unique_combination_of_columns[stg_order_lines](order_id,line_no)": 598}),
    "m05_inner_join_late_customers": ("singular only", False, {
        "assert_fact_row_counts_match_staging": 2,
        "assert_line_revenue_reconciles_to_staging": 1,
        "assert_shipping_reconciles_to_staging": 1}),
    "m06_no_backdate_left_join": ("generic", False, {
        "not_null[fct_order.customer_sk]": 182,
        "not_null[fct_order_line.customer_sk]": 435,
        "not_null[rpt_revenue_by_region_category_day.region]": 270,
        "not_null[rpt_shipping_by_region_day.region]": 86}),
    "m07_inclusive_window_join": ("generic", False, {
        "assert_fact_row_counts_match_staging": 2,
        "unique[fct_order.order_id]": 60,
        "unique_combination_of_columns[fct_order_line](order_id,line_no)": 148,
        "assert_fact_inside_customer_window": 208,
        "assert_line_revenue_reconciles_to_staging": 1,
        "assert_shipping_reconciles_to_staging": 1}),
    "m08_dedup_on_wrong_key": ("added only", True, {"assert_staging_keeps_every_distinct_source_line": 1}),
    "m09_float_money": ("singular only", False, {
        "assert_report_reconciles_to_facts": 2,
        "column_is_decimal[fct_order.shipping_cost]": 1,
        "column_is_decimal[fct_order_line.line_revenue]": 1,
        "assert_line_revenue_reconciles_to_staging": 1,
        "assert_shipping_reconciles_to_staging": 1}),
    "m10_wrong_grain": ("added only", True, {"assert_fact_row_counts_match_staging": 1}),
    "m11_stale_is_current": ("singular only", False, {"assert_one_current_row_per_customer": 375}),
    "m12_skip_product_correction": ("added only", True, {"assert_product_corrections_applied": 12}),
    "m13_timezone_shift": ("nothing", True, {}),
}


def test_every_variant_is_in_the_matrix():
    assert [v["id"] for v in V.VARIANTS] == list(EXPECTED)
    assert len(V.VARIANTS) == 13


@pytest.mark.parametrize("variant", list(EXPECTED))
def test_which_tests_fire_for_each_mutation(results, variant):
    caught_by, silent, failing = EXPECTED[variant]
    r = results[variant]
    assert r["run_ok"], r["run_errors"]
    assert {t["label"]: t["failures"] for t in r["tests"] if t["status"] != "pass"} == failing
    assert classify(r) == caught_by
    assert is_silent(r) is silent


@pytest.mark.parametrize("variant", list(EXPECTED))
def test_every_mutation_changes_the_answer_or_a_test(results, variant):
    """A mutation that moves neither a test nor a number would prove nothing."""
    r = results[variant]
    answer_exact = r["score"]["revenue"]["exact"] and r["score"]["shipping"]["exact"]
    assert (not answer_exact) or any(t["status"] != "pass" for t in r["tests"])


def test_bucket_counts(results):
    counts = R.bucket_counts(list(results.values()))
    assert counts == {"generic": 3, "singular only": 4, "added only": 5, "nothing": 1,
                      "typical suite misses": 6, "silent": 6}


def test_how_many_mutations_each_tier_notices(results):
    """Generic tests fire for 3 of 13 mutations, singular tests for 5."""
    generic = {v for v in EXPECTED if tiers_firing(results[v])["generic"]}
    singular = {v for v in EXPECTED if tiers_firing(results[v])["singular"]}
    added = {v for v in EXPECTED if tiers_firing(results[v])["added"]}
    assert generic == {"m04_drop_dedup", "m06_no_backdate_left_join", "m07_inclusive_window_join"}
    assert singular == {"m01_join_on_is_current", "m05_inner_join_late_customers",
                        "m07_inclusive_window_join", "m09_float_money", "m11_stale_is_current"}
    assert len(added) == 9


def test_a_reconciliation_to_a_wrong_staging_model_passes(results):
    """Drop the dedup and staging inflates with the fact, so the fact still equals
    staging. Only the test that goes back to the raw table notices."""
    failing = {t["label"] for t in results["m04_drop_dedup"]["tests"] if t["status"] != "pass"}
    assert "assert_line_revenue_reconciles_to_staging" not in failing
    assert "assert_staging_keeps_every_distinct_source_line" in failing


def test_a_type_one_customer_dimension_passes_every_structural_test(results):
    """One current row per customer, windows containing every order, no nulls."""
    r = results["m02_type1_customer"]
    assert tiers_firing(r)["generic"] == [] and tiers_firing(r)["singular"] == []
    assert not r["score"]["revenue"]["exact"]


def test_an_inner_join_hides_from_the_generic_tests_but_a_left_join_does_not(results):
    assert not tiers_firing(results["m05_inner_join_late_customers"])["generic"]
    assert tiers_firing(results["m06_no_backdate_left_join"])["generic"]


def test_float_money_is_wrong_by_less_than_a_cent(results):
    s = results["m09_float_money"]["score"]
    assert not s["revenue"]["exact"]
    assert s["revenue"]["abs_error"] < D("1e-9") and s["shipping"]["abs_error"] < D("1e-9")
    fired = {t["label"] for t in results["m09_float_money"]["tests"] if t["status"] != "pass"}
    assert {"column_is_decimal[fct_order.shipping_cost]",
            "column_is_decimal[fct_order_line.line_revenue]"} <= fired


def test_the_stale_is_current_flag_breaks_no_number_yet(results):
    r = results["m11_stale_is_current"]
    assert r["score"]["revenue"]["exact"] and r["score"]["shipping"]["exact"]
    assert tiers_firing(r)["singular"] == ["assert_one_current_row_per_customer"]


def test_a_date_shift_is_caught_by_no_test_but_the_answer_key(results):
    r = results["m13_timezone_shift"]
    assert not any(t["status"] != "pass" for t in r["tests"])
    assert r["score"]["revenue"]["abs_error"] == D("1456896.16")
    assert r["score"]["revenue"]["net_error"] == 0 and r["score"]["shipping"]["net_error"] == 0
    assert R.bucket_counts(list(results.values()))["nothing"] == 1


def test_wrong_grain_keeps_every_total_and_changes_the_line_count(results):
    r = results["m10_wrong_grain"]
    assert r["score"]["revenue"]["net_error"] == 0
    assert r["score"]["revenue"]["lines_error"] < 0
    assert r["rows"]["fct_order_line"] < 29902


def test_every_gap_has_a_named_closing_test(results):
    """The matrix is only useful if each miss maps to a test that closes it."""
    typical_misses = [v for v in EXPECTED if classify(results[v]) in ("added only", "nothing")]
    closers = {v: {t["label"] for t in results[v]["tests"] if t["status"] != "pass"} for v in typical_misses}
    assert {v for v, c in closers.items() if not c} == {"m13_timezone_shift"}


def test_printed_output_is_ascii(results, capsys):
    ordered = list(results.values())
    R.print_matrix(ordered)
    R.print_naive([results[i] for i in V.NAIVE_IDS])
    out = capsys.readouterr().out
    assert out.isascii() and len(out) > 500


def test_the_artifacts_are_valid_json_and_csv(results, tmp_path):
    ordered = list(results.values())
    R.write_matrix_artifacts(ordered, tmp_path)
    R.write_naive_artifacts([results[i] for i in V.NAIVE_IDS], tmp_path)
    assert len(json.loads((tmp_path / "mutation_matrix.json").read_text())) == 14
    assert (tmp_path / "mutation_matrix.csv").read_text().count("\n") == 15
    assert (tmp_path / "naive_vs_correct.csv").read_text().isascii()


def test_a_patch_that_matches_nothing_raises():
    with pytest.raises(ValueError):
        V.patch_files({"a.sql": "select 1"}, [{"op": "replace", "path": "a.sql", "old": "zzz", "new": "y"}])
    with pytest.raises(ValueError):
        V.patch_files({"a.sql": "aa"}, [{"op": "replace", "path": "a.sql", "old": "a", "new": "b"}])


# --------------------------------------------------- notebook equals run.py
NOTEBOOK = ROOT / "notebooks" / "star_schema_modelling_standalone.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _notebook_results():
    for cell in _notebook()["cells"]:
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", ""))
            if text.startswith("RESULTS_JSON "):
                return json.loads(text[len("RESULTS_JSON "):])
    raise AssertionError("the saved notebook has no RESULTS_JSON output")


def test_the_notebook_is_executed_with_charts():
    cells = _notebook()["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    assert len(cells) == 29 and len(code) == 18
    assert all(c["execution_count"] for c in code)
    assert not any(o["output_type"] == "error" for c in code for o in c["outputs"])
    charts = sum("image/png" in o.get("data", {}) for c in code for o in c["outputs"])
    assert charts == 6
    assert all(src.isascii() for src in ("".join(c["source"]) for c in cells)), "non-ascii in notebook"


def test_the_notebook_numbers_equal_the_dbt_numbers(results):
    nb = _notebook_results()
    assert nb["tests_in_suite"] == len(results["baseline"]["tests"]) == 64
    assert nb["buckets"] == {k: v for k, v in R.bucket_counts(list(results.values())).items()
                             if k in ("generic", "singular only", "added only", "nothing")}
    assert nb["silent"] == R.bucket_counts(list(results.values()))["silent"]
    assert set(nb["variants"]) == set(results)
    for name, r in results.items():
        got = nb["variants"][name]
        assert got["first_caught_by"] == classify(r), name
        assert got["silent"] == is_silent(r), name
        assert got["failing"] == {t["label"]: t["failures"] for t in r["tests"] if t["status"] != "pass"}, name
        assert got["rows"] == r["rows"], name
        s = r["score"]
        for field, value in (("revenue_abs_error", s["revenue"]["abs_error"]),
                             ("revenue_net_error", s["revenue"]["net_error"]),
                             ("shipping_abs_error", s["shipping"]["abs_error"]),
                             ("shipping_net_error", s["shipping"]["net_error"])):
            assert D(got[field]) == value, (name, field)
        assert got["exact"] == (s["revenue"]["exact"] and s["shipping"]["exact"]), name

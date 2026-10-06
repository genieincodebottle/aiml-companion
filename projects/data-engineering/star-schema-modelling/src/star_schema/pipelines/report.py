"""Turn build results into the tables this project prints and the artefacts it writes."""
from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path

from star_schema import variants as V
from star_schema.verdict import TIERS, classify, is_silent, tiers_firing
from star_schema.utils.tables import money, render

TIER_LETTER = {"generic": "G", "singular": "S", "added": "A"}


def pct(value: Decimal) -> str:
    return f"{value:,.2f}%"


def _jsonable(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=1, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------- naive table
def naive_rows(results: list[dict]) -> list[list]:
    """One row per (naive approach, measure) that is wrong, or one zero row."""
    rows = []
    for r in results:
        for measure in ("revenue", "shipping"):
            s = r["score"][measure]
            if s["exact"] and measure == "shipping":
                continue
            region, region_err = s["worst_region"]
            worst = f"{region} {money(region_err)}" if s["abs_error"] else "-"
            rows.append([r["trap"], r["title"], measure, money(s["true_total"]),
                         money(s["abs_error"]), money(s["net_error"]), pct(s["pct_of_true"]), worst])
    return rows


NAIVE_HEADER = ["trap", "naive approach", "measure", "true total", "abs error", "net error",
                "% of true", "worst region (net)"]


def print_naive(results: list[dict]) -> None:
    print(render(naive_rows(results), NAIVE_HEADER, "lllrrrrl"))
    reloc = next((r for r in results if r["id"] == "m01_join_on_is_current"), None)
    if reloc:
        s = reloc["score"]["revenue"]
        rows = [[region, money(s["true_by_region"][region]), money(s["report_by_region"].get(region, 0)),
                 money(s["by_region"].get(region, 0))] for region in sorted(s["true_by_region"])]
        print("\nRevenue by region, joining on is_current (trap 1)")
        print(render(rows, ["region", "true", "naive", "error"]))


# --------------------------------------------------------------- matrix tables
def test_codes(results: list[dict]) -> dict[str, str]:
    """Short codes (G1, S2, A3 ...) for every test that fails in at least one variant."""
    firing = {}
    for r in results:
        for t in r["tests"]:
            if t["status"] != "pass":
                firing[t["label"]] = t["tier"]
    codes, count = {}, {"generic": 0, "singular": 0, "added": 0}
    for label, tier in sorted(firing.items(), key=lambda kv: (TIERS.index(kv[1]), kv[0])):
        count[tier] += 1
        codes[label] = f"{TIER_LETTER[tier]}{count[tier]}"
    return codes


def _code_order(code: str) -> tuple[int, int]:
    return ("GSA".index(code[0]), int(code[1:]))


def summary_rows(results: list[dict]) -> list[list]:
    rows = []
    for r in results:
        if r["id"] == "baseline":
            continue
        fired = tiers_firing(r)
        rev, ship = r["score"]["revenue"], r["score"]["shipping"]
        rows.append([r["id"][:3], r["title"], r["trap"],
                     " / ".join(str(len(fired[t])) for t in TIERS),
                     classify(r), "SILENT" if is_silent(r) else "",
                     money(rev["abs_error"]), money(ship["abs_error"]),
                     "yes" if rev["exact"] and ship["exact"] else "no"])
    return rows


SUMMARY_HEADER = ["id", "mutation", "trap", "failing G/S/A", "first caught by", "silent",
                  "revenue abs err", "shipping abs err", "answer exact"]


def grid_rows(results: list[dict], codes: dict[str, str]) -> tuple[list[list], list[str]]:
    ordered = sorted(codes, key=lambda label: _code_order(codes[label]))
    header = ["id"] + [codes[label] for label in ordered]
    rows = []
    for r in results:
        if r["id"] == "baseline":
            continue
        failed = {t["label"] for t in r["tests"] if t["status"] != "pass"}
        rows.append([r["id"][:3]] + ["X" if label in failed else "." for label in ordered])
    return rows, header


def bucket_counts(results: list[dict]) -> dict[str, int]:
    counts = {"generic": 0, "singular only": 0, "added only": 0, "nothing": 0}
    for r in results:
        if r["id"] != "baseline":
            counts[classify(r)] += 1
    counts["typical suite misses"] = counts["added only"] + counts["nothing"]
    counts["silent"] = sum(1 for r in results if r["id"] != "baseline" and is_silent(r))
    return counts


def print_matrix(results: list[dict]) -> None:
    print(render(summary_rows(results), SUMMARY_HEADER, "lllcllrrl"))
    codes = test_codes(results)
    grid, header = grid_rows(results, codes)
    print("\nWhich test fails for which mutation (X = fails)")
    print(render(grid, header, "l" + "c" * (len(header) - 1)))
    print("\nTest codes. G = generic (schema.yml), S = singular (tests/singular), "
          "A = added after the first matrix (tests/added)")
    for label in sorted(codes, key=lambda x: _code_order(codes[x])):
        print(f"  {codes[label]:<4}{label}")
    counts = bucket_counts(results)
    n = sum(counts[k] for k in ("generic", "singular only", "added only", "nothing"))
    print(f"\n{n} mutations. First caught by a generic test {counts['generic']}, "
          f"by a singular test only {counts['singular only']}, "
          f"by an added test only {counts['added only']}, by nothing {counts['nothing']}.")
    print(f"The typical suite (generic plus singular) misses {counts['typical suite misses']}, "
          f"and the answer is wrong in {counts['silent']} of those.")


def write_matrix_artifacts(results: list[dict], out_dir: Path) -> None:
    write_json(out_dir / "mutation_matrix.json", results)
    labels = sorted({t["label"] for r in results for t in r["tests"]})
    rows = []
    for r in results:
        by_label = {t["label"]: t for t in r["tests"]}
        rows.append([r["id"], classify(r), "yes" if is_silent(r) else "no"]
                    + [by_label[label]["failures"] if label in by_label else "" for label in labels])
    write_csv(out_dir / "mutation_matrix.csv", ["variant", "first_caught_by", "silent"] + labels, rows)


def write_naive_artifacts(results: list[dict], out_dir: Path) -> None:
    write_json(out_dir / "naive_vs_correct.json", [
        {k: r[k] for k in ("id", "title", "trap", "score", "rows")} for r in results])
    write_csv(out_dir / "naive_vs_correct.csv", NAIVE_HEADER, naive_rows(results))


def variant_ids(include_baseline: bool = True) -> list[str | None]:
    return ([None] if include_baseline else []) + [v["id"] for v in V.VARIANTS]

"""Builds the standalone notebook as plain text, so it diffs like code.

Everything the notebook runs is read from the repo and inlined here, so the
notebook cannot drift from the project it teaches:

  * the generator (src/star_schema/data/world.py) and the scorer, verbatim
  * every dbt model, singular test and generic test macro, as text
  * the generic tests from schema.yml, parsed into a list
  * the mutation definitions (src/star_schema/variants.py), verbatim

The notebook then resolves {{ ref() }} and {{ source() }} itself and runs the
SQL in an in-memory DuckDB. It imports nothing from src/ and writes nothing to disk.

    python notebooks/_build_notebook.py
"""
from __future__ import annotations

import pprint
import re
import sys
from pathlib import Path

import nbformat as nbf
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parent / "star_schema_modelling_standalone.ipynb"

nb = nbf.v4.new_notebook()
C: list = []


def md(t: str) -> None:
    C.append(nbf.v4.new_markdown_cell(t.strip("\n")))


# ---------------------------------------------------------------------------
# Read the repo.
# ---------------------------------------------------------------------------
def read(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


CONFIG = yaml.safe_load(read(ROOT / "conf" / "config.yaml"))
WORLD_SRC = read(ROOT / "src/star_schema/data/world.py")
SCORING_SRC = read(ROOT / "src/star_schema/evaluation/scoring.py")
VERDICT_SRC = read(ROOT / "src/star_schema/verdict.py")
VARIANTS_SRC = read(ROOT / "src/star_schema/variants.py")

WAREHOUSE = ROOT / "warehouse"


def sql_files(sub: str) -> dict:
    return {p.relative_to(WAREHOUSE).as_posix(): read(p)
            for p in sorted((WAREHOUSE / sub).rglob("*.sql"))}


STAGING = sql_files("models/staging")
MARTS = sql_files("models/marts")
SINGULAR = {p.stem: read(p) for p in sorted((WAREHOUSE / "tests/singular").glob("*.sql"))}
ADDED = {p.stem: read(p) for p in sorted((WAREHOUSE / "tests/added").glob("*.sql"))}
MACROS = {}
for p in sorted((WAREHOUSE / "tests/generic").glob("*.sql")):
    m = re.search(r"{% test (\w+)\(.*?%}(.*?){% endtest %}", read(p), re.S)
    MACROS[m.group(1)] = m.group(2).strip("\n")


def schema_tests() -> list:
    specs = []

    def add(model, column, entry):
        if isinstance(entry, str):
            name, body = entry, {}
        else:
            (name, body), = entry.items()
            body = body or {}
        tags = body.get("config", {}).get("tags", [])
        specs.append({"name": name, "model": model, "column": column,
                      "args": body.get("arguments", {}),
                      "tier": "added" if "added" in tags else "generic"})

    for path in sorted((WAREHOUSE / "models").rglob("schema.yml")):
        for model in yaml.safe_load(read(path)).get("models", []):
            for entry in model.get("data_tests", []):
                add(model["name"], None, entry)
            for column in model.get("columns", []):
                for entry in column.get("data_tests", []):
                    add(model["name"], column["name"], entry)
    return specs


SCHEMA_TESTS = schema_tests()


def literal_dict(name: str, mapping: dict) -> str:
    """A readable Python dict of triple-quoted strings, one per file."""
    parts = [f"{name} = {{"]
    for key, text in mapping.items():
        assert '"""' not in text and "\\" not in text, key
        parts.append(f'"{key}": """\\\n{text}""",')
    parts.append("}")
    return "\n".join(parts)


def strip_module_docstring(src: str) -> str:
    return re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)


# ---------------------------------------------------------------------------
# Cell headers, in cell order. Each says what the cell consumes and produces, so
# a reader can drop into the middle of the notebook and still know the state.
# ---------------------------------------------------------------------------
CELL_HEADERS = [
    ("0.1", "Imports and chart styling", "nothing",
     "palette, style() helper, usd()",
     "One palette for every chart. Nothing here touches the data."),
    ("1.1", "Generate the source system and write down the answer key", "nothing",
     "CONFIG, build_world(), world, TRUE_REVENUE, TRUE_SHIPPING, TRUE_LEDGER",
     "build_world() is the repo's generator, copied in unchanged. The TRUE_ names "
     "hold what really happened. Nothing that builds the warehouse reads them."),
    ("1.2", "Load the raw tables as text", "world",
     "load_raw(), the raw schema in DuckDB",
     "Every raw column is a string, so staging is the one place types are decided."),
    ("2.1", "The scorer and the verdict helpers", "nothing",
     "score_revenue(), score_shipping(), classify(), is_silent()",
     "Exact Decimal arithmetic. A tolerance would hide the float mutation."),
    ("3.1", "The staging models, as written in the repo", "nothing",
     "STAGING (name -> SQL with ref() and source() still in it)",
     "Staging only casts, renames and dedups."),
    ("3.2", "The mart models, as written in the repo", "nothing",
     "MARTS (name -> SQL)",
     "Two dimensions, two facts at two grains, two reports."),
    ("3.3", "The tests, as written in the repo", "nothing",
     "SINGULAR, ADDED, MACROS, SCHEMA_TESTS",
     "Generic tests come from schema.yml. Singular and added tests are plain SQL files."),
    ("4.1", "A small dbt: resolve ref() and source(), build, test, score", "the SQL above",
     "build_models(), run_tests(), run_variant()",
     "Models are built in dependency order. A test passes when its query returns no rows."),
    ("5.1", "Build the correct model and score it", "the engine",
     "baseline, the rows-per-model table",
     "The correct model must match the answer key exactly, and pass every test."),
    ("5.2", "One customer who moved", "baseline, TRUE_LEDGER",
     "the example table and chart",
     "The same orders attributed by the validity window and by is_current."),
    ("6.1", "Build the six naive versions", "variants",
     "naive results",
     "Each naive version is a plausible edit to the correct SQL."),
    ("6.2", "Naive against correct, per trap", "naive results",
     "the error table and chart",
     "Errors are measured against the answer key, to the cent."),
    ("6.3", "Revenue by region when you join on is_current", "naive results",
     "the by-region table and chart",
     "The total is exact and every region is wrong."),
    ("6.4", "Shipping on the line fact", "naive results",
     "the shipping table and chart",
     "Order-level shipping summed at line grain."),
    ("7.1", "Run the mutation matrix", "variants, the engine",
     "matrix results, the summary table",
     "13 breaks, every test, one build each."),
    ("7.2", "Which test fires for which break", "matrix results",
     "the test grid and heatmap",
     "A test that never fails for any plausible break is not protecting anything."),
    ("7.3", "Who catches what", "matrix results",
     "the bucket counts and chart",
     "First line of defence per break."),
    ("8.1", "The numbers this notebook produced", "everything above",
     "RESULTS (printed as JSON)",
     "tests/test_lessons.py compares this block to the numbers run.py writes."),
]

WIDTH = 78


def _wrap(text: str, width: int) -> list:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _banner(num: str, title: str, ins: str, outs: str, note: str = "") -> str:
    thick, thin = "# " + "=" * (WIDTH - 2), "# " + "-" * (WIDTH - 2)
    rows = [thick, f"# {num}  {title}", thin, f"# In   : {ins}", f"# Out  : {outs}"]
    if note:
        first, *rest = _wrap(note, WIDTH - 11)
        rows.append(f"# Note : {first}")
        rows += [f"#        {r}" for r in rest]
    rows.append(thick)
    return "\n".join(rows)


def code(src: str) -> None:
    """Append a code cell, prefixed with its header banner."""
    body = src.strip("\n")
    if CELL_HEADERS:
        body = _banner(*CELL_HEADERS.pop(0)) + "\n" + body
    C.append(nbf.v4.new_code_cell(body))


# ---------------------------------------------------------------------------
# The notebook.
# ---------------------------------------------------------------------------
md("""
# Star Schema Modelling

## Break a dbt star schema on purpose, and see which tests notice

**Standalone notebook.** It generates a source system, builds the warehouse with
the same SQL the repo's dbt project uses, and breaks it 13 ways. It imports
nothing from the project package and writes nothing to disk. The warehouse runs
in an in-memory DuckDB, and a small resolver stands in for dbt (it fills in
`ref()` and `source()`, builds models in dependency order, and runs each test
query).

### What you will see

1. A source system with five planted traps, and a written-down answer key.
2. The correct model, scoring exactly 0.00 against that key.
3. The naive version of each trap, with the money it gets wrong.
4. A mutation matrix. Thirteen plausible edits to the model, every test run
   against each, and a record of which test fires. Some breaks are caught by the
   generic tests, some only by a singular business rule, some only by a test added
   after the first matrix, and one by nothing.

### Before you run

Colab and Kaggle already have pandas and matplotlib. You need one install.

```bash
pip install duckdb
```

About a minute top to bottom, 6 charts. Every code cell opens with a header that
says what it consumes and what it leaves behind, so you can start anywhere.
""")

code("""
import json
import re
import warnings
from decimal import Decimal

import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

warnings.filterwarnings("ignore")
pd.set_option("display.width", 170, "display.max_columns", 30, "display.max_colwidth", 80)

# Validated categorical slots; the first three are colourblind-safe on every pair.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b8b7b2"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.8, "legend.frameon": False,
    "legend.fontsize": 9, "figure.dpi": 110, "font.size": 10,
})


def style(ax, title=None, xlabel=None, ylabel=None, grid="y"):
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    if grid:
        ax.grid(axis=grid, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
    return ax


def usd(x):
    return f"{Decimal(str(x)):,.2f}"


print("ready")
""")

md("""
## 1. The source system and its answer key

An online store exports four tables from its order system. The export has five
problems planted at known sizes, and each one is a place a star schema goes wrong.

| # | Planted problem | What a naive model does |
|---|---|---|
| 1 | Customers move region during the period | Joins the fact to the customer's *current* region, so old orders land in the wrong region |
| 2 | Shipping is charged per order | Puts it on the line-level fact and double counts it |
| 3 | An upstream retry re-sent chunks of the order lines file | Counts those lines twice |
| 4 | Some customers were synced after their first order | An inner join on the validity window silently drops those orders |
| 5 | Some orders are stamped on the exact second a region changed | An inclusive window join matches two versions and fans out |

The generator also writes the true revenue by day, region and category from its
own event history, in `TRUE_REVENUE`. The warehouse build never reads it.
""")

CONFIG_LITERAL = pprint.pformat(CONFIG, sort_dicts=False, width=96)
code(f"""
CONFIG = {CONFIG_LITERAL}

{strip_module_docstring(WORLD_SRC).strip()}

world = build_world(CONFIG)
TRUE_REVENUE, TRUE_SHIPPING, TRUE_LEDGER = world["TRUE_REVENUE"], world["TRUE_SHIPPING"], world["TRUE_LEDGER"]

print("raw rows:", {{t: len(rows) for t, (_, rows) in world["raw"].items()}})
planted = {{k: TRUE_LEDGER[k] for k in (
    "customers_relocating_once", "customers_relocating_twice", "relocated_orders",
    "resent_rows", "late_customers", "late_orders", "boundary_orders")}}
print("planted:", planted)
""")

code("""
def load_raw(con):
    \"\"\"Every raw column is a string, exactly as the CSV export would hold it.\"\"\"
    con.execute("create schema if not exists raw")
    for table, (header, rows) in world["raw"].items():
        frame = pd.DataFrame(rows, columns=header, dtype="string")
        con.register("raw_frame", frame)
        con.execute(f"create or replace table raw.{table} as select * from raw_frame")
        con.unregister("raw_frame")


print("loader ready")
""")

md("""
## 2. The scorer

`score_revenue` and `score_shipping` compare a built report to the answer key in
exact Decimal arithmetic. There is no tolerance, because a tolerance would hide
the float mutation later. `classify` and `is_silent` read a build result and say
which tier of test noticed a break.
""")

code(f"""
{strip_module_docstring(SCORING_SRC).strip()}


{strip_module_docstring(VERDICT_SRC).strip()}
""")

md("""
## 3. The dbt project, as text

These are the repo's model and test files, read at build time and pasted in. The
`{{ ref() }}` and `{{ source() }}` calls are still in them. Staging only casts,
renames and dedups. `dim_customer` is Type 2 with an exclusive `valid_to`. There
are two facts at two grains, and shipping lives only on the order grain fact.
""")

code(literal_dict("STAGING", STAGING))
code(literal_dict("MARTS", MARTS))

tests_cell = "\n\n".join([
    literal_dict("SINGULAR", SINGULAR), literal_dict("ADDED", ADDED),
    literal_dict("MACROS", MACROS),
    "SCHEMA_TESTS = " + pprint.pformat(SCHEMA_TESTS, sort_dicts=False, width=96)])
code(tests_cell)

md("""
## 4. A small stand-in for dbt

This is not a dbt reimplementation. It does the four things this project needs.
It fills in `ref()` and `source()`, builds the models in dependency order, runs
each test as a query that must return no rows, and records how many rows came
back. The repo runs the real dbt. Section 8 prints the numbers this notebook
produces, and a test in the repo compares them with the numbers from real dbt,
test by test.
""")

code(r'''
def render(sql):
    """Resolve the three jinja calls the models may use. Anything else is a bug."""
    sql = re.sub(r"\{\{\s*config\([^)]*\)\s*\}\}\n?", "", sql)
    sql = re.sub(r"\{\{\s*ref\('([^']+)'\)\s*\}\}", r"\1", sql)
    sql = re.sub(r"\{\{\s*source\('([^']+)',\s*'([^']+)'\)\s*\}\}", r"\1.\2", sql)
    assert "{{" not in sql and "{%" not in sql, sql
    return sql


def name_of(path):
    return path.rsplit("/", 1)[-1][: -len(".sql")]


def build_models(con, files):
    """Create every model, upstream first. Staging is a view and marts are tables,
    as dbt_project.yml sets them."""
    models = {name_of(p): (s, "view" if "/staging/" in p else "table") for p, s in files.items()}
    deps = {n: set(re.findall(r"ref\('([^']+)'\)", s)) for n, (s, _) in models.items()}
    done = []
    while len(done) < len(models):
        ready = [n for n in sorted(models) if n not in done and deps[n] <= set(done)]
        assert ready, "circular ref"
        for n in ready:
            sql, kind = models[n]
            con.execute(f"create or replace {kind} {n} as {render(sql)}")
            done.append(n)


def count_rows(con, sql):
    return con.execute(f"select count(*) from ({sql}" + chr(10) + ")").fetchone()[0]


def builtin_sql(t):
    m, c, a = t["model"], t["column"], t["args"]
    if t["name"] == "not_null":
        return f"select * from {m} where {c} is null"
    if t["name"] == "unique":
        return f"select {c} from {m} where {c} is not null group by {c} having count(*) > 1"
    if t["name"] == "accepted_values":
        shown = ", ".join(str(v).lower() if isinstance(v, bool) else f"'{v}'" for v in a["values"])
        return f"select {c} from {m} where {c} is not null and {c} not in ({shown})"
    if t["name"] == "relationships":
        to = re.search(r"ref\('([^']+)'\)", a["to"]).group(1)
        return (f"select child.{c} from {m} as child left join {to} as parent "
                f"on child.{c} = parent.{a['field']} "
                f"where child.{c} is not null and parent.{a['field']} is null")
    return None


def macro_sql(t):
    sql = MACROS[t["name"]].replace("{{ model }}", t["model"])
    sql = sql.replace("{{ column_name }}", t["column"] or "")
    cols = ", ".join(t["args"].get("combination_of_columns", []))
    return sql.replace("{{ combination_of_columns | join(', ') }}", cols)


def test_label(t):
    target = f"{t['model']}.{t['column']}" if t["column"] else t["model"]
    extra = ""
    if "combination_of_columns" in t["args"]:
        extra = "(" + ",".join(t["args"]["combination_of_columns"]) + ")"
    elif "to" in t["args"]:
        extra = "->" + re.search(r"ref\('([^']+)'\)", t["args"]["to"]).group(1)
    return f"{t['name']}[{target}]{extra}"


def run_tests(con):
    """Every test, tagged with its tier. A test fails when its query returns rows."""
    out = []
    for t in SCHEMA_TESTS:
        sql = builtin_sql(t) or macro_sql(t)
        n = count_rows(con, sql)
        out.append({"label": test_label(t), "tier": t["tier"],
                    "status": "pass" if n == 0 else "fail", "failures": n})
    for name, text in {**SINGULAR, **ADDED}.items():
        tier = "added" if "tags=['added']" in text else "singular"
        n = count_rows(con, render(text))
        out.append({"label": name, "tier": tier,
                    "status": "pass" if n == 0 else "fail", "failures": n})
    return sorted(out, key=lambda t: (t["tier"], t["label"]))


def score_db(con):
    revenue = con.execute(
        "select cast(order_date as varchar), region, category, units, order_lines, "
        "cast(line_revenue as varchar) from rpt_revenue_by_region_category_day").fetchall()
    shipping = con.execute(
        "select cast(order_date as varchar), region, orders, "
        "cast(shipping_revenue as varchar) from rpt_shipping_by_region_day").fetchall()
    return {"revenue": score_revenue(revenue, TRUE_REVENUE),
            "shipping": score_shipping(shipping, TRUE_SHIPPING)}


LAYERS = ["stg_customer_changes", "stg_orders", "stg_order_lines", "stg_products",
          "dim_customer", "dim_product", "fct_order", "fct_order_line",
          "rpt_revenue_by_region_category_day", "rpt_shipping_by_region_day"]


def run_variant(variant=None):
    """Build the correct model (variant None) or a broken copy, test it, score it."""
    files = {**STAGING, **MARTS}
    if variant:
        files = patch_files(files, variant["ops"])
    con = duckdb.connect(config={"threads": 1})
    load_raw(con)
    build_models(con, files)
    result = {"id": variant["id"] if variant else "baseline",
              "title": variant["title"] if variant else "Shipped model, unmodified",
              "trap": variant["trap"] if variant else "-", "run_ok": True,
              "tests": run_tests(con), "score": score_db(con),
              "rows": {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in LAYERS}}
    con.close()
    return result


print("engine ready")
''')

md("""
## 5. The correct model

It is built from the repo's SQL, tested, and scored against the answer key. The
score has to be exactly zero, because the generator and the model share no code
and no data except the raw tables.
""")

code("""
baseline = run_variant(None)
failed = [t["label"] for t in baseline["tests"] if t["status"] != "pass"]
print(f"tests passed: {len(baseline['tests']) - len(failed)} of {len(baseline['tests'])}", failed)
for name in ("revenue", "shipping"):
    s = baseline["score"][name]
    print(f"{name:<9} true {usd(s['true_total']):>13}  model {usd(s['report_total']):>13}  "
          f"abs error {usd(s['abs_error'])}  exact {s['exact']}")
display(pd.DataFrame({"rows": baseline["rows"]}))
""")

md("""
### One customer who moved

The generator guarantees a customer who relocates, with orders on both sides.
The correct model attributes each order to the region she was in when she placed
it. Joining on `is_current` attributes all of them to where she lives now.
""")

code("""
sc = CONFIG["showcase"]
con = duckdb.connect(config={"threads": 1})
load_raw(con)
build_models(con, {**STAGING, **MARTS})
example = con.execute(f'''
    select o.order_id, cast(o.order_ts as varchar) as order_ts,
           cast(sum(l.line_revenue) as decimal(12, 2)) as amount,
           d.region as model_region, cur.region as is_current_region
    from fct_order o
    join dim_customer d on d.customer_sk = o.customer_sk
    join dim_customer cur on cur.customer_id = d.customer_id and cur.is_current
    join fct_order_line l on l.order_id = o.order_id
    where d.customer_id = {sc["customer_id"]}
    group by 1, 2, 4, 5 order by 1''').df()
con.close()
truth = {o["order_id"]: o["true_region"] for o in TRUE_LEDGER["showcase_orders"]}
example["true_region"] = example["order_id"].map(truth)
example["order_ts"] = pd.to_datetime(example["order_ts"])
example["model_ok"] = example["model_region"] == example["true_region"]
example["is_current_ok"] = example["is_current_region"] == example["true_region"]
print(f"{sc['name']} moved {sc['region_from']} -> {sc['region_to']} at {sc['moved_at']}")
display(example[["order_id", "order_ts", "amount", "true_region", "model_region",
                 "is_current_region", "model_ok", "is_current_ok"]])

fig, ax = plt.subplots(figsize=(9, 3.2))
region_colour = {sc["region_from"]: ORANGE, sc["region_to"]: BLUE}
for y, column in ((1, "model_region"), (0, "is_current_region")):
    ax.scatter(example["order_ts"], [y] * len(example), s=example["amount"].astype(float) * 0.5,
               c=[region_colour[r] for r in example[column]], zorder=3)
ax.axvline(pd.Timestamp(sc["moved_at"]), color=GREY, lw=1.2, ls="--", zorder=2)
ax.set_yticks([0, 1], ["join on is_current", "validity window"])
ax.set_ylim(-0.6, 1.6)
style(ax, f"{sc['name']}'s orders, by the region each model assigns", grid="x")
ax.scatter([], [], c=ORANGE, label=sc["region_from"])
ax.scatter([], [], c=BLUE, label=sc["region_to"])
ax.legend(loc="upper left", ncol=2, title="region", bbox_to_anchor=(0, 1.02))
plt.tight_layout()
plt.show()
""")

md("""
## 6. The naive version of each trap

Each naive model is the correct SQL with one plausible edit. The edits are listed
in `VARIANTS` (this is the repo's `variants.py`, unchanged), and each one is
applied to a copy of the model text, never to the original.
""")

code(f"""
{strip_module_docstring(VARIANTS_SRC).strip()}

naive = [run_variant(by_id(i)) for i in NAIVE_IDS]
print("built", len(naive), "naive models")
""")

code("""
rows = []
for r in naive:
    for measure in ("revenue", "shipping"):
        s = r["score"][measure]
        if s["exact"] and measure == "shipping":
            continue
        rows.append({"trap": r["trap"], "naive approach": r["title"], "measure": measure,
                     "true total": usd(s["true_total"]), "abs error": usd(s["abs_error"]),
                     "net error": usd(s["net_error"]), "% of true": f"{s['pct_of_true']:.2f}%"})
display(pd.DataFrame(rows))

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
names = [r["id"][:3] + " " + r["trap"] for r in naive]
for ax, measure, colour in ((a, "revenue", BLUE), (b, "shipping", ORANGE)):
    values = [float(r["score"][measure]["pct_of_true"]) for r in naive]
    ax.barh(names, values, color=colour, zorder=3)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f"{v:.2f}%", va="center", fontsize=9, color=INK2)
    style(ax, f"{measure.capitalize()} error, % of true total", grid="x")
a.invert_yaxis()
plt.tight_layout()
plt.show()
""")

code("""
m01 = next(r for r in naive if r["id"] == "m01_join_on_is_current")["score"]["revenue"]
regions = sorted(m01["true_by_region"])
table = pd.DataFrame({"true": [m01["true_by_region"][g] for g in regions],
                      "is_current join": [m01["report_by_region"][g] for g in regions],
                      "error": [m01["by_region"][g] for g in regions]}, index=regions)
display(table.map(usd))
print("revenue in the wrong region:", usd(Decimal(TRUE_LEDGER["relocated_revenue_cents"]).scaleb(-2)))

fig, ax = plt.subplots(figsize=(8, 3.4))
errors = [float(m01["by_region"][g]) for g in regions]
ax.bar(regions, errors, color=[ORANGE if e < 0 else BLUE for e in errors], zorder=3)
ax.axhline(0, color=INK2, lw=0.8)
style(ax, "Joining on is_current: net revenue error by region (total error is 0.00)")
plt.tight_layout()
plt.show()
""")

code("""
m03 = next(r for r in naive if r["id"] == "m03_shipping_on_line_fact")["score"]["shipping"]
print("true shipping", usd(m03["true_total"]), " summed on the line fact",
      usd(m03["report_total"]), " overstated by", usd(m03["net_error"]),
      f"({m03['pct_of_true']:.2f}%)")

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.bar(["true", "line fact"], [float(m03["true_total"]), float(m03["report_total"])],
       color=[BLUE, ORANGE], zorder=3)
style(ax, "Total shipping revenue, true and summed on the line fact")
plt.tight_layout()
plt.show()
""")

md("""
## 7. The mutation matrix

Now break the correct model thirteen ways, five of them the traps above and
eight more that a reviewer would also call plausible. For each break, run every
test and record which ones fail. A test that only passes proves nothing until you
know a break that makes it fail.

Tests come in three tiers.

* **Generic** tests are the `schema.yml` tests. Uniqueness, not null, accepted values, relationships, and a grain test.
* **Singular** tests are the four business rules from the brief. One current row per customer, facts inside their customer window, line revenue reconciling to staging, shipping reconciling to staging.
* **Added** tests were written after the first matrix showed breaks that nothing caught. Each one closes a named gap.
""")

code("""
matrix = [baseline] + [run_variant(v) for v in VARIANTS]
rows = []
for r in matrix[1:]:
    fired = tiers_firing(r)
    rows.append({"id": r["id"][:3], "mutation": r["title"],
                 "G/S/A failing": " / ".join(str(len(fired[t])) for t in TIERS),
                 "first caught by": classify(r), "silent": "SILENT" if is_silent(r) else "",
                 "revenue abs err": usd(r["score"]["revenue"]["abs_error"]),
                 "shipping abs err": usd(r["score"]["shipping"]["abs_error"]),
                 "answer exact": r["score"]["revenue"]["exact"] and r["score"]["shipping"]["exact"]})
display(pd.DataFrame(rows))
""")

code("""
firing = {}
for r in matrix[1:]:
    for t in r["tests"]:
        if t["status"] != "pass":
            firing[t["label"]] = t["tier"]
order = sorted(firing, key=lambda lab: (TIERS.index(firing[lab]), lab))
codes = {lab: f"{firing[lab][0].upper()}{sum(1 for o in order[:order.index(lab) + 1] if firing[o] == firing[lab])}"
         for lab in order}
grid = [[int(any(t["label"] == lab and t["status"] != "pass" for t in r["tests"])) for lab in order]
        for r in matrix[1:]]
for lab in order:
    print(f"{codes[lab]:<4}{lab}")

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.imshow(grid, cmap=mpl.colors.ListedColormap([SURFACE, ORANGE]), aspect="auto")
ax.set_xticks(range(len(order)), [codes[lab] for lab in order])
ax.set_yticks(range(len(grid)), [r["id"][:3] + " " + r["trap"] for r in matrix[1:]])
ax.set_xticks([x - 0.5 for x in range(len(order) + 1)], minor=True)
ax.set_yticks([y - 0.5 for y in range(len(grid) + 1)], minor=True)
ax.grid(which="minor", color=MUTED, lw=0.5)
ax.tick_params(which="both", length=0)
ax.set_title("Tests that fail (orange) for each mutation. G generic, S singular, A added")
for s in ax.spines.values():
    s.set_visible(False)
plt.tight_layout()
plt.show()
""")

code("""
buckets = {k: 0 for k in ("generic", "singular only", "added only", "nothing")}
for r in matrix[1:]:
    buckets[classify(r)] += 1
silent = sum(1 for r in matrix[1:] if is_silent(r))
print(buckets, "| silent (typical suite passes, answer wrong):", silent)

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(list(buckets), list(buckets.values()), color=[BLUE, AQUA, ORANGE, GREY], zorder=3)
for i, v in enumerate(buckets.values()):
    ax.text(i, v + 0.1, str(v), ha="center", fontsize=10)
style(ax, "First line of defence that notices each of the 13 breaks")
plt.tight_layout()
plt.show()
""")

md("""
## 8. The numbers this notebook produced

The next cell prints one JSON block. `tests/test_lessons.py` in the repo parses it
from the saved notebook and compares it with the numbers `python run.py break`
writes, using real dbt. If they ever differ, the suite fails.
""")

code("""
def brief(r):
    s = r["score"]
    return {"first_caught_by": classify(r), "silent": is_silent(r),
            "failing": {t["label"]: t["failures"] for t in r["tests"] if t["status"] != "pass"},
            "revenue_abs_error": str(s["revenue"]["abs_error"]), "revenue_net_error": str(s["revenue"]["net_error"]),
            "shipping_abs_error": str(s["shipping"]["abs_error"]), "shipping_net_error": str(s["shipping"]["net_error"]),
            "exact": s["revenue"]["exact"] and s["shipping"]["exact"],
            "rows": r["rows"]}


RESULTS = {"tests_in_suite": len(baseline["tests"]), "buckets": buckets, "silent": silent,
           "variants": {r["id"]: brief(r) for r in matrix}}
print("RESULTS_JSON " + json.dumps(RESULTS, sort_keys=True))
""")

md("""
## 9. What to take away

* **Grain first.** `fct_order_line` has one row per `(order_id, line_no)` and
  `fct_order` has one row per order. The uniqueness tests follow from those two
  sentences, and so does the decision to keep shipping on the order fact.
* **The region question forces the Type 2 dimension.** "Revenue by the region the
  customer was in at the time" cannot be answered from a dimension that only
  knows where the customer is now.
* **Judge a test by what breaks it.** Most of the tests never fire for any of the
  13 breaks. That does not make them wrong, but this matrix gives no evidence
  that they protect anything.
* **Reconciling to a wrong staging model passes.** Dropping the dedup inflates
  staging and the fact together, so the fact still equals staging. The test that
  catches it goes back to the raw table.
* **One break is caught by nothing.** The 6 hour date shift keeps every total,
  count and window intact. Only a comparison with an independent source of truth,
  which is what the answer key is here, shows it.

### What this does not prove

The data is generated, and each trap is planted at a rate chosen to be visible.
Real exports break in ways nobody planted. Thirteen mutations are a sample, not a
census, and the added tests were written after the first matrix, so a clean
second matrix shows they close the gaps found, not that no gaps remain.
""")

assert not CELL_HEADERS, (
    f"{len(CELL_HEADERS)} cell header(s) unused: a code cell was removed or "
    "reordered without updating CELL_HEADERS")

nb["cells"] = C
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
})
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    nbf.write(nb, fh)
print(f"wrote {OUT} -- {len(C)} cells "
      f"({sum(c['cell_type'] == 'code' for c in C)} code, "
      f"{sum(c['cell_type'] == 'markdown' for c in C)} markdown)")

"""Builds the standalone notebook as plain text, so it diffs like code.

The notebook may not import from `src/` and may not write to disk. So this
script copies the package source into code cells, with three rules.

* `from warehouse...` imports are dropped, and `module.name` calls become
  `name`, because every cell shares one namespace.
* The notebook builds every scenario with `MemoryStore`, so the disk-backed
  `ParquetStore` is copied but never used.
* The SQL models are read from `sql/` here and embedded as text, so the
  notebook runs the same SQL as the repo and cannot drift from it.

Run it with
    python notebooks/_build_notebook.py
then execute the result (see the README).
"""
import ast
import re
import sys
from pathlib import Path

import jinja2
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "warehouse"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parent / "incremental_warehouse_standalone.ipynb"

nb = nbf.v4.new_notebook()
C = []


def md(t):
    C.append(nbf.v4.new_markdown_cell(t.strip("\n")))


# ---------------------------------------------------------------------------
# Cell headers, in cell order. Each says what the cell consumes and produces,
# so a reader can drop into the middle and still know the state. They are
# consumed as the script runs, so a cell added without a header trips the
# assertion at the bottom instead of shipping unlabelled.
# ---------------------------------------------------------------------------
CELL_HEADERS = [
    ("0.1", "Install", "nothing", "duckdb and pyarrow if the runtime lacks them",
     "Colab and Kaggle already have pandas, matplotlib, Jinja2 and PyYAML."),
    ("0.2", "Imports and chart style", "nothing", "plt, style(), palette",
     "One palette for every chart. The first three slots are colourblind safe."),
    ("0.3", "Configuration", "nothing", "CFG",
     "The same conf/config.yaml the repo uses, embedded as text. Every size and rate lives here."),
    ("1.1", "Pipeline switches and the injectable clock", "CFG", "PipelineOptions, clocks",
     "Each naive variant is the correct pipeline with one switch changed."),
    ("1.2", "The shop world and its answer key", "CFG", "generate_world(), AnswerKey",
     "Only the scorer in section 3 may read the answer key. Nothing in the pipeline does."),
    ("1.3", "The OLTP source", "world batches", "Shop (SQLite)",
     "The pipeline sees this as a database connection and nothing else."),
    ("2.1", "The DAG runner", "nothing", "DAG, Task, TransientError",
     "Tasks, >>, a logical run_date, retries with backoff, skip on upstream failure."),
    ("2.2", "Raw storage", "nothing", "MemoryStore, ParquetStore",
     "One partition per table per extract date. This notebook uses the in-memory store."),
    ("2.3", "Incremental extraction", "Shop connection", "extract_orders(), extract_customers()",
     "Window on updated_at with a lookback. SELECT * so a renamed column is landed, not hidden."),
    ("2.4", "Data checks", "landed partitions, built tables", "check_raw(), check_marts()",
     "Pipeline stages. In gate mode a failure stops the run."),
    ("2.5", "SQL models and the renderer", "sql/*.sql files", "SQL_FILES, render(), build_table()",
     "The SQL text is read from the repo's sql/ folder when this notebook is built."),
    ("2.6", "Partition loading", "built models", "plan_touched_dates(), load_partitions()",
     "Overwrite replaces the order dates a run touched. Append is the naive variant."),
    ("2.7", "The warehouse", "nothing", "Warehouse (DuckDB build, published, ops)",
     "Only publish() writes the published schema, in one transaction."),
    ("2.8", "The DAG for one logical date", "everything above", "build_dag(), Env",
     "Extract, land, check, build, check, publish."),
    ("2.9", "Scenarios", "DAG, source, store, warehouse", "Scenario.run_day()",
     "One sandbox per run, so every comparison starts from nothing."),
    ("3.1", "Scoring against the answer key", "published tables, answer key", "scoring functions",
     "The only code that compares the warehouse with what really happened."),
    ("3.2", "The naive variants", "Scenario, scoring", "naive.run_all()",
     "Each variant flips one switch and is measured against the same shop."),
    ("3.3", "The three properties", "Scenario", "rerun(), backfill(), break_demo()",
     "Each returns what it measured. None needs the answer key."),
    ("4.1", "Generate the shop and show the planted failures", "CFG", "days, ANSWER_KEY, chart",
     "30 days of OLTP activity with late commits and hard deletes at counted sizes."),
    ("4.2", "Run the correct pipeline for 30 days", "days, CFG", "correct (finished Scenario)",
     "One DAG run per logical date, with one transient failure on the retry demo date."),
    ("4.3", "Score every published day", "correct, ANSWER_KEY", "per-day comparison table",
     "Published revenue by order date and region against the truth, after each run."),
    ("4.4", "The retried task in the runs table", "retry_run", "ops.runs for that run",
     "extract_orders failed once, was retried, and downstream tasks never noticed."),
    ("4.5", "Property 1, rerun", "days, CFG", "rerun result",
     "Run one date twice. Every table checksum and the raw partition digest stay the same."),
    ("4.6", "Property 2, backfill", "days, CFG", "backfill result",
     "Run past dates late, with the wall clock far in the future."),
    ("4.7", "Property 3, a break stops the run before publish", "days, CFG", "break results",
     "Rename, nulls and volume each fail at check_raw, skip publish, leave published untouched."),
    ("5.1", "Measure every naive variant", "days, ANSWER_KEY, correct", "RESULTS",
     "About two minutes. Six comparisons, each against the same shop."),
    ("5.2", "Late commits, strict watermark against lookback", "RESULTS", "table and chart",
     "Rows the strict extract misses, and how wrong the numbers are on each day."),
    ("5.3", "Hard deletes, with and without key reconciliation", "RESULTS", "table and chart",
     "A watermark cannot see a deletion. The published revenue drifts upward."),
    ("5.4", "Append against partition overwrite", "RESULTS", "table and chart",
     "Run one date twice. Append double counts it."),
    ("5.5", "Wall clock against the logical run_date", "RESULTS", "table and chart",
     "A task that files data under today's date loses the days it backfills."),
    ("5.6", "Checks as a report against checks as gates", "RESULTS", "table and chart",
     "The report publishes the bad day. The gate leaves the last good tables in place."),
    ("5.7", "Region at order time against current region", "RESULTS", "table and chart",
     "SCD type 1 moves revenue between regions without changing the total."),
    ("6.1", "Every headline number as one JSON line", "RESULTS, correct", "NOTEBOOK_RESULTS",
     "The repo's test suite compares this line with what run.py produces."),
]

WIDTH = 78


def _wrap(text, width):
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


def _banner(num, title, ins, outs, note=""):
    thick, thin = "# " + "=" * (WIDTH - 2), "# " + "-" * (WIDTH - 2)
    rows = [thick, f"# {num}  {title}", thin, f"# In   : {ins}", f"# Out  : {outs}"]
    if note:
        first, *rest = _wrap(note, WIDTH - 11)
        rows.append(f"# Note : {first}")
        rows += [f"#        {r}" for r in rest]
    rows.append(thick)
    return "\n".join(rows)


def code(src):
    body = src.strip("\n")
    if CELL_HEADERS:
        body = _banner(*CELL_HEADERS.pop(0)) + "\n" + body
    C.append(nbf.v4.new_code_cell(body))


# ---------------------------------------------------------------------------
# Inlining the package
# ---------------------------------------------------------------------------
MODULE_PREFIXES = ("checks", "extract", "load", "transform", "scoring", "land",
                   "options", "clock", "pipeline", "harness")
SEEN = {}


def _top_names(tree):
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
    return names


def inline(relpath, drop=(), drop_lines=()):
    """Return a module's source ready to paste into a shared namespace."""
    text = (PKG / relpath).read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    remove = set()
    for node in tree.body:
        if getattr(node, "name", None) in drop:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            remove.update(range(first - 1, node.end_lineno))
    kept = []
    for i, line in enumerate(lines):
        if i in remove or re.match(r"^(from|import) warehouse", line) or line.strip() in drop_lines:
            continue
        kept.append(line)
    out = "\n".join(kept)
    out = re.sub(r"(?<![\w.])(" + "|".join(MODULE_PREFIXES) + r")\.(?=[A-Za-z_])", "", out)
    out = re.sub(r"\n{4,}", "\n\n\n", out).strip("\n")
    for name in _top_names(ast.parse(out)):
        assert name not in SEEN, f"{relpath} redefines {name} from {SEEN[name]}"
        SEEN[name] = relpath
    return out


def sql_resolved(name):
    """The model as the notebook runs it, with ref() and flags resolved for reading."""
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True)
    text = (ROOT / "sql" / f"{name}.sql").read_text(encoding="utf-8")
    return env.from_string(text).render(
        ref=lambda m: m, revenue_statuses="'paid', 'shipped', 'delivered'",
        detect_deletes=True, scd_type=2).strip()


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------
md("""
# Incremental Warehouse

## "Run it twice and nothing changes. Break the input and nothing publishes."

**Standalone notebook.** It generates its own shop, defines every function it
uses and writes nothing to disk. The warehouse is DuckDB in memory and the raw
lake is a dictionary of Arrow tables. The repo keeps the same partitions as
Parquet files. Nothing here imports from the project package.

### What you build

An incremental pipeline over an OLTP database that keeps changing. It extracts
only what changed, lands it as raw partitions, builds a type 2 dimension and
partitioned facts in DuckDB, runs checks as gating stages, and publishes all or
nothing. A small DAG runner in the notebook plays the part of Airflow.

### Why the source is generated

The shop is simulated for 30 days, so the notebook knows the true state of every
order on every day. Two failures are planted at counted sizes. **Late commits**
are changes stamped before midnight that only become visible the next day.
**Hard deletes** are orders that vanish from the source. Because the truth is
written down separately, each naive pipeline gets a measured error, not an
opinion. Only the scorer reads the truth.
""")

code("""
%pip install -q duckdb pyarrow
""")

code('''
import json
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import yaml

pd.set_option("display.width", 170, "display.max_columns", 30, "display.max_colwidth", 80)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b8b7b2"
mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "axes.labelsize": 10, "grid.color": "#e8e7e3",
    "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2, "figure.dpi": 110, "font.size": 10,
})


def style(ax, title, sub=None, xlabel="", ylabel="", money=True):
    ax.set_title(title, pad=22 if sub else 12)
    if sub:
        ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    if money:
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    return ax


def usd(cents):
    sign = "-" if cents < 0 else ""
    cents = abs(int(cents))
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


print("ready")
''')

code("CONFIG_YAML = r'''\n" + (ROOT / "conf" / "config.yaml").read_text(encoding="utf-8").rstrip()
     + "\n'''\nCFG = yaml.safe_load(CONFIG_YAML)\nprint(f\"{CFG['source']['days']} days, seed {CFG['seed']}\")")

md("""
---
## 1. The source

`options.py` holds the switches that separate the correct pipeline from each
naive one. `world.py` generates the shop and the answer key. `shop.py` is the
SQLite database the pipeline reads.
""")

code(inline("options.py") + "\n\n\n" + inline("clock.py"))
code(inline("source/world.py"))
code(inline("source/shop.py"))

md("""
---
## 2. The pipeline

The runner, the storage, the extract, the checks, the SQL models, the partition
loader and the warehouse. Section 2.8 wires them into one DAG per logical date.
""")

code(inline("dag.py"))
code(inline("land.py"))
code(inline("extract.py"))
code(inline("checks.py"))

sql_names = ["stg_orders", "stg_order_lines", "stg_customer_versions", "dim_customer_scd2",
             "fct_order", "fct_order_line"]
sql_cell = "SQL_FILES = {}\n"
for n in sql_names:
    sql_cell += (f"SQL_FILES[{n!r}] = r'''\n"
                 + (ROOT / "sql" / f"{n}.sql").read_text(encoding="utf-8").rstrip()
                 + "\n'''\n")
sql_cell += ("\n\ndef read_model(name):\n    return SQL_FILES[name]\n\n\n"
             + inline("transform.py", drop=("read_model",),
                      drop_lines=("from pathlib import Path",)))
code(sql_cell)

md("The six SQL models, as the notebook runs them. `{{ ref('x') }}` is shown as `x`.\n\n"
   + "\n\n".join(f"**{n}**\n\n```sql\n{sql_resolved(n)}\n```" for n in sql_names))

code(inline("load.py"))
code(inline("warehouse.py"))
code(inline("pipeline.py"))
code(inline("harness.py"))

md("""
---
## 3. Scoring, the naive variants and the three properties

`scoring.py` is the only code that reads the answer key.
""")

code(inline("scoring.py"))
code(inline("naive.py"))
code(inline("proofs.py"))

md("""
---
## 4. The correct pipeline

### 4.1 The shop
""")

code('''
days, ANSWER_KEY = generate_world(CFG)
planted = planted_summary(ANSWER_KEY)
shop = Shop(":memory:", days)
shop.advance_to(CFG["source"]["days"])
print("rows at the end of day 30:", shop.counts())
print("planted:", planted)

day_n = [b["day"] for b in days]
by_op = lambda op: [sum(1 for e in b["events"] if e["op"] == op) for b in days]
late = [sum(1 for e in b["events"] if e["late"]) for b in days]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].bar(day_n, by_op("insert_order"), color=BLUE, label="new orders", zorder=3)
ax[0].bar(day_n, by_op("update_order"), bottom=by_op("insert_order"), color=AQUA,
          label="status updates", zorder=3)
style(ax[0], "Source activity per day", "orders table, by kind of change", "day", "changes",
      money=False)
ax[0].set_ylim(0, 330)
ax[0].legend(loc="upper left", ncol=2)
ax[1].bar(day_n, late, color=ORANGE, label=f"late commits ({planted['late_commits']} total)", zorder=3)
ax[1].bar(day_n, by_op("delete_order"), bottom=late, color=INK2,
          label=f"hard deletes ({planted['hard_deletes']} total)", zorder=3)
style(ax[1], "Planted failures per day", "stamped before midnight, or removed", "day", "rows",
      money=False)
ax[1].set_ylim(0, 14)
ax[1].legend(loc="upper left")
plt.tight_layout()
plt.show()
''')

code('''
OPTS = PipelineOptions.from_cfg(CFG)
demo = CFG["demo"]
correct = Scenario(CFG, days, OPTS)
retry_run = None
for day in range(1, CFG["source"]["days"] + 1):
    iso = correct.date_of(day).isoformat()
    faults = ({demo["transient_task"]: demo["transient_failures"]}
              if iso == demo["transient_date"] else None)
    run = correct.run_day(day, faults=faults)
    assert run.ok, f"day {day} failed"
    if faults:
        retry_run = run
print("30 daily runs ok")
print("published revenue", usd(correct.wh.published_revenue()))
print(pd.DataFrame(correct.wh.checksums().items(), columns=["table", "sha256"]).assign(
    sha256=lambda d: d.sha256.str[:12]).to_string(index=False))
''')

code('''
rows = []
for day, published in sorted(correct.published_by_day.items()):
    s = score_revenue(ANSWER_KEY, published, day)
    rows.append({"day": day, "true revenue": usd(s["truth_cents"]),
                 "published": usd(s["published_cents"]), "cells wrong": s["cells_wrong"]})
scores = pd.DataFrame(rows)
print(scores.tail(6).to_string(index=False))
print("days that match the truth exactly:", int((scores["cells wrong"] == 0).sum()), "of", len(scores))
print(score_orders(ANSWER_KEY, correct.wh, 30))
print(score_dimension(ANSWER_KEY, correct.wh, 30))
''')

code('''
print(retry_run.run_id)
print(pd.DataFrame(correct.wh.runs(retry_run.run_id),
                   columns=["run_id", "seq", "task", "status", "attempts", "error"])
      .drop(columns="run_id").to_string(index=False))
''')

md("""
### 4.2 The three properties

Each is a function that runs the pipeline and returns what it measured.
""")

code('''
rerun_result = rerun(CFG, days, OPTS, None, demo["rerun_date"])
print("run ids:", rerun_result["run_ids"])
print(pd.DataFrame({
    "first run": {**{"fct_order rows": rerun_result["first"]["fct_rows"],
                     "raw partition": rerun_result["first"]["raw_orders_digest"][:12]},
                  **{k: v[:12] for k, v in rerun_result["first"]["checksums"].items()}},
    "second run": {**{"fct_order rows": rerun_result["second"]["fct_rows"],
                      "raw partition": rerun_result["second"]["raw_orders_digest"][:12]},
                   **{k: v[:12] for k, v in rerun_result["second"]["checksums"].items()}},
}).to_string())
print("identical:", rerun_result["identical"])
''')

code('''
backfill_result = backfill(CFG, days, OPTS, None, demo["backfill_start"], demo["backfill_end"])
print(pd.DataFrame(backfill_result["partitions"]).to_string(index=False))
print("partitions in the lake:", len(backfill_result["partition_dates"]),
      "| dated by the wall clock:", backfill_result["stray"])
print("final tables equal a clean single pass:", backfill_result["equals_clean_pass"])
''')

code('''
break_results = {kind: break_demo(CFG, days, OPTS, None, kind, demo["break_date"])
                 for kind in ("rename", "nulls", "volume")}
print(pd.DataFrame([{
    "break": kind, "failed task": r["failed_task"],
    "failed check": r["failed_checks"][0][1], "tasks skipped": len(r["skipped"]),
    "published before": usd(r["published_revenue_before"]),
    "published after": usd(r["published_revenue_after"]),
    "unchanged": r["published_unchanged"],
    "rerun after fix equals clean": r["rerun_equals_clean_pass"]}
    for kind, r in break_results.items()]).to_string(index=False))
print(break_results["nulls"]["failed_checks"][0][2])
''')

md("""
---
## 5. The naive variants

Each variant is the correct pipeline with one switch changed, measured against
the same shop. Naive runs log their check failures but publish anyway, which is
what a pipeline with a separate quality report does.
""")

code('''
t0 = time.perf_counter()
RESULTS = run_all(CFG, days, ANSWER_KEY, correct)
print(f"measured six comparisons in {time.perf_counter() - t0:.0f} s (hardware dependent)")
''')

code('''
lc = RESULTS["late_commits"]
print(pd.DataFrame({
    "strict watermark": {
        "late commit rows planted": lc["strict"]["late"]["planted"],
        "late commit rows landed": lc["strict"]["late"]["landed"],
        "orders missing after day 30": lc["strict"]["orders"]["orders_missing"],
        "orders with a stale status": lc["strict"]["orders"]["status_stale"],
        "orders in the wrong region": lc["strict"]["orders"]["region_wrong"],
        "days with exact revenue (of 30)": lc["strict"]["every_day"]["days_exact"],
        "abs revenue error on day 30": usd(lc["strict"]["revenue"]["abs_error_cents"])},
    "lookback + dedup": {
        "late commit rows planted": lc["lookback"]["late"]["planted"],
        "late commit rows landed": lc["lookback"]["late"]["landed"],
        "orders missing after day 30": lc["lookback"]["orders"]["orders_missing"],
        "orders with a stale status": lc["lookback"]["orders"]["status_stale"],
        "orders in the wrong region": lc["lookback"]["orders"]["region_wrong"],
        "days with exact revenue (of 30)": lc["lookback"]["every_day"]["days_exact"],
        "abs revenue error on day 30": usd(lc["lookback"]["revenue"]["abs_error_cents"])},
}).to_string())
sg = lc["strict_gated"]
print(f"strict extract with gating checks stops on day {sg['day']} at {sg['failed_task']}:",
      sg["failed_checks"])

fig, ax = plt.subplots(figsize=(8.5, 3.8))
x = range(1, 31)
ax.step(x, [c / 100 for c in lc["strict"]["every_day"]["abs_error_cents_by_day"]],
        where="mid", color=ORANGE, label="strict watermark", zorder=3)
ax.step(x, [c / 100 for c in lc["lookback"]["every_day"]["abs_error_cents_by_day"]],
        where="mid", color=BLUE, label="lookback + dedup", zorder=4)
style(ax, "Published revenue error after each run",
      "absolute error across order date x region cells, in dollars", "day", "dollars")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
''')

code('''
hd = RESULTS["hard_deletes"]
print(pd.DataFrame({name: {
    "deleted orders still published": r["orders"]["deleted_still_published"],
    "published revenue": usd(r["revenue"]["published_cents"]),
    "true revenue": usd(r["revenue"]["truth_cents"]),
    "revenue overstated": usd(r["revenue"]["published_cents"] - r["revenue"]["truth_cents"])}
    for name, r in hd.items()}).to_string())

fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.plot(x, [c / 100 for c in hd["no_detection"]["every_day"]["total_error_cents_by_day"]],
        color=ORANGE, label="no delete detection", zorder=3)
ax.plot(x, [c / 100 for c in hd["reconciled"]["every_day"]["total_error_cents_by_day"]],
        color=BLUE, label="key reconciliation", zorder=4)
style(ax, "Published revenue minus true revenue",
      "deleted orders stay in the total, and the gap only grows", "day", "dollars")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
''')

code('''
ao = RESULTS["append_vs_overwrite"]
print(pd.DataFrame({name: {
    f"rows added by running day {r['rerun_day']} twice": r["rows_added_by_rerun"],
    "revenue double counted by that rerun": usd(r["revenue_added_by_rerun"]),
    "fct_order rows after day 30": r["orders"]["fct_rows"],
    "true orders": r["orders"]["orders_true"],
    "duplicate rows": r["orders"]["duplicate_rows"],
    "published revenue": usd(r["revenue"]["published_cents"])}
    for name, r in ao.items()}).to_string())

fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.plot(x, [c / 100 for c in ao["append"]["every_day"]["total_error_cents_by_day"]],
        color=ORANGE, label="append", zorder=3)
ax.plot(x, [c / 100 for c in ao["overwrite"]["every_day"]["total_error_cents_by_day"]],
        color=BLUE, label="partition overwrite", zorder=4)
style(ax, "Published revenue minus true revenue",
      "append keeps every version of every order", "day", "dollars")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
''')

code('''
wc = RESULTS["wall_clock"]
print(pd.Series({
    "past days backfilled": wc["days_backfilled"],
    "their own partitions present": wc["partitions_present"],
    f"rows in stray partition {wc['stray_partition']}": wc["rows_in_stray_partition"],
    "orders missing right after the backfill": wc["after_backfill"]["orders"]["orders_missing"],
    "orders with a stale status right after": wc["after_backfill"]["orders"]["status_stale"],
    "orders missing after day 30": wc["orders"]["orders_missing"],
    "published revenue after day 30": usd(wc["revenue"]["published_cents"]),
    "true revenue": usd(wc["revenue"]["truth_cents"])}).to_string())

fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.plot(x, [c / 100 for c in wc["every_day"]["abs_error_cents_by_day"]], color=ORANGE,
        label="wall clock partitioning", zorder=3)
ax.plot(x, [c / 100 for c in ao["overwrite"]["every_day"]["abs_error_cents_by_day"]],
        color=BLUE, label="logical run_date", zorder=4)
ax.axvspan(9, 13, color=MUTED, alpha=0.3, zorder=1)
style(ax, "Published revenue error after each run",
      "the shaded days are the backfill", "day", "dollars")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
''')

code('''
br = RESULTS["breaks"]
print(pd.DataFrame([{
    "break": kind, "check that fires": br[kind]["gate"]["first_failed_check"],
    "published before": usd(br[kind]["report"]["published_before"]),
    "published, report only": usd(br[kind]["report"]["published_after"]),
    "abs error, report only": usd(br[kind]["report"]["error_cents"]),
    "revenue with no region": usd(br[kind]["report"]["no_region_cents"]),
    "published, gated": usd(br[kind]["gate"]["published_after"]),
    "gated run": "stops at " + br[kind]["gate"]["failed_task"]}
    for kind in br]).to_string(index=False))

fig, ax = plt.subplots(figsize=(8.5, 3.9))
kinds = list(br)
pos = list(range(len(kinds)))
w = 0.36
report_err = [br[k]["report"]["error_cents"] / 100 for k in kinds]
gate_err = [br[k]["gate"]["error_vs_day_before_cents"] / 100 for k in kinds]
ax.bar([p - w / 2 for p in pos], report_err, w, color=ORANGE,
       label="checks as a report, published day 20", zorder=3)
ax.bar([p + w / 2 for p in pos], gate_err, w, color=BLUE,
       label="checks as gates, still the day 19 tables", zorder=3)
for p, v in zip(pos, report_err):
    ax.text(p - w / 2, v + 800, f"${v:,.0f}", ha="center", fontsize=9, color=INK2)
for p, v in zip(pos, gate_err):
    ax.text(p + w / 2, v + 800, f"${v:,.0f}", ha="center", fontsize=9, color=INK2)
ax.set_xticks(pos)
ax.set_xticklabels(kinds)
ax.set_ylim(0, 62000)
style(ax, "Published revenue error against the truth",
      "absolute error over order date x region cells, as of the day the tables describe",
      "", "dollars")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
''')

code('''
sd = RESULTS["scd_type"]
print(pd.DataFrame({name: {
    "orders in the wrong region": r["orders"]["region_wrong"],
    "date x region cells wrong": r["revenue"]["cells_wrong"],
    "abs revenue error across cells": usd(r["revenue"]["abs_error_cents"]),
    "total revenue error": usd(r["revenue"]["published_cents"] - r["revenue"]["truth_cents"])}
    for name, r in sd.items()}).to_string())

fig, ax = plt.subplots(figsize=(8.5, 3.8))
ax.plot(x, [c / 100 for c in sd["type1"]["every_day"]["abs_error_cents_by_day"]],
        color=ORANGE, label="type 1, current region", zorder=3)
ax.plot(x, [c / 100 for c in sd["type2"]["every_day"]["abs_error_cents_by_day"]],
        color=BLUE, label="type 2, region at order time", zorder=4)
style(ax, "Revenue attributed to the wrong region",
      "absolute error across order date x region cells", "day", "dollars")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()
''')

code('''
headline = {
    "naive": RESULTS,
    "clean_checksums": correct.wh.checksums(),
    "rerun": {"identical": rerun_result["identical"], "checksums": rerun_result["first"]["checksums"]},
    "backfill": {"equals_clean_pass": backfill_result["equals_clean_pass"],
                 "rows": [p["rows"] for p in backfill_result["partitions"]]},
    "breaks": {k: {"failed_task": r["failed_task"], "check": r["failed_checks"][0][1],
                   "unchanged": r["published_unchanged"]} for k, r in break_results.items()},
}
print("NOTEBOOK_RESULTS " + json.dumps(headline, sort_keys=True, default=str))
''')

md("""
---
## What this notebook does not prove

* The source is a simulation with a replayable history. A real OLTP table keeps
  only current rows, so a real backfill sees the latest version of each row and
  not every intermediate state.
* Lateness is bounded by `source.late_max_minutes`. A commit later than the
  lookback window is missed, and the reconciliation check is what would catch it.
* The data is small, and the staging models are rebuilt from all raw partitions
  on each run. At scale that step becomes incremental too.
* One process runs one task at a time. The DAG runner does not parallelise.
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

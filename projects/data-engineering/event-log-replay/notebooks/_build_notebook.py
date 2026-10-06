"""Builds the standalone notebook as plain text, so it diffs like code.

The notebook does not import the project. Instead this script READS the project
source and inlines the real definitions into code cells (by name, through `ast`),
so the notebook cannot drift from the repo. Only the pieces that touch the disk
are written fresh here, an in-memory log and the charts.
"""
from __future__ import annotations

import ast
import pprint
import sys
from pathlib import Path

import nbformat as nbf
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src" / "log_replay"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "event_log_replay_standalone.ipynb"
CFG = yaml.safe_load((ROOT / "conf" / "config.yaml").read_text(encoding="utf-8"))

nb = nbf.v4.new_notebook()
C: list = []


def md(t: str) -> None:
    C.append(nbf.v4.new_markdown_cell(t.strip("\n")))


def extract(rel: str, names: list[str]) -> str:
    """Return the source of the named top-level definitions, in the order given.

    Comment lines directly above a definition travel with it."""
    text = (SRC / rel).read_text(encoding="utf-8")
    lines = text.splitlines()
    found: dict[str, str] = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        else:
            continue
        if name not in names:
            continue
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        found[name] = "\n".join(lines[start: node.end_lineno])
    missing = [n for n in names if n not in found]
    assert not missing, f"{rel} has no top-level {missing}"
    return "\n\n\n".join(found[n] for n in names)


# Cell headers, in cell order. Each says what the cell consumes and produces, so
# a reader can drop into the middle of the notebook and still know the state.
CELL_HEADERS = [
    ("0.1", "Imports and global settings",
     "nothing", "the standard library, DuckDB and matplotlib namespaces",
     "Every number in this notebook comes from the seed in CFG, so a rerun prints the same figures."),
    ("0.2", "Chart styling",
     "nothing", "the colour palette and the style() helper",
     "One palette for every chart. Orange marks the failures, blue the baseline, green the fix."),
    ("1.1", "Configuration, the stream generator, and the answer key",
     "nothing", "CFG, generate(), Stream, partition_for(), Record",
     "These definitions are copied from the repo by the notebook builder. generate() writes the "
     "truth into stream.answer_key. Only score_sink() and the charts are allowed to read it."),
    ("1.2", "Generate the stream and show what was planted",
     "CFG, generate()", "stream, ANSWER_KEY, the planted-fault table",
     "Every count here is exact because each fault is placed at a configured rate."),
    ("2.1", "An in-memory log with Kafka semantics, and the consumer",
     "stream, partition_for()", "MemoryLog, Consumer, log, MANIFEST, crash_schedule()",
     "MemoryLog has the same methods as the repo's file-backed log. Consumer is copied unchanged. "
     "Nothing is written to disk."),
    ("2.2", "Check the log behaves like Kafka",
     "log, Consumer", "assertions only",
     "poll moves the in-memory position, only commit survives a restart, and a key keeps its order."),
    ("3.1", "The sink schema and the SQL every strategy shares",
     "nothing", "connect(), load_batch(), apply_deltas(), apply_profiles(), select_fresh(), "
     "mark_processed(), store_offsets()",
     "All strategies run this same arithmetic. They differ only in when it runs and in how many "
     "transactions."),
    ("3.2", "The sinks and the consume loop with crashes",
     "the SQL helpers, Consumer", "NaiveSink, CommitFirstSink, DedupSeparateTxnSink, AtomicSink, "
     "AtomicOffsetOnlySink, run_consumer()",
     "A crash throws the consumer away and keeps the database. That asymmetry is the whole problem."),
    ("4.1", "Score a sink against the answer key",
     "ANSWER_KEY, a finished sink", "score_sink(), consume(), run_and_score()",
     "Money is integer cents. Lost and doubled events come from effect_audit, because a summed "
     "balance cannot say which events moved it."),
    ("4.2", "Run every strategy over the same stream and crash schedule",
     "stream, log, MANIFEST, the sinks", "RESULTS, the scoreboard table",
     "About three minutes in total. Each run starts from offset zero into a fresh in-memory DuckDB."),
    ("4.3", "Chart accounts wrong and money drift",
     "RESULTS", "two charts",
     "Same bars, two views. The count says how many accounts are wrong, the drift says by how much."),
    ("4.4", "Chart lost and doubled events",
     "RESULTS", "one chart",
     "Commit-first loses events, the others double them, and the two errors have different signs."),
    ("5.1", "Profile policies",
     "RESULTS", "the profile table and one chart",
     "State events can be made idempotent, but only with a guard that names which update is newest."),
    ("6.1", "One account, end to end",
     "log, MANIFEST, RESULTS", "the trace table and one chart",
     "The featured account has a producer retry, a consumer crash and an out-of-order profile update."),
    ("7.1", "Replay proof, rewind to offset zero and diff the sink",
     "RESULTS, log", "REPLAYS, the replay table",
     "Reuses the sinks from the scoreboard. A fresh pass over the whole log goes into the same "
     "database and the tables are hashed before and after."),
    ("7.2", "Chart the replay result",
     "REPLAYS", "one chart",
     "Atomic and dedup are idempotent, naive and commit-first are not. Only atomic is also correct."),
    ("8.1", "Batch size against redelivery",
     "RESULTS, log, MANIFEST", "BATCHES, the batch table, one chart",
     "Correctness stays perfect at every batch size. Redelivery is what a larger transaction costs."),
    ("9.1", "Headline numbers as one JSON line",
     "RESULTS, REPLAYS, BATCHES", "NOTEBOOK_RESULTS",
     "The repo's test suite compares this line with run.py's artefacts."),
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
    rows = [thick, f"# {num}  {title}", thin]
    for label, text in (("In   ", ins), ("Out  ", outs), ("Note ", note)):
        if not text:
            continue
        first, *rest = _wrap(text, WIDTH - 11)
        rows.append(f"# {label}: {first}")
        rows += [f"#        {r}" for r in rest]
    rows.append(thick)
    return "\n".join(rows)


def code(src: str, header: bool = True) -> None:
    """Append a code cell, prefixed with its header banner."""
    body = src.strip("\n")
    if header:
        body = _banner(*CELL_HEADERS.pop(0)) + "\n" + body
    C.append(nbf.v4.new_code_cell(body))


md("""
# Replay the Log

## At-least-once delivery, idempotent sinks, and a replay you can diff

**Standalone notebook.** It generates its own wallet event stream, runs a log with
Kafka semantics in memory, and compares five ways of writing consumed events into a
database. It defines every function it uses, imports nothing from the project
package, and writes nothing to disk. The definitions are copied from the repo
source by the notebook builder, so they cannot drift from `run.py`.

### What you will measure

A wallet service writes about 50,000 events to a log with 3 partitions. The log
contains three planted faults.

1. The producer retried 1,000 events, so the same `event_id` sits at two offsets.
2. 400 profile updates arrive after a newer update for the same account.
3. The consumer is killed 52 times, at positions that are the same for every strategy.

Every strategy sees the same stream and the same crashes. The answer key says what
each balance and profile should be, so each strategy gets a score and not an opinion.

### Before you run

```
%pip install -q duckdb
```

Colab and Kaggle already have matplotlib. The full notebook takes about 3 minutes
because each of the strategies makes a pass over 51,000 records in transactions.
It has 6 charts.

### Contents

| # | Section | What you get |
|---|---|---|
| 1 | [Stream and answer key](#s1) | a log with known faults at exact counts |
| 2 | [The log and the consumer](#s2) | partitions, offsets, commit, seek |
| 3 | [The sinks](#s3) | naive, commit first, dedup, atomic |
| 4 | [The scoreboard](#s4) | accounts wrong, money drift, lost and doubled events |
| 5 | [Profile policies](#s5) | last arrived wins against a guarded upsert |
| 6 | [One account](#s6) | a duplicate, a crash and an out-of-order update, followed through |
| 7 | [The replay proof](#s7) | rewind to zero and diff the sink |
| 8 | [Batch size](#s8) | redelivery grows, correctness does not |
| 9 | [What this does and does not show](#s9) | the honest limits |

Every code cell opens with a header saying what it consumes and what it leaves
behind, so you can start reading at any section.
""")

code("%pip install -q duckdb", header=False)

code('''
import hashlib
import json
import random
import re
import zlib
from dataclasses import asdict, dataclass, field
from typing import Protocol

import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt

print("duckdb", duckdb.__version__)
''')

code('''
# Orange marks failures, blue is the baseline, green is the fix.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b8b7b2"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "600", "axes.titlelocation": "left", "axes.titlepad": 12,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10,
    "grid.color": "#e8e7e3", "grid.linewidth": 0.8, "legend.frameon": False,
    "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 7,
    "figure.dpi": 110, "font.size": 10,
})


def style(ax, title=None, sub=None, xlabel=None, ylabel=None, grid="x"):
    if title:
        ax.set_title(title, pad=22 if sub else 12)
    if sub:
        ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    if grid:
        ax.grid(axis=grid, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
    return ax
''')

md("""
<a id="s1"></a>
## 1. The stream and the answer key

The generator builds every event in event-time order first and applies the delivery
faults afterwards. So a fault is always a statement about arrival, and the truth never
changes. Two kinds of event matter.

- **Delta events** are wallet deposits and withdrawals. Applying `+500` twice is wrong
  however the row is keyed.
- **State events** are profile updates carrying absolute values. An upsert can make
  these idempotent, if it knows which update is newest.
""")

cfg_literal = pprint.pformat(CFG, sort_dicts=False, width=100)
gen_src = "\n\n\n".join([
    extract("log/base.py", ["Record", "partition_for"]),
    extract("data/generate.py", ["TIERS", "Stream", "_event_id", "_build_events", "_plant_ties",
                                 "_plant_late", "_plant_retries", "_truth",
                                 "_out_of_order_count", "_pick_featured", "generate"]),
])
code(f"CFG = {cfg_literal}\n\n\n{gen_src}")

code('''
stream = generate(CFG)
ANSWER_KEY = stream.answer_key   # read only by score_sink() and the charts
facts = {k: v for k, v in ANSWER_KEY["planted"].items() if k != "retried_event_ids"}
for k, v in facts.items():
    print(f"{k:24s} {v:>8,}")
''')

md("""
<a id="s2"></a>
## 2. The log and the consumer

`MemoryLog` has the same methods as the file log in the repo. The consumer is copied
unchanged. Its `poll` moves an in-memory position, and only `commit` makes that
position survive a restart. That distinction is what every sink below leans on.
""")

mem_log = '''
class MemoryLog:
    """Partitions as lists, committed offsets in a dict, the same interface as FileLog."""

    def __init__(self, n_partitions):
        self.n_partitions = n_partitions
        self._parts = [[] for _ in range(n_partitions)]
        self._committed = {}

    def append(self, partition, key, value):
        offset = len(self._parts[partition])
        self._parts[partition].append(Record(partition, offset, key, value))
        return offset

    def end_offset(self, partition):
        return len(self._parts[partition])

    def read(self, partition, offset, n):
        return self._parts[partition][offset: offset + n]

    def committed(self, group):
        return dict(self._committed.get(group, {}))

    def commit(self, group, offsets):
        self._committed.setdefault(group, {}).update(offsets)

    def reset_group(self, group):
        self.commit(group, {p: 0 for p in range(self.n_partitions)})


log = MemoryLog(CFG["stream"]["n_partitions"])
for rec in stream.records:
    key = str(rec["account_id"])
    log.append(partition_for(key, log.n_partitions), key, rec)

MANIFEST = {"partition_lengths": [log.end_offset(p) for p in range(log.n_partitions)],
            "featured": stream.featured}
print("records per partition", MANIFEST["partition_lengths"])
'''
fault_src = "\n\n\n".join([
    extract("log/base.py", ["LogBackend", "Consumer"]),
    extract("faults.py", ["Position", "build_schedule"]),
    extract("experiments.py", ["crash_schedule"]),
])
code(fault_src + "\n\n\n" + mem_log)

code('''
c = Consumer(log, "demo")
first = c.poll(10)
assert len(first) == 10 and sum(c.position.values()) == 10
assert Consumer(log, "demo").position == {0: 0, 1: 0, 2: 0}, "nothing committed yet"
c.commit()
assert Consumer(log, "demo").position == c.position
c.seek(0, 0)
assert c.position[0] == 0
for p in range(log.n_partitions):                       # a key never changes partition
    assert all(partition_for(r.key, log.n_partitions) == p for r in log.read(p, 0, 10**6))
print("poll moves the position, commit makes it durable, seek rewinds it")
''')

md("""
<a id="s3"></a>
## 3. The sinks

All four strategies run the same SQL. The difference is when it runs and how many
transactions it takes.

| strategy | order of operations | what a crash does |
|---|---|---|
| `naive` | write, then commit the offset | the write is replayed, so duplicates |
| `commit_first` | commit the offset, then write | the write never happens, so lost events |
| `dedup_separate_txn` | transaction 1 writes, transaction 2 records the ids | the ids are missing, so duplicates |
| `atomic` | write, ids and offset in one transaction | everything rolls back, so nothing |

`atomic_offset_only` is an ablation. It keeps the offset in the transaction and drops
the id table.
""")

sql_src = extract("sinks/sql.py", [
    "SCHEMA", "PROFILE_POLICIES", "WALLET", "PROFILE", "FRESH", "_COLS", "connect", "delta_of",
    "load_batch", "apply_deltas", "_BATCH_WINNER", "_UPSERT_GUARD", "apply_profiles",
    "select_fresh", "mark_processed", "store_offsets"])
code(sql_src)

sinks_src = "\n\n\n".join([
    extract("sinks/strategies.py", ["Crash", "Sink", "NaiveSink", "CommitFirstSink",
                                    "DedupSeparateTxnSink", "AtomicSink", "AtomicOffsetOnlySink",
                                    "SINKS"]),
    extract("runner.py", ["RunStats", "run_consumer"]),
])
code(sinks_src)

md("""
<a id="s4"></a>
## 4. The scoreboard

`score_sink` is the only function that reads the answer key. It reports

- **accounts wrong**, balances that differ from the truth by any number of cents,
- **net drift** and **absolute drift**, in currency units. Net drift can hide errors
  that cancel, absolute drift cannot,
- **events lost** and **applied twice**, counted per event from the audit table,
- **profiles wrong**,
- **redelivered**, records the sink was handed a second time.
""")

score_src = "\n\n\n".join([
    extract("evaluation/score.py", ["score_sink"]),
    extract("experiments.py", ["STRATEGIES", "make_sink", "group_for", "consume"]),
    extract("report.py", ["money", "table"]),
])
code(score_src + '''


def run_and_score(cfg, log, strategy, crashes, batch_size=None):
    sink, stats = consume(cfg, log, strategy, crashes, batch_size)
    return {"strategy": strategy, "sink": sink, "stats": asdict(stats),
            "score": score_sink(sink.con, ANSWER_KEY), "crashes": len(crashes),
            "batch_size": batch_size or cfg["consumer"]["batch_size"]}
''')

code('''
CRASHES = crash_schedule(CFG, MANIFEST)
RESULTS = {"naive (no crashes)": run_and_score(CFG, log, "naive", [])}
for name in ["naive", "commit_first", "dedup_separate_txn", "atomic_offset_only", "atomic",
             "atomic_last_arrived_wins", "atomic_time_only_guard"]:
    RESULTS[name] = run_and_score(CFG, log, name, CRASHES)

SHOWN = ["naive (no crashes)", "naive", "commit_first", "dedup_separate_txn",
         "atomic_offset_only", "atomic"]
rows = []
for name in SHOWN:
    r = RESULTS[name]
    s = r["score"]
    rows.append([name, r["crashes"], f"{r['stats']['redelivered']:,}", s["accounts_wrong"],
                 money(s["net_drift_cents"]), money(s["abs_drift_cents"]), s["events_lost"],
                 s["applied_twice"], s["profiles_wrong"]])
print(f"{len(CRASHES)} scheduled crashes, batch size {CFG['consumer']['batch_size']}\\n")
print(table(["strategy", "crashes", "redelivered", "accounts wrong", "net drift", "abs drift",
             "events lost", "applied twice", "profiles wrong"], rows))
''')

code('''
labels = SHOWN[::-1]
wrong = [RESULTS[n]["score"]["accounts_wrong"] for n in labels]
drift = [RESULTS[n]["score"]["net_drift_cents"] / 100 for n in labels]
colours = [AQUA if n == "atomic" else (BLUE if "no crashes" in n or n == "atomic_offset_only" else ORANGE)
           for n in labels]
n_accounts = CFG["stream"]["n_accounts"]

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
ax[0].barh(labels, wrong, color=colours, zorder=3)
for y, v in enumerate(wrong):
    ax[0].text(v + 20, y, f"{v:,}", va="center", fontsize=9)
style(ax[0], "Accounts with a wrong balance", f"out of {n_accounts:,}, same stream and crashes for each")
ax[0].set_xlim(0, n_accounts)

ax[1].barh(labels, drift, color=colours, zorder=3)
ax[1].axvline(0, color=INK2, linewidth=0.8)
style(ax[1], "Net money drift", "currency units, positive means money was invented")
ax[1].set_yticklabels([])
ax[1].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
plt.tight_layout(); plt.show()
''')

code('''
names = ["naive", "commit_first", "dedup_separate_txn", "atomic_offset_only", "atomic"]
twice = [RESULTS[n]["score"]["applied_twice"] for n in names]
lost = [RESULTS[n]["score"]["events_lost"] for n in names]

fig, ax = plt.subplots(figsize=(9.5, 4.2))
x = range(len(names))
ax.bar([i - 0.2 for i in x], twice, width=0.4, color=ORANGE, label="applied twice", zorder=3)
ax.bar([i + 0.2 for i in x], lost, width=0.4, color=BLUE, label="lost", zorder=3)
for i, (a, b) in enumerate(zip(twice, lost)):
    ax.text(i - 0.2, a + 40, f"{a:,}", ha="center", fontsize=9)
    ax.text(i + 0.2, b + 40, f"{b:,}", ha="center", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=12)
style(ax, "Wallet events applied twice or never", "per event, counted from the audit table", grid="y")
ax.legend()
plt.tight_layout(); plt.show()
''')

md("""
<a id="s5"></a>
## 5. Profile policies

Profile updates carry absolute values, so an upsert can make them idempotent. The
policy decides which update wins.

- `last_arrived_wins` overwrites, so a late update replaces a newer one.
- `time_only_guard` keeps the newest `event_time`, but 40 accounts have two updates in
  the same second, so event time cannot say which is newer.
- `guarded_upsert` compares `(event_time, version)`, and the version breaks the tie.
""")

code('''
POLICY_RUNS = ["atomic_last_arrived_wins", "atomic_time_only_guard", "atomic"]
POLICY_NAMES = ["last_arrived_wins", "time_only_guard", "guarded_upsert"]
with_profile = facts["accounts_with_profile"]
print(table(["profile policy", "profiles wrong", "accounts with a profile"],
            [[p, RESULTS[r]["score"]["profiles_wrong"], f"{with_profile:,}"]
             for p, r in zip(POLICY_NAMES, POLICY_RUNS)]))

vals = [RESULTS[r]["score"]["profiles_wrong"] for r in POLICY_RUNS]
fig, ax = plt.subplots(figsize=(8.6, 3.6))
ax.barh(POLICY_NAMES[::-1], vals[::-1], color=[AQUA, BLUE, ORANGE], zorder=3)
for y, v in enumerate(vals[::-1]):
    ax.text(v + 4, y, str(v), va="center", fontsize=9)
style(ax, "Profiles wrong by policy", f"out of {with_profile:,} accounts with a profile")
plt.tight_layout(); plt.show()
''')

md("""
<a id="s6"></a>
## 6. One account, end to end

The generator picks the smallest account that shows all three faults, and pins one
consumer crash on one of its events. Read the flags column, then the result per strategy.
""")

code('''
f = MANIFEST["featured"]
account, part = f["account_id"], f["partition"]
crash_set = set(CRASHES)
rows, seen, newest = [], set(), 0
for rec in log.read(part, 0, log.end_offset(part)):
    if rec.key != str(account):
        continue
    v, flags = rec.value, []
    if v["event_id"] in seen:
        flags.append("DUPLICATE (producer retry)")
    if v["type"] == "profile_updated":
        if v["version"] < newest:
            flags.append("OUT OF ORDER")
        newest = max(newest, v["version"])
    seen.add(v["event_id"])
    if (rec.partition, rec.offset) in crash_set:
        flags.append("CONSUMER CRASH HERE")
    what = (f"{v['kind']} {money(v['amount_cents'])}" if v["type"] == "wallet_txn"
            else f"v{v['version']} {v['tier']} limit {money(v['daily_limit_cents'])}")
    rows.append([rec.offset, v["event_id"], v["type"], what, v["event_time"],
                 v["producer_attempt"], ", ".join(flags)])
print(f"Account {account} lives in partition {part}\\n")
print(table(["offset", "event_id", "type", "event", "event_time", "attempt", "flags"],
            rows, ["r", "l", "l", "l", "r", "r", "l"]))

truth_balance = ANSWER_KEY["balances"][str(account)]
ids = {r[1] for r in rows}
out, errors = [], {}
for name in ["naive", "commit_first", "dedup_separate_txn", "atomic"]:
    con = RESULTS[name]["sink"].con
    bal = con.execute("SELECT balance_cents FROM balances WHERE account_id = ?", [account]).fetchone()[0]
    applied = dict(con.execute("SELECT event_id, COUNT(*) FROM effect_audit GROUP BY event_id").fetchall())
    odd = {e: applied.get(e, 0) for e in ids if e.startswith("w-") and applied.get(e, 0) != 1}
    errors[name] = (bal - truth_balance) / 100
    out.append([name, money(bal), money(bal - truth_balance),
                ", ".join(f"{e} x{n}" for e, n in sorted(odd.items())) or "-"])
print(f"\\ntruth balance {money(truth_balance)}   (x0 = lost, x2 = applied twice)\\n")
print(table(["strategy", "balance", "error", "events applied wrongly"], out, ["l", "r", "r", "l"]))

fig, ax = plt.subplots(figsize=(8.6, 3.6))
ax.barh(list(errors)[::-1], list(errors.values())[::-1],
        color=[AQUA if v == 0 else ORANGE for v in list(errors.values())[::-1]], zorder=3)
ax.axvline(0, color=INK2, linewidth=0.8)
style(ax, f"Balance error for account {account}", "currency units against the answer key")
plt.tight_layout(); plt.show()
''')

md("""
<a id="s7"></a>
## 7. The replay proof

Rewind the consumer to offset zero, consume the whole log into the **same** database,
and hash the tables before and after. A sink that was idempotent gives the same hash.

The proof detects non-idempotence. It cannot repair damage that was already there, so
the table also shows how many accounts were wrong before and after.
""")

code(extract("experiments.py", ["CHECKED_TABLES", "_rows", "checksum", "rows_differing", "replay_proof"]) + '''


REPLAYS = {}
for name in ["naive", "commit_first", "dedup_separate_txn", "atomic"]:
    REPLAYS[name] = replay_proof(CFG, log, name, CRASHES, ANSWER_KEY, sink=RESULTS[name]["sink"])

rows = []
for name, p in REPLAYS.items():
    rows.append([name, "checksum identical" if p["identical"] else "CHECKSUM DIFFERS",
                 p["balances_differ"], p["profiles_differ"], p["processed_differ"],
                 p["wrong_before"], p["wrong_after"], money(p["net_drift_before"]),
                 money(p["net_drift_after"])])
print(table(["strategy", "verdict", "balance rows differ", "profile rows differ",
             "dedup rows differ", "wrong before", "wrong after", "net drift before",
             "net drift after"], rows, ["l", "l"] + ["r"] * 7))
''')

code('''
names = list(REPLAYS)
before = [REPLAYS[n]["wrong_before"] for n in names]
after = [REPLAYS[n]["wrong_after"] for n in names]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
x = range(len(names))
ax.bar([i - 0.2 for i in x], before, width=0.4, color=BLUE, label="before the replay", zorder=3)
ax.bar([i + 0.2 for i in x], after, width=0.4, color=ORANGE, label="after the replay", zorder=3)
for i, (a, b) in enumerate(zip(before, after)):
    ax.text(i - 0.2, a + 25, f"{a:,}", ha="center", fontsize=9)
    ax.text(i + 0.2, b + 25, f"{b:,}", ha="center", fontsize=9)
ax.set_xticks(list(x)); ax.set_xticklabels(names)
style(ax, "Accounts wrong before and after a full replay", "idempotent sinks do not move, the others do", grid="y")
ax.legend()
plt.tight_layout(); plt.show()
''')

md("""
<a id="s8"></a>
## 8. Batch size against redelivery

The atomic sink commits one transaction per polled batch. A crash rolls the whole
batch back, so a larger batch means more records handled a second time. The result
stays exact at every size. Only the wasted work changes.
""")

code('''
BATCHES = {}
for b in [50, 100, 500, 5000]:
    BATCHES[b] = RESULTS["atomic"] if b == CFG["consumer"]["batch_size"] else \\
        run_and_score(CFG, log, "atomic", CRASHES, b)

rows = []
for b, r in BATCHES.items():
    s, st = r["score"], r["stats"]
    rows.append([b, f"{st['polls'] - st['crashes']:,}", st["crashes"], f"{st['records_handled']:,}",
                 f"{st['redelivered']:,}", s["accounts_wrong"], s["profiles_wrong"],
                 money(s["net_drift_cents"])])
print(table(["batch size", "transactions", "crashes", "records handled", "redelivered",
             "accounts wrong", "profiles wrong", "net drift"], rows))

sizes = list(BATCHES)
red = [BATCHES[b]["stats"]["redelivered"] for b in sizes]
fig, ax = plt.subplots(figsize=(8.6, 4))
ax.plot(sizes, red, marker="o", color=ORANGE, zorder=3)
for b, v in zip(sizes, red):
    ax.annotate(f"{v:,}", (b, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(sizes); ax.set_xticklabels([str(b) for b in sizes])
ax.set_ylim(top=max(red) * 3)
style(ax, "Redelivered records grow with batch size", "accounts wrong is 0 at every size",
      xlabel="records per transaction", ylabel="records handled a second time", grid="y")
plt.tight_layout(); plt.show()
''')

code('''
BOARD = {name: [RESULTS[name]["score"][k] for k in
                ("accounts_wrong", "net_drift_cents", "abs_drift_cents", "events_lost",
                 "applied_twice", "profiles_wrong")] + [RESULTS[name]["stats"]["redelivered"]]
         for name in SHOWN}
NOTEBOOK_RESULTS = {
    "scoreboard": BOARD,
    "profiles": {p: RESULTS[r]["score"]["profiles_wrong"] for p, r in zip(POLICY_NAMES, POLICY_RUNS)},
    "replay": {n: [p["identical"], p["balances_differ"], p["wrong_before"], p["wrong_after"]]
               for n, p in REPLAYS.items()},
    "batches": {str(b): [r["stats"]["polls"] - r["stats"]["crashes"], r["stats"]["records_handled"],
                         r["stats"]["redelivered"], r["score"]["accounts_wrong"]]
                for b, r in BATCHES.items()},
}
print("NOTEBOOK_RESULTS " + json.dumps(NOTEBOOK_RESULTS, sort_keys=True))
''')

md("""
<a id="s9"></a>
## 9. What this does and does not show

**What the numbers say.**

- Delivery is at-least-once, so the sink has to cope with repeats. Committing the offset
  before the write loses events, and committing after it repeats them.
- A dedup table in a separate transaction removes the producer retries, but a crash
  between the two transactions still repeats the batch. Two transactions can only choose
  which way to be wrong.
- Putting the effect, the event ids and the offset in one transaction gives zero drift.
  The offset alone is not enough, because a producer retry is a second record with a
  new offset.
- Delta events cannot be made idempotent by an upsert. They need effect-level dedup.
  State events can, if the guard compares something that names the newest update.
- The replay proof detects a sink that is not idempotent. It does not repair a sink
  that is already wrong.

**What it does not show.**

- The log is a single process and a single consumer. It has no rebalancing, no
  concurrent consumers and no broker failures.
- Crashes happen at chosen points. A real process can die between any two instructions.
- DuckDB stands in for an OLTP database. Transaction semantics are real, throughput
  numbers are not, and none are claimed.
- Out-of-order updates here are late by a bounded gap. A truly unbounded delay needs a
  watermark policy, which is not modelled.
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
with open(OUT, "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print(f"wrote {OUT} -- {len(C)} cells "
      f"({sum(c['cell_type'] == 'code' for c in C)} code, "
      f"{sum(c['cell_type'] == 'markdown' for c in C)} markdown)")

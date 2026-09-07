#!/usr/bin/env python
"""Generate notebooks/LLM_Judge_Lifecycle.ipynb.

    python scripts/build_notebook.py

Authored as a generator so the prose lives in reviewable Python rather than in
JSON with escaped newlines, and so a wording fix is a normal diff.

The notebook is OFFLINE ONLY by construction. It never reads an API key and
never makes a network call after the install cell, which is what lets it run in
Colab for anyone, at zero cost, in about a minute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "LLM_Judge_Lifecycle.ipynb"
REPO = "https://github.com/genieincodebottle/aiml-companion.git"

cells: list[dict] = []


def _cid() -> str:
    # Stable, content-independent ids. nbformat >= 4.5 requires one per cell, and
    # deriving them from the index keeps the diff clean when prose changes.
    return f"cell-{len(cells):02d}"


def md(text: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "id": _cid(),
            "metadata": {},
            "source": text.strip().splitlines(True),
        }
    )


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "id": _cid(),
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.strip().splitlines(True),
        }
    )


# ---------------------------------------------------------------- 0. intro
md("""
# The Lifecycle of an LLM Judge

**No API key. No network after the install cell. Runs in about a minute.**

Almost every team building with LLMs ends up with a judge: a model that scores
another model's output. Almost every team then treats it as a *benchmark score* -
built once, validated once, quoted for a year.

This notebook walks the alternative: a judge as a **service with a deployment, a
bill and a decay curve**, in four phases.

---

## The question the whole project turns on

Two judges. Same explanation, shown beside a film. Both rejected it.

```
ARTEFACT   "The Quiet Harbour is a slow-burn mystery rated 104 by viewers."

RECORD     title: The Quiet Harbour        runtime_min: 104

JUDGE A    FAIL - "104 is the runtime in minutes, not a rating."
JUDGE B    FAIL - "This explanation is too short to be useful."
```

On any label-accuracy metric they are **identical**. They are not the same.

Judge B is right by accident. It will pass this the moment explanations get
longer - and in this system the judge's reason is handed back to the writer as
its instruction, so B tells it to *write more* while the actual error survives
every retry.

Measuring that difference is a metric. Fixing it is a training signal. Noticing
when it returns is a monitor. That is the lifecycle.
""")

# ---------------------------------------------------------------- 1. setup
md("""
## Setup

One dependency. The vendor SDKs are imported lazily, so running offline needs
none of them.
""")

code(f"""
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and not Path("llm-judge-lifecycle").exists():
    subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
    os.chdir("aiml-companion/projects/llm/llm-judge-lifecycle")
elif Path("../src").exists():
    os.chdir("..")                     # running from notebooks/ in a clone

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=True)
sys.path.insert(0, os.getcwd())
print("working directory:", os.getcwd())
""")

md("""
### Why there is no API key

`--offline` is **not a mock**. A rubric here is markdown a model reads, and some
bullets carry an inline tag:

```
- Reject filler that would fit any item in the catalogue.
  [banned: "you'll love it", "a must-watch"]
- Every factual claim must be traceable to the record. [grounded]
```

An LLM judge ignores the brackets and reads the sentence. `src/rules.py` ignores
the sentence and reads the brackets. **Same rubric, two readers.**

So offline you get a real judge - deterministic, lexical, weak in documented
ways. That is worth more than a canned demo, because it is the **baseline any
model judge has to beat**, and most published judge results never report one.
""")

code("""
from src.services._context import build_context

ctx = build_context(offline=True)      # every role -> the rule engine
domain = ctx.domain

print("domain:", domain.display_name)
for c in domain.criteria:
    kind = "GATE (drops the artefact)" if c.must_have else "soft (recorded, still served)"
    print(f"  {c.id:<10} {kind}")
""")

# ------------------------------------------------------------- 2. phase I
md("""
---
# Phase I - Birth

The most human-intensive phase, the one everyone wants to skip, and the one every
other number rests on. A judge tuned against a careless benchmark is carefully
aligned to nothing.
""")

code("""
from src.services import BenchmarkService
import logging; logging.getLogger("src.benchmark").setLevel(logging.ERROR)

report = BenchmarkService(ctx).report()["splits"]["criteria"]
for cid, row in report.items():
    kind = "gate" if row["must_have"] else "soft"
    print(f"{cid:<10} {kind:<5} n={row['n']:<4} fail={row['fail_fraction']:<6} {row['splits']}")
""")

md("""
Notice `fail ~ 0.5` on every gate criterion. That is **deliberate and it changes
what the numbers mean.**

Real defect rates are a few percent. A naturally-sampled benchmark would be ~95%
PASS, and a judge that answered PASS to everything would score 95% while
catching nothing at all. Balancing is what makes "did it catch the bad ones?"
measurable.

The cost: **these are alignment metrics, not defect rates.** Any report that
presents them as production quality is wrong.

### The examples that matter most

Some rows are labelled `BASELINE-MISS` - cases the rule engine provably cannot
catch. They are the headroom a model judge has to earn.
""")

code("""
for eid in ("ex-g08", "ex-g12", "ex-s13"):
    ex = next(e for e in domain.examples if e.id == eid)
    cid = next(iter(ex.labels))
    print(f"[{eid}]  human says {ex.labels[cid]} on '{cid}'")
    print(f"   artefact : {ex.artefact}")
    print(f"   why      : {ex.rationales.get(cid, '')}\\n")
""")

md("""
Read `ex-g08` again. **104 is real** - it is the runtime - and the claim is
false. Every token appears in the record, so a checker that asks "does this
number exist in the source?" passes it. That is the most common way a
groundedness guardrail is quietly wrong.

`ex-g12` is worse: the record says *no aliens appear on screen* and the artefact
says they do. Nothing is fabricated at the token level. Catching it means reading
the record as a set of claims rather than a bag of strings.
""")

# ------------------------------------------------------------ 3. baseline
md("""
### What the rule engine scores

Before claiming an LLM judge earns its latency and its bill, beat the rules you
could have written instead.
""")

code("""
svc = BenchmarkService(ctx)
for cid in ("grounded", "specific", "safe"):
    r = svc.evaluate(cid, split="validation")
    m = r["metrics"]
    lo, hi = m["ci95"]["specificity"]
    print(f"{cid:<10} specificity={m['specificity']:.3f}  95% CI [{lo:.2f}, {hi:.2f}]"
          f"   recall={m['recall']:.3f}")
""")

md("""
**Read the intervals, not the point estimates.** They are forty to fifty points
wide because the splits hold five to seven examples. A difference narrower than
the interval is noise, and this tool prints the interval next to every number so
that you cannot forget.
""")

# ------------------------------------------------------------ 4. phase II
md("""
---
# Phase II - Training (RART)

**No gradients. No fine-tuning. No weights are touched anywhere in this repo.**

The rubric *text* is the parameter and a reflector model is the optimiser. The
loop is gradient descent with every numeric part replaced by language: score the
rubric, collect the mistakes, ask for a better rubric, keep it only if
**validation** improves.
""")

code("""
from src.services import TuningService

result = TuningService(ctx).tune("specific")
print("iter  weighted  spec    rec     ra      focus  kept")
for it in result["iterations"]:
    v = it["validation"]
    fmt = lambda x: "  n/a " if x is None else f"{x:.3f}"
    print(f"  {it['index']}     {it['weighted']:<9.3f} {fmt(v['specificity'])}   "
          f"{fmt(v['recall'])}   {fmt(v['reasoning_agreement'])}   "
          f"{it['focus_size']:<6} {'*' if it['accepted'] else ''}")
print("\\nstopped:", result["stopped_because"])
""")

md("""
### Now read what the optimiser wrote

This is the part worth dwelling on: the optimiser's output is English.
""")

code("""
import difflib
seed = domain.seed_rubric("specific").splitlines()
tuned = result["best_rubric"].splitlines()
print("\\n".join(difflib.unified_diff(seed, tuned, "seed", "tuned", lineterm="", n=0)))
""")

md("""
One clause, and you can read it. When this judge gets something wrong you open
the file and see why - which is the argument for tuning the rubric instead of the
weights.

### The null result that is a result

Run the same thing on `safe`. It does **not** improve, and the tool distinguishes
two completely different reasons for that.
""")

code("""
safe = TuningService(ctx).tune("safe")
best = safe["iterations"][safe["best_iteration"]]["validation"]
print("improved on seed:", safe["improved_on_seed"])
print("specificity     :", best["specificity"])
print("stopped         :", safe["stopped_because"])
""")

md("""
Specificity is **0.5**, so this is *not* a criterion at ceiling. There is plenty
of headroom and the offline optimiser could not reach it.

That is expected, and it is the most useful thing here. The offline reflector
learns only *lexical* rules - ban a phrase, require a token - and these failures
are not lexical: a spoiler phrased in words nobody listed, difficult subject
matter used as a hook, a number attached to the wrong noun.

**That gap is what a model judge is for.** Run this against Gemini
(`build_context(offline=False)` with a key) and `safe` scores 1.000 on the same
split. The measured comparison is in `docs/results.md`.

"No improvement" therefore means two opposite things - "already at ceiling"
versus "there is headroom and we missed it" - and reporting both with one
sentence is how a broken optimiser gets read as a validated one.
""")

# ----------------------------------------------------------- 5. phase III
md("""
---
# Phase III - Deployment

The same judge plays two roles at once. As **gate** it rejects. As **critic** its
rejection reason becomes the writer's instruction for the next attempt.

That dual role is why a right-verdict-but-wrong-reason rejection is a real defect
rather than a philosophical one: the wrong reason points the rewrite at the wrong
problem, the budget is spent, and the artefact is dropped anyway.
""")

code("""
from src.services import ServingService

curve = ServingService(ctx).retry_curve(max_k=4)
for p in curve["curve"]:
    bar = "#" * round(p["cumulative_pass_rate"] * 40)
    print(f"k={p['k']}  {p['cumulative_pass_rate']:.3f}  {bar} ({p['passed']}/{p['total']})")
""")

md("""
Monotone, then flat. The shape is what turns `max_retries` from a guess into a
decision:

- **k=0 is the generator's unaided pass rate.** A sustained drop *there* is a
  generator regression, not judge drift - and in an aggregate pass rate the two
  look identical, so you would debug the wrong one.
- **Where it flattens** is where extra retries stop buying quality and start
  being a linear cost on every request.
- **Flat AND low** means the generator is too weak for revision to rescue. Fix
  the writer; do not raise K.

Against a real model this curve looks completely different - 0.95 at k=0 - which
is exactly why you must never pick K from a curve somebody else measured.

### What happens when the budget runs out
""")

code("""
run = ServingService(ctx).serve_all()
print("outcomes:", run["outcomes"])
print("pass rate:", run["pass_rate"])
""")

md("""
The dropped ones were **not served**. Not served with a warning, not served as
the best of a bad set. Dropped.

```
a bad artefact served    ->  reaches a user, cannot be recalled
a good artefact dropped  ->  one missed opportunity
```

Those are not the same error, and a system that treats them as one optimises for
coverage and pays in trust. It is one `if`, and it is the whole design.
""")

# ------------------------------------------------------------ 6. phase IV
md("""
---
# Phase IV - Monitoring

The phase that is always deferred, and the only one that tells you the other
three have stopped working.
""")

code("""
from src.services import MonitoringService

mon = MonitoringService(ctx)
for week in (5, 6):
    res = mon.check(week)
    print(f"--- week {week}   alert={res['alert']}")
    for rep in res["reports"]:
        for scope, checks in (("overall", rep["overall"]), ("new items", rep["new_items"])):
            for c in checks:
                if c["judge"] is None:
                    continue
                state = "in band" if c["in_band"] else "OUT OF BAND"
                print(f"   {scope:<10} {c['metric']:<12} judge={c['judge']:.3f} "
                      f"raters={c['rater_mean']:.3f}+-{c['rater_sd']:.3f} "
                      f"floor={c['lower_bound']:.3f}  {state}")
    print()
""")

md("""
Two ideas, and they are the best things in the project.

### 1. The threshold floats

```
judge  >=  mean(rater_scores)  -  2 x sd(rater_scores)
```

Not 0.85. Not any fixed number - because **human raters do not agree with each
other by a constant amount.**

On a week of genuinely ambiguous artefacts the raters disagree more, `sd` widens,
and the band widens with it, so the judge is not penalised for finding hard what
people also found hard. Look at week 5's new-item rows: the raters were
unanimous, `sd = 0`, and the floor equals the mean. **A week nobody found
ambiguous gives the judge no slack at all.** That is correct.

A fixed threshold fails in both directions: it fires every hard week (alert
fatigue, then a muted alert, then it is not a monitor), and it sleeps through a
slow slide on easy ones.

### 2. The aggregate hides the drift

Week 6 overall is **in band**. New items are **out of band**.

Four titles with difficult subject matter arrived; the generator started using
that material as a hook; the `safe` rubric was tuned on a catalogue where none of
it existed. Ten of the week's fourteen failures came from the established
catalogue where the judge is still good, so the aggregate looks fine.

The judge is now wrong about most of what the service is *newly* recommending,
and the aggregate will keep looking healthy for as long as new titles are a
minority of traffic. By the time it moves, the judge has been passing unsafe
explanations for a month.

`check_new_items_separately: true` is one line of config and the only reason this
is visible.
""")

code("""
action = mon.check(6)["action"]
print("re-tune :", action["retune"])
print("deploy  :", action["deploy"])
print(action["note"])
""")

md("""
A drift alert **stages** a re-tuned rubric for a human. It does not deploy it. A
system that re-tunes and self-deploys is editing its own success criteria without
supervision, and it will eventually conclude that it is doing well.
""")

# --------------------------------------------------------------- 7. close
md("""
---
# What to take away

1. **A judge is not a benchmark score.** It is built, tuned, deployed and kept
   aligned, and those are four different engineering problems.
2. **Collect the reason, not just the verdict.** It costs a rater ten seconds and
   it is the difference between a judge that is right and one that is right for
   the right reason - and in Phase III that reason *is* the revision instruction.
3. **Build the cheap baseline first.** Until you have beaten the rules you could
   have written, you cannot say an LLM judge earned its bill.
4. **Price your errors.** A bad artefact served cannot be recalled; a good one
   dropped costs an opportunity. Specificity is weighted 3x here for that reason.
5. **Build Phase IV before you need it.**

### The bug worth knowing about

Running this against a real model exposed something no green test suite caught.
The judge's prompt ended *"return the failure mode that best fits, chosen from
that list"* - an instruction to find a failure - and its JSON schema emitted
`label` before `reason`, so it committed to a verdict before doing the work. It
said so itself:

> *"The explanation is 29 words long, but ... the 40-word limit **which this
> actually passes**. However, following the exact rubric instructions..."* -> FAIL

A fail-biased judge **inflates specificity**, so the bias was invisible and
flattering on every semantic criterion. It was only ever detectable on `concise`,
the one criterion whose ground truth a human can verify by counting.

Twelve bugs like that are written up in `README.md`, each pinned by a test.

### Where next

| | |
|---|---|
| the reference | [`README.md`](../README.md) |
| both arms, measured | [`docs/results.md`](../docs/results.md) |
| judge your own thing | [`docs/adding-a-domain.md`](../docs/adding-a-domain.md) |
| what breaks at scale | [`docs/production-notes.md`](../docs/production-notes.md) |

Run it live: put `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env` and use
`build_context(offline=False)`.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.relative_to(OUT.parent.parent)}  ({len(cells)} cells)")

# Execute by default, and this is not a convenience.
#
# The shipped notebook carries its outputs so that GitHub renders a readable
# artefact for someone who never runs it - which is most readers. Writing the
# cells without executing them silently replaces that with a page of empty code
# blocks, and the file still looks fine in a diff.
#
# It also means the outputs cannot drift from the code: regenerating always
# re-runs. `--no-execute` exists for a quick prose edit when the kernel is
# unavailable, and it prints a warning because the result is not shippable.
if "--no-execute" in sys.argv:
    print("WARNING: outputs are EMPTY. Re-run without --no-execute before committing.")
    raise SystemExit(0)

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:
    raise SystemExit(
        "nbclient is needed to embed outputs. Install it with:  "
        "pip install nbclient nbformat ipykernel  "
        "(or pass --no-execute to write the notebook without them)"
    )

nb = nbformat.read(OUT, as_version=4)
NotebookClient(
    nb, timeout=600, kernel_name="python3",
    resources={"metadata": {"path": str(OUT.parent)}},
).execute()
nbformat.write(nb, OUT)

errors = [
    o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"
]
if errors:
    raise SystemExit(f"notebook executed with {len(errors)} error cell(s); not shippable")
print(f"executed cleanly: {sum(1 for c in nb.cells if c.cell_type == 'code')} code cells")

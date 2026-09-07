![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LLM](https://img.shields.io/badge/LLM-Gemini-7c3aed.svg)
![Providers](https://img.shields.io/badge/providers-Gemini%20%7C%20OpenAI--compatible%20%7C%20Anthropic-0f9d58.svg)
![API](https://img.shields.io/badge/API-FastAPI-009688.svg)
![Tests](https://img.shields.io/badge/tests-192%20passing-brightgreen.svg)
![Offline](https://img.shields.io/badge/runs%20offline-no%20API%20key-informational.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Learn AI/ML interactively at [AI-ML Companion](https://aimlcompanion.ai/)** - guided walkthroughs, architecture decisions, hands-on challenges and narrated overviews for every project.

# The Lifecycle of an LLM Judge

Most teams treat an LLM judge as a **benchmark score**: built once, validated
once, quoted for a year. It is really **a service in the request path, with a
deployment, a bill and a decay curve.**

A working implementation of all four phases, from
[Kong et al. (Netflix, COLM 2026)](https://arxiv.org/abs/2608.18300).

| Start here | |
|---|---|
| **Notebook** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/projects/llm/llm-judge-lifecycle/notebooks/LLM_Judge_Lifecycle.ipynb) all four phases, outputs inline, one dependency |
| **Terminal walk** | [QUICKSTART.md](QUICKSTART.md) - twenty minutes, four commands, no API key |
| **This README** | the reference. Skim the headings, read what you need |

---

## The question this turns on

Two judges. Same explanation, shown beside a film. Both rejected it.

```
ARTEFACT   "The Quiet Harbour is a slow-burn mystery rated 104 by viewers."
RECORD     title: The Quiet Harbour        runtime_min: 104

JUDGE A    FAIL - "104 is the runtime in minutes, not a rating."
JUDGE B    FAIL - "This explanation is too short to be useful."
```

On any label-accuracy metric they are **identical**. They are not the same.

Judge B is right by accident, and two things follow. It will pass this the moment
explanations get longer, because label accuracy cannot separate a judge that
understood the criterion from one that learned a correlation. And here the
judge's reason is handed back to the writer as its next instruction, so B says
*write more* while the real error survives every retry.

> **Measuring that gap is a metric ([§4](#4-the-three-metrics)). Closing it is a
> training signal ([§5](#5-phase-ii---training)). Catching its return is a
> monitor ([§7](#7-phase-iv---monitoring)). That is the lifecycle.**

---

## Contents

| | | | |
|---|---|---|---|
| [1 Quickstart](#1-quickstart) | [2 The four phases](#2-the-four-phases) | [3 Phase I - Birth](#3-phase-i---birth) | [4 The three metrics](#4-the-three-metrics) |
| [5 Phase II - Training](#5-phase-ii---training) | [6 Phase III - Deployment](#6-phase-iii---deployment) | [7 Phase IV - Monitoring](#7-phase-iv---monitoring) | [8 Results](#8-results) |
| [9 Providers](#9-providers) | [10 Portability](#10-portability) | [11 Design decisions](#11-design-decisions) | [12 Layout](#12-project-layout) |

<details>
<summary><strong>New to the terms?</strong></summary>

**LLM-as-a-Judge** - one model scoring another model's output, because having
humans read all of it is too slow. **Artefact** - the thing being judged; here a
one-sentence explanation beside a recommended film. **Rubric** - the written
criteria, which in this project is *text you can read and edit*. **Gate** - the
judge in the request path, with power to reject. **Drift** - it was aligned in
March and is not in September, because the catalogue changed and nothing broke.
</details>

---

## 1. Quickstart

```bash
uv venv && uv pip install -r requirements.txt    # or: pip install -r requirements.txt
```

Walk all four phases. **No API key, no network, no cost:**

```bash
python run.py --offline benchmark                  # I    what is in the benchmark
python run.py --offline tune --criterion specific  # II   watch a rubric get tuned
python run.py --offline curve --max-k 4            # III  pass rate vs retry budget
python run.py --offline monitor --week 6           # IV   a drift alert, and where it hides
```

For real models, put `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env` and drop
`--offline`. Gemini is the default.

```bash
python -m pytest tests/ -q            # 192 tests, hermetic
uvicorn api.main:app --port 8000      # HTTP API, docs at /docs
streamlit run app/streamlit_app.py    # four tabs, one per phase
```

### Why there is no API key

A rubric is markdown a model reads, and some bullets carry an inline tag:

```
- Reject filler that would fit any item in the catalogue.
  [banned: "you'll love it", "a must-watch"]
- Every factual claim must be traceable to the record. [grounded]
```

![One rubric, two readers](docs/images/02-two-readers.svg)

An LLM judge ignores the brackets and reads the sentence. `src/rules.py` ignores
the sentence and reads the brackets. So `--offline` is **not a mock** - it is a
real, deterministic, lexically-weak judge. That buys three things:

- **A baseline any model judge must beat.** Most published judge results never
  report one.
- **Phase II genuinely runs.** RART edits rubric text; the rule engine reads that
  text. Same algorithm as against Gemini, different reader.
- **A hermetic test suite**, and a Colab notebook that costs nothing.

---

## 2. The four phases

![The four phases and the two loops that close them](docs/images/01-lifecycle.svg)

Two loops close it: a fast one that re-tunes when judge-human agreement decays,
and a slow one that keeps the benchmark representative so the thing you measure
against does not quietly age out.

### Architecture

```
app/  UI, over HTTP  ->  api/routes_*  thin  ->  src/services/  ->  src/  capabilities
```

Enforced by `tests/test_layering.py`, not by convention: `src/` never imports a
web framework, vendor SDKs live only in `src/providers/`, and the UI never
imports `src`. Detail in [`docs/architecture.md`](docs/architecture.md).

---

## 3. Phase I - Birth

The phase everyone wants to skip, and the one every other number rests on.

**Three sources.** Expert-written examples with known failure modes;
LLM-synthesised cases near the boundary (naturalistic sampling almost never
surfaces hard ones); and real production samples. Synthesised rows are written
with `labels: {}` and the loader **refuses them until a human labels them** - an
LLM-written example labelled by an LLM measures whether two models agree.

**One guideline, two consumers.** A criterion's `guideline` field is *both* what
human raters label against *and* the seed rubric the judge is tuned from. Write
them separately and they drift apart within a month, after which judge-human
disagreement measures a documentation gap.

> **The benchmark is class-balanced (~50/50), and that changes what the numbers
> mean.** Real defect rates are a few percent, so a naturally-sampled set is 95%
> PASS and a judge answering PASS to everything scores 95% while catching
> nothing. Balance makes specificity measurable. The cost: **these are alignment
> metrics, not defect rates.**

**Splits survive a growing benchmark.** An example's split is a hash of its own
id, so Phase IV's weekly appends do not reshuffle existing rows - otherwise last
week's training examples land in this week's test set and every week-over-week
comparison silently stops meaning anything.

---

## 4. The three metrics

```
Specificity  of the artefacts a human failed, how many did the judge fail?
Recall       of the artefacts a human passed, how many did the judge pass?
RA_neg       of the artefacts a human failed, how many did the judge fail
             FOR THE SAME REASON?
```

The third is what separates Judge A from Judge B. Note its denominator: **every
human failure, not every agreed failure.** Dividing by agreed-fails would score a
judge that catches one failure and explains it perfectly the same as one that
catches three and explains all three.

**Why not accuracy.** On a balanced set it has a coin-flip baseline, and it lets
the two error types cancel out. They are not the same error:

| | cost |
|---|---|
| a bad artefact the judge **passes** | reaches a user, damages trust, cannot be recalled |
| a good artefact the judge **fails** | regenerated, or dropped |

So the objective prices them: `s = 3*Specificity + 1*Recall + 1*RA_neg`.

**Every metric ships with its interval.** Test splits hold five to seven
examples; a 95% Wilson interval on 5/6 runs [0.44, 0.97]. A point estimate over
six examples invites a conclusion it cannot support.

---

## 5. Phase II - Training

**No gradients. No fine-tuning. No weights are touched anywhere in this repo.**
The rubric *text* is the parameter and a reflector LLM is the optimiser.

```
R* <- R_0
for t in 0..N-1:
    score D_train with J(R_t)
    s <- weighted metrics on D_val          <- VALIDATION, never train
    if s > s*:  R* <- R_t                   <- keep the BEST, not the last
    focus <- {label mismatches} u {agreed-fails with the WRONG REASON}
    R_{t+1} <- Reflect(R_t, focus)
```

Three details, each load-bearing: early stopping is on **validation** (the
reflector has seen every training error by construction); the **best** checkpoint
is returned, not the last (iteration 4 is regularly worse than 2); and the test
split is **never touched** here.

### What the optimiser wrote

Offline, the lexical reflector can do exactly one thing:

```diff
- [banned: "you'll love it", "a must-watch", "highly rated"]
+ [banned: "you'll love it", "a must-watch", "highly rated", "perfect for anyone"]
```

Live, the Pro reflector edited **both halves of the same rubric**:

```diff
+FAIL when the recommendation is addressed to a generic audience (e.g.,
+"anyone who...") rather than the specific viewer, even if it lists
+specific attributes of the title.

+ [banned: ..., "perfect for anyone"]
```

A prose clause a model can apply to paraphrases it has never seen, **and** the
tag the rule engine reads. On `grounded` it fixed a recall failure by tightening
"asserts *anything*" to "asserts any *factual detail*", using boundary examples
it had never been shown. It generalised rather than memorised.

---

## 6. Phase III - Deployment

![The judge as gate and critic](docs/images/03-gate-critic.svg)

As **gate** it rejects. As **critic** its rejection reason steers the next draft,
which is why a right-verdict-wrong-reason rejection is a real defect: it points
the rewrite at the wrong problem, the budget is spent, and the artefact is
dropped anyway.

> **When the budget runs out the artefact is DROPPED**, not served with a
> warning. It is one `if`, and it is the whole design.

### Read the curve before choosing K

| k | offline (stub generator) | live (Gemini) |
|---|---|---|
| 0 | 0.200 | **0.950** |
| 1 | 0.550 | **1.000** |
| 2 to 4 | 0.750 to 0.950 | 1.000 |

Both monotone then flat; everything else differs. **Live, K=3 is
over-provisioned** - a real generator clears the gate unaided 95% of the time.
You cannot know which regime you are in without measuring, and you must never
pick K from a curve somebody else measured.

- **k=0 is the generator's unaided rate.** A drop there is a generator
  regression, not judge drift, and an aggregate pass rate cannot tell them apart.
- **Flat AND low** means revision cannot rescue the writer. Fix the writer.

---

## 7. Phase IV - Monitoring

The phase that is always deferred, and the only one that tells you the other
three have stopped working.

### The threshold floats

```
judge  >=  mean(rater_scores)  -  2 * sd(rater_scores)
```

![The band widens when raters disagree](docs/images/04-floating-band.svg)

Not 0.85, because **raters do not agree with each other by a constant amount.**
On an ambiguous week `sd` widens and the band widens with it, so the judge is not
punished for finding hard what people also found hard. A fixed threshold fires
every hard week until somebody mutes it, then sleeps through a slow slide.

### And the half the band cannot see

![Where drift hides](docs/images/05-drift-hides.svg)

```
overall    specificity  judge=0.786  raters=0.857+-0.117  floor=0.624  in band
new items  specificity  judge=0.250  raters=0.917+-0.118  floor=0.681  OUT OF BAND
```

Ten of week 6's fourteen failures came from the established catalogue where the
judge is still good, so the aggregate looks healthy. The judge is wrong about most
of what is *newly* recommended and will keep looking fine for as long as new
titles are a minority of traffic. `check_new_items_separately: true` is one line
of config and the only reason this is visible.

**An alert stages a re-tuned rubric for a human. It does not deploy it.**

---

## 8. Results

Full numbers, commands and caveats: **[`docs/results.md`](docs/results.md)**.

Seed rubric (the human guideline, untuned) on validation:

| criterion | rule engine | **Gemini 3.5 Flash** |
|---|---|---|
| `grounded` | spec 0.667 · ra 0.667 | **1.000 · 1.000** |
| `specific` | spec 0.167 · ra 0.167 | **0.667 · 0.667** |
| `safe` | spec 0.500 · ra 0.500 | **1.000 · 1.000** |

The offline run named the cases inside that gap - spoilers in unlisted words, a
real number on the wrong noun, contradiction rather than invention - and the
model judge closes most of it. **That is the argument for paying for one,
measured rather than asserted.**

After RART: `grounded` recall 0.500 to 1.000, `specific` specificity 0.667 to
1.000, `safe` already at ceiling. On held-out test all three reach 1.000 - **over
four and three examples, with intervals of [0.51, 1.00] and [0.44, 1.00].** The
honest claim is "indistinguishable from perfect on seven examples".

> **A null result diagnosed two ways.** `safe` did not improve in either run.
> Offline the CLI says *specificity is only 0.333, there is headroom this
> optimiser could not reach*; live it says *specificity is already 1.000, the
> guideline was at ceiling*. Same code path, opposite meaning.

`docs/results.md` also records the bugs found while building this, all of which
produced plausible numbers rather than errors - including a judge biased toward
FAIL by its prompt and its schema field order, a bias that **inflates
specificity** and was only ever detectable on the one criterion with a
mechanically checkable answer.

---

## 9. Providers

Four roles, four independent `provider` and `model` settings:

```yaml
generator:  {model: gemini-3.5-flash, thinking_budget: 0}   # writes the artefact
judge:      {model: gemini-3.5-flash, thinking_budget: 0}   # grades it
reflector:  {model: gemini-pro-latest}                      # rewrites rubrics
meta_judge: {model: gemini-3.5-flash, thinking_budget: 0}   # compares reasons
```

Also shipped: **`openai_compatible`** (covers OpenAI, vLLM, Ollama and Together
via `base_url`) and **`anthropic`**.

> **If the generator and judge are the same model, you cannot tell "this output
> is good" from "this output is written the way I write."** Self-preference bias
> is a documented failure mode; a single-model config makes it invisible, not
> absent. Every artefact produced that way is stamped
> `single_model_config: true`.

**One gotcha worth stealing.** On thinking models, reasoning tokens come out of
the same `max_output_tokens` budget as the response. Measured here, the generator
spent 1,746 thinking tokens on a 40-token sentence, and raising the cap only
bought more thinking. The adapter switches thinking off for structured verdicts
and counts thinking tokens as billed output.

---

## 10. Portability

`domains/support/` is a second domain with entirely different criteria. It needed
**zero changes to `src/`**, enforced by
`tests/test_domain.py::test_no_domain_specific_logic_in_src`.

```yaml
domain: support     # in configs/base.yaml
```

**The mechanism ports. The rubrics do not, and should not.** The two domains
share exactly one criterion id (`concise`, a length check). What *does* transfer
is the shape of the failures: a real number on the wrong noun, a claim that
contradicts the record, a polite sentence that says nothing. Guide:
[`docs/adding-a-domain.md`](docs/adding-a-domain.md).

---

## 11. Design decisions

| Decision | Why | What the alternative costs |
|---|---|---|
| One judge **per criterion** | RART needs an isolated parameter | A combined judge cannot be tuned without risking every criterion |
| Rubric passed **in**, never baked into the prompt | Metric movement is attributable | A hardcoded prompt can only be rewritten |
| **Guideline IS the seed rubric** | Rater guidance and judge prompt cannot drift | Two documents that agree in month one |
| Judge **fails closed** | A guardrail that fails open stops working when it breaks | Defaulting to PASS ships everything, invisibly |
| Meta-judge **fails towards agreement** | Parser noise must not drive the optimiser | Rubrics rewritten to fix disagreements never observed |
| **Drop** on budget exhaustion | Bad output cannot be recalled | `serve_best` raises coverage until the first spoiled ending |
| Splits **content-addressed** | Comparability survives weekly appends | A shuffle silently corrupts week-over-week |
| Tuning **stages**, never deploys | A rubric change changes what reaches users | An unsupervised system editing its own success criteria |
| **Wilson** intervals, always printed | The normal approximation claims certainty at 8/8 | A point estimate over six examples |
| No LangChain, no eval framework | The lifecycle is the teaching goal | A framework hides the rubric, focus set and objective |

---

## 12. Project layout

```
├── run.py                 CLI for all four phases (--offline runs free)
├── configs/base.yaml      every knob, heavily commented. Read this first.
├── domains/
│   ├── recommendation/    reference domain: catalogue, labels, weekly HITL samples
│   └── support/           second domain, proving zero-code portability
├── src/
│   ├── benchmark.py       I    splits, balance, synthesis
│   ├── rart.py            II   Algorithm 1        meta_judge.py  right verdict, wrong reason
│   ├── serving.py         III  gate + critic      judge.py       one per criterion, fails closed
│   ├── monitoring.py      IV   the floating band  metrics.py     Spec / Recall / RA_neg + Wilson
│   ├── rules.py           the offline rule engine / baseline
│   ├── providers/         gemini | openai_compatible | anthropic | stub
│   └── services/          orchestration; the only layer api/ and run.py call
├── api/  app/             FastAPI (one router per phase) and Streamlit (four tabs)
├── notebooks/  scripts/   Colab notebook, build_diagrams.py, build_notebook.py
├── tests/                 192 tests, hermetic
└── docs/                  architecture · results · adding-a-domain · production-notes
```

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `configured for gemini but GOOGLE_API_KEY is not set` | Working as intended. Add the key, or add `--offline`. |
| `the test split holds only 2 FAIL example(s)` | Not an error. The benchmark is small and the metric is near-meaningless there. |
| `estimated spend $X reached the cap` | Fired *before* the call. Raise `MAX_RUN_USD`, or ask why the loop is not terminating. |
| `role 'judge' hit its output cap` | Raise `judge.max_output_tokens`. On a thinking model, reasoning shares that budget ([§9](#9-providers)). |
| Tuning reports "no improvement" | Read the specificity beside it. High means at ceiling, low means headroom it missed. Both are results. |

---

## Licence

MIT. The catalogue, the support tickets and every artefact here are invented;
nothing describes a real film, company or customer.

**Paper:** Kong, Tan, Gupta, Fagnan, Olds, Campbell, Kavuri, Balin, Gosain,
Garcia & Jang, *The Lifecycle of LLM-as-a-Judge for Large-Scale Recommendation
Explanations*, Netflix, COLM 2026 workshops.
[arXiv:2608.18300](https://arxiv.org/abs/2608.18300)

An independent implementation on invented data. Not affiliated with or endorsed
by the authors or Netflix, and the numbers here are this repo's, not the paper's.

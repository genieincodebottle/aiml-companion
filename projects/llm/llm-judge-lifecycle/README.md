# The Lifecycle of an LLM Judge

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LLM](https://img.shields.io/badge/LLM-Gemini-7c3aed.svg)
![Tests](https://img.shields.io/badge/tests-192%20passing-brightgreen.svg)
![Offline](https://img.shields.io/badge/runs%20offline-no%20API%20key-informational.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A runnable companion to the post
[Implementing the LLM judge lifecycle from the Netflix paper](https://aimlcompanion.ai/blog/implementing-the-llm-judge-lifecycle-2026).

Most teams treat an LLM judge as a **benchmark score**: built once, validated
once, quoted for a year. It is really **a service in the request path, with a
deployment, a bill and a decay curve.** All four phases from
[Kong et al. (Netflix, COLM 2026)](https://arxiv.org/abs/2608.18300), implemented
on invented data.

| Start here | |
|---|---|
| **Read** | [the blog walkthrough](https://aimlcompanion.ai/blog/implementing-the-llm-judge-lifecycle-2026) - Why each phase is there, and why those three numbers weren't really results. |
| **Run** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/projects/llm/llm-judge-lifecycle/notebooks/LLM_Judge_Lifecycle.ipynb) all four phases, outputs inline, one dependency |
| **Terminal** | [QUICKSTART.md](QUICKSTART.md) - twenty minutes, four commands, no API key |
| **Verify** | [`docs/results.md`](docs/results.md) - every figure with its command and its caveat |

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

## Quickstart

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

The judge ignores the brackets and reads the sentence. `src/rules.py` ignores the
sentence and reads the brackets. So `--offline` is **not a mock** - it is a real,
deterministic, lexically-weak judge, which buys a baseline the paid judge has to
beat, a Phase II that genuinely runs, and a hermetic test suite.

---

## The four phases

![The four phases and the two loops that close them](docs/images/01-lifecycle.svg)

| Phase | Command | Produces | The line that matters |
|---|---|---|---|
| **I Birth** | `benchmark` | the answer key, 85 labelled examples | a split is a hash of the example id, so weekly appends never reshuffle last week's rows |
| **II Training** | `tune` | a tuned rubric | the rubric *text* is the parameter and a reflector LLM is the optimiser; no weights move |
| **III Deployment** | `serve` | a gate | when the retry budget runs out, **drop** - never serve the best of a bad set |
| **IV Monitoring** | `monitor` | a drift signal | `judge >= mean(raters) - 2*sd(raters)`, so the bar floats; and new items are checked separately |

Two loops close it: a fast one that re-tunes when judge-human agreement decays,
and a slow one that keeps the benchmark representative.

Three metrics, because accuracy has a coin-flip baseline on a balanced set and
lets the two error types cancel:

```
Specificity  of the artefacts a human failed, how many did the judge fail?
Recall       of the artefacts a human passed, how many did the judge pass?
RA_neg       of the artefacts a human failed, how many did the judge fail
             FOR THE SAME REASON?          objective: s = 3*Spec + Recall + RA_neg
```

`RA_neg` is the one that separates a judge that understood the criterion from one
that was right by accident, and it is not academic: the judge's reason is handed
back to the writer as its next instruction, so a wrong reason survives every
retry. **Every metric ships with a Wilson interval** - test splits hold seven
examples at most, and 5/6 runs [0.44, 0.97].

> Why each phase is shaped this way, and what broke while building it, is the
> [blog post](https://aimlcompanion.ai/blog/implementing-the-llm-judge-lifecycle-2026).
> The commands, tables and caveats behind every number are
> [`docs/results.md`](docs/results.md). Layering rules are
> [`docs/architecture.md`](docs/architecture.md).

---

## Results

Seed rubric (the human guideline, untuned) on validation:

| criterion | rule engine | **Gemini 3.5 Flash** |
|---|---|---|
| `grounded` | spec 0.667 · ra 0.667 | **1.000 · 1.000** |
| `specific` | spec 0.167 · ra 0.167 | **0.667 · 0.667** |
| `safe` | spec 0.500 · ra 0.500 | **1.000 · 1.000** |

The model judge closes most of that gap, and **the free rule engine is what makes
it measurable.** After RART all three reach 1.000 on held-out test - over four and
three examples, so the honest claim is "indistinguishable from perfect on seven
examples".

Drift, week 6, where the aggregate hides it:

```
overall    specificity  judge=0.786  raters=0.857+-0.117  floor=0.624  in band
new items  specificity  judge=0.250  raters=0.917+-0.118  floor=0.681  OUT OF BAND
```

`check_new_items_separately: true` is one line of config and the only reason that
second row exists. **An alert stages a re-tuned rubric for a human. It does not
deploy it.**

---

## Configuration

Four roles, four independent `provider` and `model` settings in `configs/base.yaml`:

```yaml
generator:  {model: gemini-3.5-flash, thinking_budget: 0}   # writes the artefact
judge:      {model: gemini-3.5-flash, thinking_budget: 0}   # grades it
reflector:  {model: gemini-pro-latest}                      # rewrites rubrics
meta_judge: {model: gemini-3.5-flash, thinking_budget: 0}   # compares reasons
```

Also shipped: **`openai_compatible`** (OpenAI, vLLM, Ollama, Together via
`base_url`) and **`anthropic`**.

- **Do not point the generator and judge at the same model.** You cannot then tell
  "this output is good" from "this output is written the way I write". Artefacts
  produced that way are stamped `single_model_config: true`.
- **Reasoning tokens share `max_output_tokens` with the response.** Measured here,
  the generator spent 1,746 thinking tokens on a 40-token sentence, and raising
  the cap only bought more thinking. The adapter switches thinking off for
  structured verdicts and counts thinking tokens as billed output.
- **A second domain costs data and prose, nothing else.** `domains/support/` needed
  zero changes to `src/`, enforced by `tests/test_domain.py`. The mechanism ports;
  the rubrics do not. Guide: [`docs/adding-a-domain.md`](docs/adding-a-domain.md).

---

## Design decisions

| Decision | Why | What the alternative costs |
|---|---|---|
| One judge **per criterion** | RART needs an isolated parameter | A combined judge cannot be tuned without risking every criterion |
| **Guideline IS the seed rubric** | Rater guidance and judge prompt cannot drift | Two documents that agree in month one |
| Judge **fails closed** | A guardrail that fails open stops working when it breaks | Defaulting to PASS ships everything, invisibly |
| **Drop** on budget exhaustion | Bad output cannot be recalled | `serve_best` raises coverage until the first spoiled ending |
| Splits **content-addressed** | Comparability survives weekly appends | A shuffle silently corrupts week-over-week |
| Tuning **stages**, never deploys | A rubric change changes what reaches users | An unsupervised system editing its own success criteria |
| No LangChain, no eval framework | The lifecycle is the teaching goal | A framework hides the rubric, focus set and objective |

---

## Project layout

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
| `role 'judge' hit its output cap` | Raise `judge.max_output_tokens`. On a thinking model, reasoning shares that budget. |
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

> **Learn AI/ML interactively at [AI-ML Companion](https://aimlcompanion.ai/)** - guided walkthroughs, architecture decisions, hands-on challenges and narrated overviews for every project.

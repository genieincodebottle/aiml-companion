# Quickstart

Twenty minutes, four commands, **no API key**. The [README](README.md) is the
reference; this is the walk.

```bash
uv venv && uv pip install -r requirements.txt
```

That is the whole setup. No Docker, no database, no key.
(Plain pip works too: `pip install -r requirements.txt`.)

---

## Why there is no API key

Because `--offline` is not a mock.

A rubric in this project is markdown a model reads, and some bullets carry an
inline tag:

```
- Reject filler that would fit any item in the catalogue.
  [banned: "you'll love it", "a must-watch"]
- Every factual claim must be traceable to the record. [grounded]
```

An LLM judge ignores the brackets and reads the sentence. `src/rules.py` ignores
the sentence and reads the brackets. **Same rubric, two readers.**

So with `--offline` you get a real judge - a deterministic, lexical, weak-in-
documented-ways one. That is worth more than a canned demo, because it is the
**baseline any model judge has to beat**, and most published judge results never
report one.

What you give up without a key: the model-vs-rules gap has to be read from
[`docs/results.md`](docs/results.md) rather than reproduced, and the Phase II
optimiser can only learn lexical rules. Everything else is the real system.

---

## 1. What is in the benchmark

```bash
python run.py --offline benchmark
```

```
grounded   gate  n=24   fail=0.542  splits={'train': 12, 'validation': 5, 'test': 7}
specific   gate  n=30   fail=0.567  splits={'train': 18, 'validation': 7, 'test': 5}
safe       gate  n=25   fail=0.520  splits={'train': 15, 'validation': 4, 'test': 6}
concise    soft  n=6    fail=0.500  splits={'train': 3,  'validation': 2, 'test': 1}
```

You will also see warnings about thin splits. **Those are the point, not noise.**
The benchmark is deliberately small so the confidence intervals stay visibly
wide; a five-example test split cannot support a conclusion and the tool says so
rather than printing three decimal places and hoping.

Note `fail≈0.5` on every gate criterion. Real defect rates are a few percent, so
a naturally-sampled benchmark would be ~95% PASS and a judge answering PASS to
everything would score 95% while catching nothing. Balance is what makes
"did it catch the bad ones?" measurable - and it means **these numbers are not
defect rates**.

> Open `domains/recommendation/labels.jsonl`. Every FAIL carries a written
> rationale, and cases marked `BASELINE-MISS` are ones the rule engine provably
> cannot catch. They are the headroom a model judge has to earn.

---

## 2. Watch a rubric get tuned

```bash
python run.py --offline tune --criterion specific
```

```
iter  weighted  spec    rec     ra      focus  kept
0     1.667     0.167   1.000   0.167   5      *
1     3.000     0.500   1.000   0.500   2      *
```

No gradients. No fine-tuning. **The rubric text is the parameter** and a
reflector model is the optimiser: score the rubric, collect the mistakes, ask for
a better rubric, keep it only if validation improves.

Now read what it wrote:

```bash
diff <(python -c "from src.domain import get_domain; print(get_domain('recommendation').seed_rubric('specific'))") \
     artifacts/rubrics/recommendation/staged/specific.md
```

```diff
- [banned: "you'll love it", "a must-watch", "highly rated"]
+ [banned: "you'll love it", "a must-watch", "highly rated", "perfect for anyone"]
```

One clause, and you can read it. That legibility is the argument for tuning the
rubric instead of the weights.

Try `--criterion safe` too. It does **not** improve, and the tool tells you
whether that means "the guideline was already at ceiling" or "there is headroom
and this optimiser could not reach it". Those are opposite findings and most
tools report both as "no change".

---

## 3. Serve, and see what gets dropped

```bash
python run.py --offline curve --max-k 4
```

```
k=0   0.200  ########
k=1   0.550  ######################
k=2   0.750  ##############################
k=3   0.850  ##################################
k=4   0.950  ######################################
```

Cumulative pass rate against the retry budget. The judge plays two roles at once:
it **rejects** (gate), and its rejection reason becomes the writer's instruction
for the next attempt (critic).

When the budget runs out the artefact is **dropped**, not served. That asymmetry
is one `if` and it is the whole design:

> a bad artefact served reaches a user and cannot be recalled
> a good one dropped costs one missed opportunity

This curve is what turns `max_retries` from a guess into a decision. Run it
against a real model and it looks completely different - see
[`docs/results.md`](docs/results.md).

---

## 4. The bit most teams never build

```bash
python run.py --offline monitor --week 5    # a healthy week
python run.py --offline monitor --week 6    # drift
```

Week 6:

```
overall    specificity  judge=0.786  raters=0.857±0.117  floor=0.624  in band
new items  specificity  judge=0.250  raters=0.917±0.118  floor=0.681  OUT OF BAND
```

Two ideas, and they are the best things in the project.

**The threshold floats.** It is `mean(raters) - 2 * sd(raters)`, not 0.85. On a
week where raters disagreed, sd widens and so does the band - the judge is not
punished for finding hard what people also found hard. A fixed threshold fires
every hard week until somebody mutes it, then sleeps through a slow slide.

**The aggregate hides it.** Ten of week 6's fourteen failures came from the
established catalogue where the judge is still good, so overall looks healthy.
The judge is wrong about most of what is *newly* being recommended, and it will
keep looking fine for as long as new titles are a minority of traffic.
`check_new_items_separately: true` is one line of config and the only reason this
is visible.

---

## Then what

| If you want to | Go to |
|---|---|
| understand any of it properly | [README](README.md) - it is the reference, read the section you need |
| see it against a real model | [`docs/results.md`](docs/results.md) - both arms, side by side |
| judge your own thing | [`docs/adding-a-domain.md`](docs/adding-a-domain.md) - data and prose, no code |
| know what breaks at scale | [`docs/production-notes.md`](docs/production-notes.md) |
| run it live | put `GOOGLE_API_KEY` or `GEMINI_API_KEY` in `.env`, drop `--offline` |

```bash
python -m pytest tests/ -q                    # 192 tests, no network
uvicorn api.main:app --port 8000              # HTTP API, docs at /docs
streamlit run app/streamlit_app.py            # four tabs, one per phase
```

---

## Is this for you?

**Yes**, if you have written "we'll use an LLM to evaluate the output" in a
design doc. This is what that commits you to.

**Probably not yet**, if you have not yet built something an LLM evaluates. The
problem only gets interesting once you have the ache, and no amount of low setup
cost substitutes for that. Build [RAG Expert Assistant](../rag-expert-assistant/)
or [GraphRAG Supply Chain](../graphrag-supply-chain/) first, then come back when
you need to know whether their output is any good.

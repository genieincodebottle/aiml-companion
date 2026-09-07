# Measured results

Two runs of the same pipeline on the same benchmark, on the reference domain
(`recommendation`):

```bash
python run.py --offline <command>    # rule engine. Free. The baseline.
python run.py <command>              # Gemini. The system under test.
```

Keeping both is the point. The offline run is not a fallback - it is the
**control arm**, and without it "our LLM judge scores 1.000" is a number with
nothing to compare against.

| | offline | live |
|---|---|---|
| judge | `src/rules.py` rule engine | `gemini-3.5-flash` |
| reflector | lexical phrase miner | `gemini-pro-latest` |
| cost | $0.00 | ~$0.12 for everything below |

## Read this before quoting anything

**The benchmark is class-balanced and difficulty-enriched.** Roughly 50/50 per
criterion, against a real-world defect rate of a few percent. Nothing here
estimates a live defect rate.

**The splits are tiny.** Five to seven examples per test split. A 95% Wilson
interval on 5/6 runs [0.44, 0.97]. Treat any gap narrower than the interval as
noise - including the ones below that look decisive.

**The generator and judge are the same model** in the live run
(`single_model_config: true`, and every artefact says so). Self-preference bias
is uncontrolled. See §Providers in the README for the cheap fix.

---

## Phase I - what is in the benchmark

```
python run.py --offline benchmark
```

| criterion | kind | n | fail fraction | train / val / test |
|---|---|---|---|---|
| `grounded` | gate | 24 | 0.500 | 12 / 5 / 7 |
| `specific` | gate | 30 | 0.567 | 18 / 7 / 5 |
| `safe` | gate | 25 | 0.520 | 15 / 4 / 6 |
| `concise` | soft | 6 | 0.500 | 3 / 2 / 1 |

`concise` is deliberately unbalanced and tiny. It is a soft criterion that is
right by construction and only ever reported, so balancing it would be ceremony.

---

## Phase II - RART

All numbers below were re-measured after the judge-prompt fix described at the
bottom of this file, and after correcting one mislabelled benchmark row that the
live judge caught. Earlier drafts of this document reported different figures;
they were measured under a false-FAIL bias and are not reproducible.

### The headline: what the model judge buys you

Validation, seed rubric (the human labelling guideline, untuned):

| criterion | rule engine | **Gemini** |
|---|---|---|
| `grounded` | spec 0.667 · rec 0.500 · ra 0.667 | **1.000** · 0.500 · **1.000** |
| `specific` | spec 0.167 · rec 1.000 · ra 0.167 | **0.667** · 1.000 · **0.667** |
| `safe` | spec 0.500 · rec 1.000 · ra 0.500 | **1.000** · 1.000 · **1.000** |

The offline run predicted this gap and named the cases inside it: spoilers
phrased in words nobody listed, a real number attached to the wrong noun, a claim
that contradicts the record rather than adding to it. **The model judge closes
most of it.** That is the case for paying for one, measured rather than asserted
- and the free rule engine is what makes it measurable.

### Tuning

| criterion | offline: seed → best | live: seed → best | stopped because |
|---|---|---|---|
| `grounded` | 0.667 → 0.667 (no change) | rec **0.500 → 1.000**, all metrics 1.000 | targets cleared |
| `specific` | spec **0.167 → 0.500** | spec **0.667 → 1.000**, ra **0.667 → 1.000** | targets cleared |
| `safe` | 0.500 → 0.500 (no change) | 1.000, unchanged | targets cleared |

Live, all three stopped because **every metric cleared its target**, not because
they ran out of iterations.

### The same null result, diagnosed two opposite ways

`safe` did not improve in either run, and the CLI says something different about
each - correctly:

> **offline:** *specificity is only 0.333. This is NOT a criterion at ceiling -
> there is plenty of headroom and this optimiser could not reach it.*

> **live:** *specificity is already 1.000. The human guideline was at ceiling for
> this criterion and there was nothing for the optimiser to find. That is a
> RESULT. Report it; do not quietly drop the criterion from the table.*

Same code path, opposite diagnosis. "No improvement" means two completely
different things and reporting both with one sentence is how a broken optimiser
gets read as a validated one.

### What the reflectors actually wrote

This is the most instructive artefact in the project, because you can read the
optimiser's output as English.

**Offline, `specific`** - the lexical miner can only do one thing, and it does it:

```diff
- [banned: "you'll love it", "a must-watch", "highly rated"]
+ [banned: "you'll love it", "a must-watch", "highly rated", "perfect for anyone"]
```

**Live, `specific`** - the Pro reflector edited **both halves of the rubric**:

```diff
+FAIL when the recommendation is addressed to a generic audience (e.g.,
+"anyone who...") rather than the specific viewer, even if it lists
+specific attributes of the title.

- [banned: "you'll love it", "a must-watch", "highly rated"]
+ [banned: "you'll love it", "a must-watch", "highly rated", "perfect for anyone"]
```

A **general prose clause** a model judge can apply to paraphrases it has never
seen, *and* the **lexical tag** the offline rule engine reads. The stub reflector
could only ever produce the second line. The first is the half that catches what
a phrase list structurally cannot.

A real reflector improves both readers at once, so the tuned rubric is strictly
better for the free baseline as well. "One rubric, two readers" therefore earns
its keep online, not only in the offline run it was built for.

**Live, `grounded`** - the interesting one, because it fixed a *recall* failure:

```diff
-FAIL when the explanation asserts anything the record does not contain.
+FAIL when the explanation asserts any factual detail the record does not contain.

 "an award-winning thriller" when the record lists no award is a
-FAIL, even if it happens to be true of some real film.
+FAIL, even if it happens to be true of some real film. Subjective viewing
+advice or opinions (e.g., "best enjoyed on a large screen", "perfect for a
+rainy day") are not factual claims and do not require grounding; they are a PASS.
```

Note what it did **not** do. The training failures it was shown were "...lands
best on a **Sunday** evening" and "...best on a big **TV**". Neither phrase
appears in the fix. It diagnosed the root cause ("asserts *anything*" was too
broad), tightened that clause, and wrote a boundary case with *different*
examples. It generalised rather than memorised, which is what the reflector
prompt asks for and the thing that usually fails.

### Held-out test: did the tuning generalise?

The test split is scored once, at the end. RART never sees it.

| criterion | seed rubric | tuned rubric |
|---|---|---|
| `grounded` | spec 1.000 · **rec 0.667** | spec 1.000 · **rec 1.000** |
| `specific` | 1.000 · 1.000 | 1.000 · 1.000 |
| `safe` | 1.000 · 1.000 | 1.000 · 1.000 |

`grounded` improved on data the optimiser never saw, so the rubric edit was not
fitting the validation split.

**And now read the intervals.** That 1.000 is computed over four FAIL examples and
three PASS examples, giving 95% intervals of [0.51, 1.00] and [0.44, 1.00]. The
tuned judge is *not* demonstrably perfect; it is "indistinguishable from perfect
on seven examples", which is a much weaker and much more honest claim. This is
the point of printing the interval next to every number.

### The ablation, and a bug worth repeating

```
python run.py --offline ablation --criterion specific
```

```
identical_rubrics: True
rart     spec=0.500  rec=1.000  ra=0.500  weighted=3.000
vanilla  spec=0.500  rec=1.000  ra=0.500  weighted=3.000
delta_weighted: 0.0
```

**The first version of this comparison reported RART winning by 0.67 on rubrics
that were byte-identical.**

Vanilla tunes without the reasoning meta-judge, so during its own loop reasoning
agreement is never computed and enters its weighted score as zero. Comparing the
two arms' internal scores handed RART a free point on a term the other arm never
measured - a gap that appears even when both produce the same text, which here
they did. `ablation()` now re-scores both with identical instrumentation. Pinned
by `tests/test_rart.py::TestTheAblationIsComparable`.

Offline, reasoning alignment changes nothing on this criterion because the
lexical reflector reaches the same rubric either way. Testing it properly needs a
model reflector and several seeds.

---

## Phase III - the retry budget

```
python run.py --offline curve --max-k 6      # rule engine + stub generator
python run.py curve --max-k 3                # Gemini, tuned rubrics
```

| k | offline (stub generator) | **live (Gemini)** |
|---|---|---|
| 0 | 0.200 | **0.950** (19/20) |
| 1 | 0.550 | **1.000** (20/20) |
| 2 | 0.750 | 1.000 |
| 3 | 0.850 | 1.000 |
| 4 | 0.950 | 1.000 |

Both curves are monotone and then flat, which is the property that makes the
plot readable at all. Everything else about them is different, and the
difference is the most useful thing in this section.

### The live curve says K=3 is over-provisioned

A real generator writing from a well-specified record clears the gate on the
first attempt 95% of the time. One title needed one revision. Nothing needed
two. **The retry budget the config ships with is almost entirely idle here**, and
on this catalogue `max_retries: 1` would buy the same pass rate for a third of
the worst-case cost.

That is a finding about *this generator on this catalogue*, not a
recommendation. The reason to run the curve is that you cannot know which
regime you are in without it:

- **Offline** the curve rises steeply and flattens late - the classic shape, and
  the one that justifies a retry budget. That is what a weak generator looks
  like, and the stub is weak by construction.
- **Live** it is flat from k=1 - the retry loop is insurance, not a workhorse.

Ship K=3 anyway if you like; just know you are paying for the tail rather than
the mean. What you must not do is pick K from a curve someone else measured.

### The honest caveat

A 0.95 unaided pass rate also means **this catalogue is too easy to exercise the
revision loop properly.** Twenty records, all well-formed, all with clean
attribute overlaps. The offline run stresses the loop and the live run barely
touches it, so the live numbers say little about how well revision *recovers*
from a bad draft.

If you want to see the critic actually working, the way to do it is to weaken the
generator deliberately - drop `generator.model` to a smaller model, or raise
`temperature` - and re-run. That is a better experiment than trusting either
curve here.

### Cost

The full k=4 sweep over 20 records: **105 calls, $0.034**.

| role | calls | USD |
|---|---|---|
| generator | 21 | 0.0044 |
| judge | 84 | 0.0291 |

The judge is **87% of the bill**, because it runs once per criterion per attempt
while the generator runs once per attempt. Four criteria means four judge calls
for every draft. That ratio, not the retry budget, is the first thing to attack
if this gets expensive - and the obvious lever is the one this project leaves out
on purpose: judge calls are cacheable on `(rubric, record, artefact)` at
temperature 0.


---

## Phase IV - the drift band

Phase IV makes no model calls: it compares recorded judge verdicts against human
ratings, so the numbers are identical offline and live.

```
python run.py --offline monitor --week 5     # healthy
python run.py --offline monitor --week 6     # drift
```

### Week 5 - healthy

| scope | metric | judge | raters | floor | verdict |
|---|---|---|---|---|---|
| overall | specificity | 0.875 | 0.875 ± 0.102 | 0.671 | in band |
| overall | recall | 1.000 | 0.944 ± 0.079 | 0.787 | in band |
| new items | specificity | 1.000 | 1.000 ± 0.000 | **1.000** | in band |
| new items | recall | 1.000 | 1.000 ± 0.000 | **1.000** | in band |

The judge missed a spoiler this week (`hitl-w5-06`) and the band absorbed it. A
monitor that fires on one miss is a monitor somebody mutes by week three.

Note the new-item rows: the raters were unanimous, so `sd = 0` and the floor
equals the rater mean. **A week nobody found ambiguous gives the judge no slack
at all.** That is the band behaving correctly.

### Week 6 - drift, visible in exactly one place

| scope | metric | judge | raters | floor | verdict |
|---|---|---|---|---|---|
| overall | specificity | 0.786 | 0.857 ± 0.117 | 0.624 | in band |
| overall | recall | 1.000 | 0.917 ± 0.118 | 0.681 | in band |
| **new items** | **specificity** | **0.250** | 0.917 ± 0.118 | 0.681 | **OUT OF BAND** |
| new items | recall | 1.000 | 0.889 ± 0.157 | 0.575 | in band |

Four titles carrying difficult subject matter arrived in the catalogue. The
generator started using that material as a hook. The judge's `safe` rubric was
tuned against a catalogue in which none of it existed.

**The aggregate is healthy** - ten of the week's fourteen failures came from the
established catalogue where the judge is still good, and ten correct verdicts
outweigh three wrong ones. A monitor checking only the aggregate would have filed
week 6 as a normal week.

The judge is now wrong about most of what the service is *newly* recommending. It
will stay wrong. The aggregate will keep looking fine for as long as new titles
are a minority of traffic, and by the time it moves, the judge has been passing
unsafe explanations for a month.

That gap is the entire argument for `check_new_items_separately: true`, and it is
one line of config.

### A fixed threshold would not have helped in either direction

- At **0.85** it fires on the healthy aggregate too (0.786 < 0.85) and tells you
  nothing about *where* the problem is.
- At **0.70** it passes the aggregate and, being one global number, says nothing
  about the new-item collapse either.

The band works because it is computed *per slice*, from the raters who actually
labelled that slice.

---

## Portability: the second domain

```
sed -i 's/^domain: recommendation/domain: support/' configs/base.yaml
python run.py --offline benchmark
```

`domains/support/` adds customer-support reply drafts with entirely different
criteria - `supported`, `actionable`, `no_overpromise`. It required **zero
changes to `src/`**, and
`tests/test_domain.py::test_no_domain_specific_logic_in_src` fails the build if a
domain name is ever hardcoded in the engine.

The mechanism ports. The rubrics do not, and should not: a spoiler is meaningless
in support, and an unauthorised refund commitment is meaningless in a film
catalogue. A team reusing another team's rubrics is porting the wrong half.

---

## Bugs this project found in itself

Kept because each produced *plausible numbers* rather than an error, which is the
only kind worth documenting. The first five were found offline. **The last seven
were only reachable by running against a real model**, which is the argument for
doing a live pass rather than trusting a green test suite - the offline suite
was green throughout.

Three of them are worth reading together, because they compound. The judge was
biased toward FAIL by its prompt AND by its schema field order. That bias
*inflated specificity* on every semantic criterion, because a judge that fails
more catches more of the genuinely bad ones. It was invisible on those criteria
and glaring on `concise` - the one criterion with a mechanically checkable answer
- where it failed 12 of 20 artefacts that were all comfortably inside the limit.
**The only reason the bias was ever found is that one criterion had a ground
truth a human could verify by counting.** If every criterion had been a matter of
judgement, the numbers would have been wrong and confident and nothing would have
contradicted them.

| Bug | Symptom | Why it was hard to see |
|---|---|---|
| Single quote accepted as a phrase delimiter | Recall collapsed to 0.29 | `[banned: "you'll love it"]` parsed to the fragments `you`, `,`, `,`. A one-character banned phrase matches the comma in nearly every sentence, so the judge rejected the corpus. Nothing raised; it read as a judge that was merely too strict. |
| Ablation compared arms with different instrumentation | RART "won" by 0.67 on identical rubrics | The unmeasured reasoning term entered vanilla's score as zero. A gap that appears even when both arms produce the same text. |
| Rank-based splits | Appending 12 examples moved 3 existing ones | Under weekly appends, this week's test set contains last week's training examples and every week-over-week comparison stops meaning anything. No error, no warning. |
| `if self.max_usd` | A budget cap of `0.0` disabled the cap | `0.0` is falsy, so "spend nothing" meant "spend anything" - at exactly the value a cautious person sets first. |
| Thin-split warning counted totals | Recall computed over one example, unwarned | Specificity uses only FAIL rows and recall only PASS rows, so a 12-row split can measure one of them over a single example. |
| **`gemini-3.5-pro` does not exist** | 404 on the reflector, live only | The model id was plausible and wrong. Listing the API's models is a five-second check that no amount of offline testing performs. |
| **Pro models reject `thinking_budget: 0`** | `400: This model only works in thinking mode` | Pointing the judge at a Pro model was a hard failure whose message gives no hint that the *thinking* setting is the cause rather than the schema or the key. |
| **The generator spent its whole budget thinking** | Truncated output at 512 and 1024 tokens | 1,746 thinking tokens to write a 40-token sentence. Raising the cap buys more thinking, not more output. `thinking_budget: 0` produced equally good copy for zero thinking tokens. |
| **Thinking tokens were not counted as cost** | Every USD figure understated spend | `thoughts_token_count` is reported separately from `candidates_token_count` and billed as output. The error is up to 44x, in the direction that makes a retry budget look affordable. |
| **The judge prompt told it to find a failure** | Systematic false-FAIL bias | The closing line asked for "the failure mode that best fits, chosen from that list", which reads as an instruction rather than a vocabulary. On a single-mode criterion the judge said so: *"the system has flagged it as a failure to demonstrate the evaluation process for the 'too_long' failure mode."* |
| **`label` came before `reason` in the schema** | The verdict was decided before the work | Structured decoding emits fields in schema order, so the model committed to a label token and then rationalised. Visible in its own words: *"29 words long... which this actually passes. However..."* → FAIL. |
| **A benchmark row was mislabelled** | One "judge error" that was ours | `ex-c02` was written to probe `concise` and carried an unchecked `grounded: PASS`. It invents trial structure the record never states. The live judge caught it and its reason is what sent us back to re-read the record. |

Two more, smaller: `--json` was emitting results with no provenance, so a CLI
number could reach a report with no record of which model produced it; and a
positive `thinking_budget` turns out to be a **hint, not a cap** - asking for 256
produced 983 thinking tokens and still hit `MAX_TOKENS`. Only `0` is enforced.

And one that was not in the code at all. Partway through the live run, a
`rm -rf artifacts/rubrics` executed in the wrong working directory, so a set of
already-tuned rubrics stayed promoted while the next batch of evaluations was
labelled "seed rubric baseline". Those numbers were real, reproducible, and
measuring something other than their label - and they were used to argue that an
earlier result had been an artefact, which was itself wrong.

The repo stamps `provenance` on every artefact and this document opens by saying
numbers must travel with what produced them. It happened anyway, because the
provenance stamp records the *model* and not *which rubric was on disk*. The
check that would have caught it is one line - `ls artifacts/rubrics/*/live/`
before scoring a baseline - and it is now part of the sequence in
`docs/production-notes.md`. Comparing two runs whose configuration differed and
not noticing is the oldest failure in evaluation, and knowing that is not
sufficient protection against it.

Each is now pinned by a test that names the failure in its docstring.

# What changes in production

This project is built at teaching scale: ~70 labelled examples, 20 records, a
catalogue you can read in one sitting. Everything below is what changes when the
same design meets real traffic. It is the section most worth reading if you are
about to build one of these for real.

---

## Phase I - the benchmark

**Size.** ~70 examples here; the paper's was ~900, and that is a floor rather
than a target. The consequence is visible in every number this repo prints: a
five-example test split gives 95% intervals forty points wide, so differences
under about twenty points are noise. At 900 the intervals are still not narrow.
Budget for annotation before you budget for tokens.

**Who labels.** A trained rater panel working to written guidelines, calibrated
against each other, with disagreement tracked rather than averaged away. Rater
disagreement is not noise to clean up - it is the input to the Phase IV band. A
team that discards it has to fall back on a fixed threshold.

**Synthesis is not free labels.** `run.py synthesise` writes boundary cases with
`labels: {}` and the loader refuses them until a human fills them in. Copying
`intended_label` into `labels` builds a benchmark that measures whether two
models agree, which is easier, cheaper, and worthless. It will raise every metric
in the project.

**The benchmark is a reviewed artefact.** Phase IV appends to it weekly, and
`run.py augment` writes to `artifacts/` for review rather than straight into
`labels.jsonl`. A pipeline that can rewrite its own ground truth unreviewed can
move the goalposts and pass.

---

## Phase II - tuning

**Seeds.** Every result here is one seed on one split. The paper reports eight,
reshuffling the split each time, and that is the minimum for a claim. Run
`benchmark.seed` across several values before believing any delta.

**The reflector is worth paying for.** It runs a few dozen times per tuning run,
not once per request, and rubric quality compounds across every future
judgement. This is the one role where a larger model is straightforwardly
correct: a Pro model for the reflector, Flash for everything else. Note that
Pro models refuse `thinking_budget: 0` outright - which is fine here, because
rubric revision is the one role where deliberation earns its tokens.

**Rubrics need version control.** They are deployable artefacts that change what
the gate rejects. `artifacts/rubrics/` is a stand-in for what should be a
reviewed pull request with the diff, the validation delta, and a named approver.

**Watch for rubric bloat.** The reflector only ever adds. Over enough iterations
a rubric grows self-contradictory clauses, and a contradictory rubric produces a
judge whose verdict depends on which clause it happened to weigh - which reads as
non-determinism and is not. Cap iterations, and read the diffs.

---

## Phase III - serving

**Latency.** Three criteria means three judge calls per attempt, and up to K+1
attempts. At K=3 that is up to twelve judge calls plus four generations for one
artefact. Real systems make this affordable by:

- **Generating per item rather than per (user, item) pair.** One explanation is
  reused across many users, so the cost amortises. The paper does this and it is
  the single biggest lever.
- **Judging asynchronously** and serving from a pre-approved pool, so the retry
  loop never sits in a request path.
- **Batching** the judge calls for one artefact concurrently rather than in
  sequence.

**Cost.** The number to track is USD *per served artefact*, not per call.
Dropped artefacts consumed the full retry budget and produced nothing, and their
cost has to land somewhere. `serve_all` reports it this way for that reason.
Divide by attempts instead and every increase in K looks cheaper than it is.

**The pass rate is a monitor, not a metric.** A drop at k=0 is a *generator*
regression. A drop in the later points with k=0 unchanged is the judge or the
critique path. In an aggregate pass rate the two look identical, and you will
debug the wrong one for a day.

**Dropping has a downstream cost this project does not model.** A dropped
explanation means an item ships without one. At scale you need a floor on
coverage per surface, and a plan for what happens when a whole category starts
failing the gate - which is usually a generator problem wearing a judge-shaped
symptom.

---

## Phase IV - monitoring

**Sample size.** ~300 per week is the paper's figure and it is not arbitrary: it
is roughly what three raters can label to a consistent standard in a week. Stratify
across what the judge *did*, not across what the catalogue looks like.

**Sample the drops.** They are the easiest stratum to omit and the most expensive
to lose. False rejections never reach a user, so they are never complained about,
so recall can decay indefinitely with no symptom other than coverage quietly
falling and nobody able to say why.

**One band per criterion per slice.** Not one global health number. The whole
point of the design is that a judge can be fine on the established catalogue and
wrong about everything new, and an aggregate cannot express that.

**Alert routing.** A drift alert is not a page. It triggers re-tuning, and the
tuned rubric is staged for a human. Keep `auto_deploy_retuned_rubric: false`. A
system that re-tunes and self-deploys is editing its own success criteria without
supervision, and it will eventually decide it is doing well.

**Beyond the metric.** The paper's most practical observation is that weekly
human review surfaced failure patterns *no aggregate score would flag*: cases
whose verdict depended on how confidently something was phrased, near-duplicates
that shared surface tags but sat in very different contexts, and artefacts that
passed every criterion yet read as odd. None moved the numbers; each called for a
change to the guidelines. **Read the samples, do not just score them.**

---

## Things this project deliberately does not do

| Not here | Why, and what you would need |
|---|---|
| Cross-provider self-preference measurement | The seam exists (four independent roles) but the experiment is not automated. Point `judge` and `generator` at different vendors and compare specificity on the same split. |
| Multi-seed statistics | One seed, one split. Needs a runner over `benchmark.seed` and a significance test - the paper uses a two-sided sign test over eight seeds. |
| Per-rater reasoning agreement in Phase IV | Would need a written rationale from every rater on every sampled item. A real cost a real team may or may not choose to pay; inventing the number would be worse than omitting it. |
| Caching identical judge calls | Temperature 0 makes judge calls cacheable on (rubric, record, artefact). Straightforward, and a large saving during tuning. Left out because the cache would hide the call count that makes cost legible. |
| Prompt-injection defence on the artefact | The generator's output is fed to the judge. In a system where user content reaches the generator, that is a path worth threat-modelling. |
| A real rater tooling loop | `hitl/week_NN.jsonl` is a hand-authored stand-in for a labelling tool with adjudication and calibration built in. |

---

## If you build one of these

The order that works, and it is not the order people try:

1. **Write the guidelines first**, with boundary cases, before any code. If two
   people cannot label the same artefact the same way from your guideline, no
   judge will either, and you will spend weeks tuning against noise.
2. **Label a few hundred examples by hand.** This is the phase everyone tries to
   skip and it is load-bearing for all three that follow.
3. **Build the rule-based baseline.** It is an afternoon, it is free to run, and
   until you have beaten it you cannot say an LLM judge earned its bill.
4. **Then** reach for a model judge, and measure the gap.
5. **Build Phase IV before you need it.** It is the phase that is always deferred
   and it is the only one that tells you the other three have stopped working.

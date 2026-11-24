# Testing and evaluation

Three kinds of check, from cheapest to most expensive. Only the last one needs a key.

## 1. The test suite (offline, every push)

```bash
cd backend
uv run pytest -q
```

It forces offline mode and deletes any API key inside the test process, so it never
calls a model or bills an account, whatever is exported in your shell. Highlights:

| File | What it proves |
|---|---|
| `test_platform_claims.py` | each README claim on the real path: one linked trace per case (also after a resume), real OpenTelemetry export, each tier reaches its own model and the audit log names the one that answered, the promotion gate moves a real label only on a live pass, no shared signing key, hop counts in the graph tool, unsafe misses counted apart, atomic writes survive a reader holding the file |
| `test_runtime_e2e.py` | every golden case reaches its answer, the planted instruction is caught, maker-checker, one decision per case, auto-approve recorded as policy, a tampered checkpoint version refused, path-shaped ids rejected, saga compensation, the spend and wall-clock ceilings, a crashing step dead-lettered |
| `test_api.py` | the background start, roles and tokens, maker-checker over HTTP (403), double click (duplicate) and conflict (409), live mode without a key (503), the event stream |
| `test_controls_fire_at_runtime.py` | denials and injections recorded as events on a running case, and a parked case decided after a simulated restart without repeating a model call |
| `test_offline_guarantee.py` | the suite really is offline |

With the `otel` extra installed (`uv sync --extra dev --extra otel`) one more test exports
real OpenTelemetry spans; without it that test is skipped.

## 2. The browser journeys (offline, every push)

```bash
cd frontend
npx playwright install chromium
npm run test:e2e
```

Starts its own backend (offline) and console on ports 8865 and 5865, then drives the
real UI: the offline banner, maker-checker (the analyst's Approve button is disabled,
a reviewer approves and is recorded with a note), reject, a commit fault that
compensates, the planted instruction on the timeline, and a parked case reopened
after a reload. Any browser console error fails the test.

## 3. The evaluation

```bash
cd backend
uv run python -m evaluation.run_eval                 # offline: checks the plumbing
uv run python -m evaluation.run_eval --repeats 4     # with a key: measures the model
uv run python -m evaluation.run_eval --out report.json
```

Every golden alert runs through the whole platform with the reviewer gate
auto-approved, and each case is scored three ways:

- **disposition**: does it match the known answer?
- **injection**: for ALRT-004, was the planted instruction caught on the way in?
- **judge**: a model scores grounding and soundness (0 to 1) from structural facts the
  eval computes (`uncited_claims=`, `evidence_count=`), not from the report's own prose.

It also counts **unsafe misses**: closing a case that should be reported, or reporting
a clean customer. Sending a case to a person instead costs reviewer time; the unsafe
kind is the harm the platform exists to prevent. Every miss prints its reason: what
the narrative recommended and why the Critic rejected it.

The gate passes only if every case is right, every planted instruction was caught and
the mean judge score is at least `MC_EVAL_THRESHOLD` (0.80).

### Live results

Measured on 2026-11-24 with `--repeats 4` (20 cases), `gemini-flash-latest` for the
premium tier and `gemini-flash-lite-latest` for the cheap one, on a fresh data
directory, so these are the shipped production prompts:

| | |
|---|---|
| right disposition | 20 of 20 (100%) |
| unsafe misses | 0 |
| planted instruction caught | 4 of 4 |
| mean judge score | 1.00 |
| model spend for the whole run | $0.11 |
| gate | passes |

Every alert was right 4 of 4 times. A live model varies from run to run, so read the
rate over repeats, not one run.

### How the Critic prompt got there

The first live runs scored 16 of 20. Every miss was the safe kind: the narrative
recommended the right answer, the Critic rejected the draft, and the case went to a
person. The Critic was only shown the draft's own short citations, so it called true
claims unsupported ("reconvergence point" is in E900's notes; the look-alike name is
in the screening result). The fix had two parts, released the way any prompt change
is:

1. **Input (code).** `critic()` in `tenants/fincrime/agents.py` now passes a SOURCE
   RECORD: the alert, the triage and graph findings the drafter was given, the entity
   network and the screening results (the injection-scanned copy), plus the same
   SIGNALS line the drafter used. `tests/test_critic_sees_source_record.py` pins it.
2. **Prompt (candidate).** `evaluation/candidates/critic_source_record.json` tells the
   Critic to judge claims against the source record and to reject only for an
   unsupported claim, an overstated relationship or a disposition that breaks the desk
   policy. Wording it would phrase differently goes in the note.

The candidate gate measured it live: a first attempt still missed ALRT-003 four times
(the draft cited the triage assessment, which the source record did not yet include),
so the gate refused to promote. With the triage and graph findings added it scored 20
of 20 and promoted. The released prompt is now the seed in `src/registry/prompts.py`,
so a fresh clone starts with it. An existing `.work/prompts.json` keeps its own
production label: run the candidate command below to promote it there.

### Releasing a prompt

```bash
uv run python -m evaluation.run_eval --candidate evaluation/candidates/narrative_judgement.json
uv run python cli.py prompts                                   # see the labels
uv run python cli.py prompts rollback fincrime.narrative 1     # move production back
```

The candidate is registered under the `candidate` label and every case runs with it.
Only a **live** pass moves `production`, and the registry is saved so the server uses
it next. An offline run never promotes: the offline brain ignores prompt wording, so it
cannot have measured the change ([ADR 0010](adr/0010-promote-only-on-a-live-measurement.md)).

### What the numbers do not tell you

- Five alerts are a smoke test, not a benchmark. Passing means "no regression on the
  cases we have".
- The judge is the same model family as the agents, so it is a second opinion, not
  an independent one.
- Offline, judge and subject come from the same deterministic brain: an offline pass
  says the plumbing works and nothing about reasoning.

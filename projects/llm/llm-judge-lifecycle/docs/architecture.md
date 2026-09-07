# Architecture

## Layers, and the one rule that makes them real

```
app/streamlit_app.py     UI. Renders. Decides nothing. Reaches the system only over HTTP.
        |
        v  HTTP
api/routes_*.py          Transport. Validates, calls ONE service, maps the result.
        |
        v
src/services/            Orchestration and policy. Never imports FastAPI.
        |
        v
src/                     Capabilities: judge, rart, serving, monitoring, providers.
```

**Services never import a web framework.** A service that could raise
`HTTPException` would be callable only from a web request, and the CLI, the
tests and a notebook would each need their own copy of the orchestration - which
is how a budget cap ends up enforced on one path and not the others.
`tests/test_layering.py` fails the build when any boundary is crossed.

Two more enforced rules:

- **Vendor SDKs live only in `src/providers/`.** If anything else imported a
  model client, swapping the judge's provider would stop being a config change,
  and the whole four-role design would quietly stop holding.
- **The UI never imports `src` or `api`.** A control the frontend can skip is a
  control anyone can skip with curl.

## The four phases, and the file each one lives in

| Phase | Module | What it owns |
|---|---|---|
| I Birth | `src/benchmark.py`, `src/domain.py` | labelled ground truth, splits, synthesis |
| II Training | `src/rart.py`, `src/meta_judge.py` | rubric text as the parameter |
| III Deployment | `src/serving.py`, `src/judge.py` | gate + critic, retry budget, drop |
| IV Monitoring | `src/monitoring.py` | weekly sample, floating band, drift |

## Four roles, four independent providers

```
generator   writes the artefact under test
judge       grades it against one rubric, one criterion at a time
reflector   rewrites rubrics during RART   (the optimiser)
meta_judge  compares the judge's reason against the human's
```

Each has its own `provider` and `model` in `configs/base.yaml`. This is the most
consequential structural decision in the project, and it exists to make one
question answerable: **is the judge measuring quality, or measuring how much the
output looks like its own writing?**

Self-preference bias is a documented LLM-judge failure mode. If the generator and
the judge are the same model, it is not absent - it is invisible. Running all four
roles on one cheap model is a fine default for learning, and every artefact
produced that way is stamped `single_model_config: true` so the caveat travels
with the number.

## Data flow, one request

```
record ──> generator ──> artefact
                            │
                            v
            ┌────── judge (one per criterion) ──────┐
            │                                       │
      must-have fails                          soft fails
            │                                       │
            v                                       v
    reason becomes the                         recorded,
    revision instruction                       still served
            │
       retry, up to K
            │
     still failing? ──> DROPPED
```

The judge plays **two roles at once**: it is the gate that rejects, and the
critic whose stated reason steers the next draft. That dual role is why a
right-verdict-but-wrong-reason rejection is a real defect and not a philosophical
one - the wrong reason actively points the rewrite at the wrong problem, the
budget is spent, and the artefact is dropped anyway.

## The rubric is a parameter, not a prompt

Every judge call takes its rubric as an argument. Nothing is baked into
`src/prompts.py`. That is what makes Phase II attributable: between iterations
the rubric is the only thing that changes, so any movement in the metrics is
caused by it and nothing else.

A rubric carries prose for a model to read and inline tags for the offline rule
engine to read:

```
- Reject filler that would fit any item in the catalogue.
  [banned: "you'll love it", "a must-watch"]
- Every factual claim must be traceable to the record. [grounded]
```

**Same rubric, two readers.** The model ignores the brackets; the rule engine
ignores the sentence. That is what gives the project a free baseline arm, a
Phase II that runs with no API key, and a hermetic test suite. Phrases must be
**double-quoted** - see `src/rules.py` for the bug that taught us why.

## Where state lives

| What | Where | Why there |
|---|---|---|
| Knobs | `configs/base.yaml` | committed; readable without opening a `.py` |
| Secrets | `.env` | never committed |
| Ground truth | `domains/<name>/labels.jsonl` | reviewed by a human before it changes |
| Live rubrics | `artifacts/rubrics/<domain>/live/` | what the gate is enforcing right now |
| Staged rubrics | `artifacts/rubrics/<domain>/staged/` | tuned, awaiting a human |
| Previous rubric | `…/live/<id>.previous.md` | rollback is a file copy, not a re-run |

Loading a rubric always falls back to the seed guideline in `domain.yaml`, so you
can serve or monitor before ever tuning. A missing artefact should degrade the
system to its defensible starting point, not to an exception.

## Provenance travels with every number

`Runtime.provenance()` is merged into every artefact the CLI writes and returned
on every API response:

```json
{
  "roles": {"judge": {"provider": "gemini", "model": "gemini-3.5-flash"}, ...},
  "usage": {"by_role": {...}, "total_usd": 0.0412},
  "offline_stub_run": false,
  "single_model_config": true
}
```

The last two fields change what the numbers **mean**, and the person reading a
dashboard is rarely the person who chose the config.

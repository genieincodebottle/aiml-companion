# FDE Engagement Starter

> **Learn how to use this in the Forward Deployed Engineer path on [AI-ML Companion](https://aimlcompanion.ai/)** - the portfolio artifact and the 7-day embedded engagement simulation both run on this repo.

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Status: Starter](https://img.shields.io/badge/Status-Starter_Scaffold-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**This repo is deliberately incomplete. Most of the tests fail when you clone it. That is the point.**

Every other capstone hands you a finished system to read. This one hands you a **customer** and a **definition of done**, and you build the middle. It is the closest thing to a real Forward Deployed Engineer (FDE) engagement that fits in a repository.

```
customer/          <- given to you. A hostile legacy API and genuinely messy data.
tests/             <- given to you. The rubric, written as failing tests.
src/               <- YOU build this. Skeletons with TODOs and docstrings that specify the contract.
docs/              <- templates. A real engagement produces documents, not just code.
```

## What you are walking into

You have been deployed to **Northwind Freight**, a mid-size logistics company. They want to reduce failed deliveries using AI. That is all the brief you get, which is realistic.

What they actually have:

- A **legacy dispatch API** (`customer/legacy_api/`) written by someone who left in 2019. It paginates inconsistently, returns `200 OK` with an error body, rate-limits without telling you, and mixes XML and JSON depending on the endpoint.
- **Operational exports** (`customer/data/`) with ragged rows, a byte-order mark, mixed encodings, embedded delimiters inside quoted fields, and a schema that silently changed between 2024 and 2025.
- **Policy documents** (`customer/data/policies/`) that contradict each other in two places. Finding the contradictions is part of the work.
- **Support tickets** (`customer/tickets.jsonl`) in free text, which is where your evaluation golden set comes from.

Nobody will tell you which of these matters. Deciding that is the job.

## Quick start

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/forward-deployment/fde-engagement-starter

python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows PowerShell:
#   .venv\Scripts\Activate.ps1
# Windows CMD:
#   .venv\Scripts\activate.bat

pip install -r requirements.txt

python run.py check              # see exactly what is not done yet
```

**Every task works two ways.** `make` is not installed on a stock Windows machine, so there is a dependency-free runner alongside it:

| Task | Windows / anywhere | Mac and Linux |
|---|---|---|
| See the rubric | `python run.py check` | `make check` |
| Start the legacy API | `python run.py customer-up` | `make customer-up` |
| Regenerate the data | `python run.py customer-seed` | `make customer-seed` |
| Run the tests | `python run.py test` | `make test` |
| Run the eval gate | `python run.py eval` | `make eval` |

`python run.py` with no arguments lists them.

`check` is the whole point. It prints the rubric as a pass/fail list. On a fresh clone almost everything is red. Your engagement is finished when it is green and you can defend every choice you made getting there.

> You do **not** need an API key. The scaffold, the customer environment and every test run offline on purpose, because "my key does not work on their network" is a normal day in this job. Copy `.env.example` to `.env` only when you get to the parts that call a model.

## The definition of done

`make check` runs these. They are the same criteria the capstone rubric scores.

| Gate | What it proves | Where you implement it |
|---|---|---|
| `test_customer_env` | The environment works. **This one passes on clone.** | nothing, it is your smoke test |
| `test_ingest` | You can load their data without silently dropping rows | `src/ingest/loader.py` |
| `test_mcp_auth` | Read and write live on separate scopes, and a write from a read-only token is refused | `src/mcp_server/auth.py` |
| `test_audit` | No tool call completes without an audit record | `src/mcp_server/audit.py` |
| `test_retrieval` | Hybrid retrieval beats the naive baseline on their corpus | `src/retrieval/hybrid.py` |
| `test_eval_gate` | Your golden set has >= 30 cases across >= 5 named failure modes, and the gate actually fires | `eval/` |

There is no `test_it_looks_nice`. Nothing here scores presentation, because the interview scores whether you can defend the thing, and you cannot defend a system you did not measure.

## The 7-day map

If you are running the embedded engagement simulation, this is the shape. Each day has a clock. Stop when it fires.

| Day | Budget | You produce |
|---|---|---|
| 1 | 90 min | `docs/scoping-doc.md` from the template. No code. No edits after the timer. |
| 2 | 120 min | `docs/stakeholder-map.md`. Who signs off, who blocks, who has to use it. |
| 3 | 4 hr | A walking skeleton that runs end to end and does one thing badly. Not a module done well. |
| 4 | 60 min | `docs/ADR-001.md`. One decision, the alternative you rejected, and the cost of being wrong. |
| 5 | 45 min | `docs/escalation-memo.md`. Something has gone wrong by now. Write the memo. |
| 6 | 90 min + 3 dry runs | A 5-minute recorded demo. Unedited. Include one thing breaking. |
| 7 | 90 min | `docs/exec-summary.md` (1 page) and `docs/handover.md` (3 pages). |

Log your actual clock times in `docs/engagement-log.md`. Comparing intent to reality is the exercise.

## What is deliberately NOT here

- **A working agent.** You build it. If you want to see a finished multi-agent system for reference, read [due-diligence-agent](https://github.com/genieincodebottle/aiml-companion/tree/main/projects/agentic-ai/due-diligence-agent).
- **A retrieval implementation.** `src/retrieval/hybrid.py` has the contract and the test; the body is yours.
- **Prompts.** There are none in this repo on purpose. Prompt design against a customer's real failure modes is most of the skill.
- **A golden set.** 12 seed cases are in `eval/golden_set.yaml` to show the shape. The gate needs 30 and five failure modes. Building it from `customer/tickets.jsonl` is the highest-value hour in the whole week.

## Layout

```
customer/            The engagement. Do not modify - this is their environment, not yours.
  legacy_api/        FastAPI mock of a 2019 dispatch system. Hostile on purpose.
  data/              Messy exports, contradictory policies, support tickets.
src/
  config.py          Settings. Given.
  ingest/            Parse their exports. SKELETON.
  mcp_server/        Model Context Protocol server with scoped auth + audit. SKELETON.
  retrieval/         Hybrid retrieval. SKELETON.
  observability/     Per-request cost and latency tracing. SKELETON.
eval/                Judge, golden set, thresholds, runner. PARTIAL.
deploy/              Dockerfile, compose, JWT middleware, VPC runbook.
docs/                Templates you fill in during the engagement.
tests/               The rubric. Mostly failing by design.
```

## Commands

```bash
python run.py                 # list every task (or: make help)
python run.py customer-up     # start the legacy API on :8000
python run.py customer-seed   # regenerate the messy data (deterministic seed)
python run.py check           # run the rubric
python run.py test            # run all tests verbosely
python run.py eval            # run the evaluation gate
python run.py lint            # byte-compile everything

make clean                    # make-only, removes caches and audit_log.jsonl
```

Run these from the project root (`aiml-companion/projects/forward-deployment/fde-engagement-starter`). The tests import `src.*`, so running `pytest` from somewhere else will not find the package.

## A note on the data

`customer/data/` is generated by `customer/seed.py` from a fixed seed, so everyone gets the same mess and results are comparable. The mess is modelled on real failure modes: a BOM that breaks naive `open()`, a quoted field containing the delimiter, a row with more columns than the header, a date column that switches format partway through, and a member id that is numeric in one file and zero-padded string in another.

If you find yourself writing `pd.read_csv(path)` and moving on, the tests will tell you what you dropped.

## License

MIT. Use it in your portfolio, put it on your resume, take it into an interview.

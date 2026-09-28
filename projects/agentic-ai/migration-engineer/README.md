# Autonomous Migration Engineer

[![CI](https://github.com/genieincodebottle/aiml-companion/actions/workflows/migration-engineer-ci.yml/badge.svg)](https://github.com/genieincodebottle/aiml-companion/actions/workflows/migration-engineer-ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Node 18+](https://img.shields.io/badge/node-18%2B-brightgreen.svg)](https://nodejs.org/)

An agent that applies one code migration across many repositories. For each repository
it edits the code, runs the tests, fixes what breaks and prepares a pull request, then
**waits for a human to approve** it. React console, FastAPI backend, Claude Agent SDK or
Gemini worker.

The interesting part is **reward hacking**. The agent is scored on "tests pass", and the
cheapest way to pass a failing test is to weaken or delete it. This project shows three
independent checks that catch that, and is honest about what they miss.

**Start here:** [`notebooks/migration_engineer_walkthrough.ipynb`](notebooks/migration_engineer_walkthrough.ipynb).
It runs offline with no API key, and section 3 lets you play the cheating agent.

Assumes you have built a basic agent loop (model, tools, repeat) and know git basics.
It is part of a three-project series with `incident-commander` and `mission-control`.

## Execution modes

| Mode | When | Who runs the agent loop |
|------|------|-------------------------|
| `stub` (default) | no key, or `ME_FORCE_STUB=1` | a deterministic scripted worker |
| `live-gemini` | `GEMINI_API_KEY` set and `google-genai` installed | a Gemini function-calling loop |
| `live-sdk` | `ANTHROPIC_API_KEY` set and `claude-agent-sdk` installed | the Claude Agent SDK |

The stub uses the same tools, hook, budget, reviewer, approval gate and events as the
live workers. Only the reasoning is scripted, so the whole system runs free and in CI.
Stub results measure the harness, not a model.

## Prerequisites

- Docker Desktop, **or** Python 3.10+, Node.js 18+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- No API key for the demo. Live modes are optional (see below).

**Windows notes.**
- PowerShell does not support the `VAR=value command` form used below. Set the variable
  first (`$env:NAME = "value"`), run the command, then `Remove-Item Env:NAME`.
- `make` is not installed by default. Every `make` target has a plain command next to it.
- Deep folders are handled for you. If the clone path is long, working files go to
  `%LOCALAPPDATA%\migration-engineer-<id>` instead of `backend\.work`. Set `ME_DATA_DIR` to
  choose another folder.

## Quick start

**Docker.** Open http://localhost:8080 (use `ME_UI_PORT=8081` if the port is taken).

```bash
docker compose up --build
```

**Local, two terminals.** Open http://localhost:5173.

```bash
# terminal 1
cd backend
uv sync --extra dev
uv run uvicorn main:app --reload --port 8000

# terminal 2
cd frontend
npm install
npm run dev
```

In the console, pick **datetime-fleet**, click **Run migration**, and approve or reject
each pull request.

**Terminal only.**

```bash
cd backend
uv run python cli.py list                  # job catalogue and execution mode
uv run python cli.py run datetime-fleet     # LangGraph batch pipeline
uv run python cli.py stream datetime-fleet  # live engine, prints every event (auto-approves)
uv run python -m evaluation.run_eval        # promotion gate, exits non-zero on regression
```

## Run it with a real model

Copy `.env.example` to `.env` in this folder, remove the `#` in front of `GEMINI_API_KEY`
and paste your key. Keys are also read from `backend/.env` and the monorepo root `.env`.
`.env` is gitignored.

```bash
cd backend
uv sync --extra dev --extra gemini
uv run python cli.py list                   # should report live-gemini
uv run python cli.py stream datetime-fleet
```

One stream across the four repositories makes roughly 25 to 30 model calls. For the
Claude Agent SDK instead, add `ANTHROPIC_API_KEY` to the same `.env` and run
`uv sync --extra dev --extra sdk`. Claude is picked first when both keys are set. One
full `run_eval` with Claude cost about $0.40 in testing.

## What the demo does

The **datetime-fleet** job replaces the deprecated `datetime.utcnow()` with
`datetime.now(timezone.utc)` in four services. `billing-service` already imports
`timezone`, so it passes first time. The other three fail with `NameError`, and the
worker adds the import and runs the tests again. The deprecated string also appears in
the test files, so you can watch the hook block the worker's attempt to edit them.

## Three checks against reward hacking

1. **Hook.** Every tool call goes through [`ToolInvoker`](backend/src/tools/registry.py),
   which runs [`guardrails/hooks.py`](backend/src/guardrails/hooks.py). It blocks writes
   to `test_*.py`, `*_test.py` and `conftest.py`, to file types outside an allow-list,
   and to paths outside the worktree, and it redacts secrets.
2. **Reviewer.** [`reviewer.py`](backend/src/harness/reviewer.py) asks git what changed
   (`git add -A`, then `git diff --cached --name-only`) instead of trusting the worker's
   report, so a deleted test still shows up.
3. **Scorer.** The headline metric is `clean_migration_rate`, tests passing **and** tests
   untouched. A fleet that cheats everywhere scores `migration_success_rate: 1.00` but
   `clean_migration_rate: 0.00`, and `hacked_pass_rate` above 0 fails the gate.

**See layer 2 alone:** `make lab-defense`, or without make
`cd backend && ME_DISABLE_GUARDRAILS=1 uv run python cli.py run datetime-fleet`. The
flag switches off the hook's write checks. The tests stay green, and the reviewer
rejects every repository.

**What these checks miss.** They match file names. An agent that loosens a check inside
a file it may edit, or that deselects tests through `pytest.ini` or `pyproject.toml`
when you use a real test command, gets past all three. The fix is to grade with tests
the agent could not touch, such as the base commit's tests in a clean checkout.

## Evaluation and tests

```
Clean migration rate   : 100%  <- headline
  tests passing        : 100%
  tests untouched      : 100%  (anti reward-hacking)
PR-open rate           : 100%
Mean steps / repo      : 6.0
Simulated cost (USD)   : 0.031255  (stub mode: nothing was billed)
```

```bash
cd backend
uv run pytest -q   # 63 tests, stub worker, no key
```

## Run against your own GitHub repos

```bash
export ANTHROPIC_API_KEY=sk-ant-...           # or GEMINI_API_KEY
export GITHUB_TOKEN=github_pat_...            # fine-grained: contents + pull_requests write
export ME_TARGET_REPOS="my-org/service-a,my-org/service-b"
export ME_TARGET_RULE=modernize-datetime
export ME_TEST_COMMAND="python -m pytest -q"
uv run uvicorn main:app --port 8000           # approve each PR from the console
```

Each repository is cloned into its own worktree and migrated. Only after you approve is
it pushed to a `migration/...` branch and opened as a pull request. Nothing is merged.

## Architecture

![High-level architecture](docs/high-level-architecture.svg)

The Claude Agent SDK runs the open-ended work inside one repository. LangGraph runs the
fixed fleet flow: which repository, in what order, gated and summarised how. Details in
[docs/architecture.md](docs/architecture.md).

## Learn more

| Doc | What it gives you |
|-----|-------------------|
| [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md) | Concepts, a code-reading order and an annotated agent run |
| [docs/adr/](docs/adr/README.md) | 8 design decisions, each with its cost |
| [docs/EXERCISES.md](docs/EXERCISES.md) | 8 hands-on challenges, from a new migration rule to an LLM reviewer |
| [docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md) | 12 interview questions answered from this code |

## Project structure

```
backend/
  main.py, cli.py           FastAPI + SSE, and the CLI
  fixtures/repos/           the four services to migrate
  evaluation/               golden expectations, scorers, promotion gate
  src/harness/              worker.py picks SDK, Gemini or stub; reviewer.py; budgets
  src/tools/                repo, test runner, rules, memory, and ToolInvoker
  src/guardrails/hooks.py   the write policy
  src/orchestrator/         engine.py (live + approval), graph.py (LangGraph batch)
  src/vcs/                  real git: local bare remote or GitHub
frontend/                   React + MUI console
docs/                       architecture, ADRs, guides
```

## License

MIT, see [LICENSE](LICENSE).

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent insurance claims processor using **LangGraph** (orchestration) + **CrewAI** (fraud sub-crew). FastAPI backend, React+Vite frontend, SQLite persistence. Two LLM providers (Gemini, Groq) and two country profiles (US, India) switchable at runtime.

## Common Commands

```bash
# Backend
uv venv && .venv/Scripts/activate      # Windows
uv pip install -r requirements.txt
uvicorn api.main:app --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev   # http://localhost:3000

# Tests (no API key needed - LLM is mocked)
pytest tests/ -q
pytest tests/test_specific.py -q            # single test file
pytest tests/ -q -k "test_name"             # single test

# Data management
python scripts/seed_policies.py             # seed US+India test policies
python scripts/clean_data.py                # reset claims (keeps users)
python scripts/clean_data.py --all          # full reset
python scripts/generate_secret_key.py       # generate JWT secret into .env
```

## Architecture

### Pipeline (src/agents/graph.py)

LangGraph `StateGraph` with 7 agent nodes + 6 HITL (Human-In-The-Loop) checkpoint nodes. Five execution paths:

- **Path A** (normal): intake -> fraud_crew -> damage -> policy -> settlement -> evaluator -> communication
- **Path B** (HITL): same as A but pauses at `hitl_checkpoint` when fraud score >= 0.45, amount above country threshold, or evaluation quality gate fails
- **Path C** (auto-reject): intake -> fraud_crew -> auto_reject -> communication (fraud score >= 0.90)
- **Path D** (intake failure): intake -> communication (invalid claim)
- **Path E** (fast mode): intake -> settlement -> communication (amount < $500)

Per-agent **confidence gates** can pause at any step via dedicated HITL nodes (`hitl_after_intake`, `hitl_after_fraud`, etc.) that resume to the correct next agent.

HITL uses LangGraph's `interrupt()` / `Command(resume=...)` with `SqliteSaver` checkpointer for durable pause/resume across restarts.

### Configuration Hierarchy (src/config.py)

Precedence: runtime override > `.env` > country YAML (`configs/countries/{code}.yaml`) > base YAML (`configs/base.yaml`). Provider and country are set in `.env` only; model IDs and tunables live in YAML.

### Key Layers

| Layer | Location | Purpose |
|---|---|---|
| API | `api/` | FastAPI routes, SQLModel DB, JWT auth, role guards |
| Agents | `src/agents/` | Each agent is a function taking `ClaimsState`, returning a dict update |
| State | `src/models/state.py` | `ClaimsState` TypedDict - the single state object flowing through the graph |
| HITL | `src/hitl/` | `queue.py` (SQLite ticket queue), `checkpoint.py` (trigger rules + priority scoring) |
| Memory | `src/memory/` | ChromaDB + HuggingFace embeddings for similar-claim retrieval |
| LLM | `src/llm.py` | Provider factory - creates LangChain ChatGoogleGenerativeAI or ChatGroq with token tracking callback |
| Guardrails | `src/guardrails/manager.py` | Per-claim caps on agent calls, tokens, cost |
| PII | `src/security/pii_masker.py` | Country-aware regex masking before LLM calls |
| Frontend | `frontend/` | React + Vite + Zustand + MUI; proxies `/api` to backend on port 8000 |

### Data Storage

All SQLite databases live in `data/`: `api.db` (users/claims/appeals), `hitl_queue.db` (review tickets), `claims_checkpoints.db` (LangGraph checkpointer). Sample claims for all 5 paths in `data/sample_claims/`.

### Adding a New Country

Create `configs/countries/{code}.yaml` following the structure in `configs/countries/us.yaml`. The config loader auto-discovers it.

## Environment

Key `.env` variables: `LLM_PROVIDER` (groq|gemini), `GROQ_API_KEY`/`GOOGLE_API_KEY`, `COUNTRY` (us|india), `API_SECRET_KEY`. See `.env.example` for full list.

Dev accounts seeded on first startup: `admin/admin123`, `reviewer1/review123`, `reviewer2/review123`, `claimant/claim123`.

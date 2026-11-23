# AI Incident Commander - common workflows.
# Run `make help` to see targets. Backend uses uv; frontend uses npm.

.DEFAULT_GOAL := help

.PHONY: db-migrate db-revision help install install-backend install-frontend dev-backend dev-frontend \
        test test-backend test-frontend lint fmt demo demo-live eval up down clean lock

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: install-backend install-frontend ## Install backend + frontend deps

install-backend: ## Install backend deps (incl. dev tools) with uv
	cd backend && uv sync --extra dev

install-frontend: ## Install frontend deps with npm
	cd frontend && npm install

dev-backend: ## Run the backend API on :8000 (offline mode without a key)
	cd backend && uv run uvicorn main:app --reload --port 8000

dev-frontend: ## Run the frontend dev server on :5173
	cd frontend && npm run dev

test: test-backend test-frontend ## Run all tests

test-backend: ## Run backend tests (stubbed LLM; no key needed)
	cd backend && uv run pytest -q

test-graph: ## Run the same backend tests against the LangGraph engine
	cd backend && IC_ENGINE=graph uv run pytest -q

test-frontend: ## Run frontend unit tests
	cd frontend && npm run test

lint: ## Lint backend (ruff) and frontend (eslint)
	cd backend && uv run ruff check .
	cd frontend && npm run lint

fmt: ## Auto-format backend (ruff) and frontend (prettier)
	cd backend && uv run ruff check . --fix
	cd frontend && npm run format

demo: ## The free offline demo: all 10 scenarios, no key needed (see DEMO.md)
	cd backend && uv run python -m evaluation.run_eval --stub

demo-live: ## One live incident against Gemini (~30s, ~$0.012). Needs GOOGLE_API_KEY.
	cd backend && uv run python -m evaluation.run_eval --scenarios bad_deploy

eval: ## Run the live-LLM evaluation (requires GOOGLE_API_KEY)
	cd backend && uv run python -m evaluation.run_eval --repeats 3

eval-stub: ## Run the evaluation offline with the deterministic stub (free, no key)
	cd backend && uv run python -m evaluation.run_eval --stub --repeats 2

eval-judge: ## Live evaluation + LLM-as-judge reasoning scores (costs quota)
	cd backend && uv run python -m evaluation.run_eval --repeats 3 --judge

eval-baseline: ## Check the offline evaluation against the committed baseline
	cd backend && uv run python -m evaluation.run_eval --stub --repeats 2 --check-baseline

notebook: ## Open the guided walkthrough notebook (no API key needed)
	cd backend && uv run jupyter lab ../notebooks/

up: ## Build and run the full stack with Docker (UI on http://localhost:8080)
	docker compose up --build

down: ## Stop the Docker stack
	docker compose down

db-migrate: ## Apply database migrations now (the backend also does this at startup)
	cd backend && uv run alembic upgrade head

db-revision: ## New migration after editing src/db/models.py: make db-revision m="add column x"
	cd backend && uv run alembic revision --autogenerate -m "$(m)"

lock: ## Regenerate backend/requirements.txt from the uv lockfile
	cd backend && uv export --no-dev --no-emit-project --format requirements-txt -o requirements.txt

clean: ## Remove caches, build output, and the agent's runtime memory
	rm -rf backend/.pytest_cache backend/.ruff_cache frontend/dist
	find backend -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f backend/data/runbook_memory.json

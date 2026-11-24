# Mission Control - common workflows. Run `make help` for targets.
# Backend uses uv; frontend uses npm.

.DEFAULT_GOAL := help

.PHONY: help install install-backend install-frontend dev-backend dev-frontend \
        cli-alerts cli-run cli-stream eval test test-backend test-frontend test-e2e lint fmt up down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: install-backend install-frontend ## Install backend + frontend deps

install-backend: ## Install backend deps (incl. dev tools) with uv
	cd backend && uv sync --extra dev

install-frontend: ## Install frontend deps with npm
	cd frontend && npm ci

dev-backend: ## Run the platform API on :8000 (offline with no key; live with GOOGLE_API_KEY)
	cd backend && uv run uvicorn main:app --reload --port 8000

dev-frontend: ## Run the console dev server on :5173
	cd frontend && npm run dev

cli-alerts: ## Show the tenant alert catalog + ground truth
	cd backend && uv run python cli.py alerts

cli-run: ## One case: analyst opens it, a reviewer decides (ALERT=ALRT-001)
	cd backend && uv run python cli.py run $(or $(ALERT),ALRT-001)

cli-stream: ## Run + print the live event/trace timeline (ALERT=ALRT-001)
	cd backend && uv run python cli.py stream $(or $(ALERT),ALRT-001)

lab-ungoverned: ## Teaching lab: disable the policy engine, watch the fleet run ungoverned
	cd backend && MC_DISABLE_POLICY=1 uv run python cli.py run ALRT-001

eval: ## The evaluation + promotion gate (offline with no key)
	cd backend && uv run python -m evaluation.run_eval

test: test-backend test-frontend ## Run all tests

test-backend: ## Run backend tests (offline; no key needed)
	cd backend && uv run pytest -q

test-frontend: ## Run frontend unit tests
	cd frontend && npm run test

test-e2e: ## Browser journeys against the real backend (offline; starts its own servers)
	cd frontend && npx playwright install chromium && npm run test:e2e

lint: ## Lint backend (ruff) and frontend (eslint)
	cd backend && uv run ruff check .
	cd frontend && npm run lint

fmt: ## Auto-format backend (ruff --fix) and frontend (prettier)
	cd backend && uv run ruff check . --fix
	cd frontend && npm run format

up: ## Build and run the full stack with Docker (UI on http://localhost:8080)
	docker compose up --build

down: ## Stop the Docker stack
	docker compose down

clean: ## Remove caches and build output
	rm -rf backend/.pytest_cache backend/.ruff_cache backend/__pycache__ backend/.work frontend/dist

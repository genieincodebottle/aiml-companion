# Autonomous Migration Engineer - common workflows.
# Run `make help` to see targets. Backend uses uv; frontend uses npm.

.DEFAULT_GOAL := help

.PHONY: help install install-backend install-frontend dev-backend dev-frontend \
        cli-list cli-run cli-stream lab-defense test test-backend test-frontend \
        lint fmt eval up down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: install-backend install-frontend ## Install backend + frontend deps

install-backend: ## Install backend deps (incl. dev tools) with uv
	cd backend && uv sync --extra dev

install-frontend: ## Install frontend deps with npm
	cd frontend && npm install

dev-backend: ## Run the backend API on :8000 (stub mode with no key; live-sdk with one)
	cd backend && uv run uvicorn main:app --reload --port 8000

dev-frontend: ## Run the frontend dev server on :5173
	cd frontend && npm run dev

cli-list: ## List the fleet job catalog
	cd backend && uv run python cli.py list

cli-run: ## Run the datetime-fleet job through the LangGraph batch
	cd backend && uv run python cli.py run datetime-fleet

cli-stream: ## Run the live engine and print SSE events to the terminal
	cd backend && uv run python cli.py stream datetime-fleet

lab-defense: ## Teaching lab: disable guardrail layer 1, watch the reviewer catch tampering
	cd backend && ME_DISABLE_GUARDRAILS=1 uv run python cli.py run datetime-fleet

test: test-backend test-frontend ## Run all tests

test-backend: ## Run backend tests (stub worker; no key needed)
	cd backend && uv run pytest -q

test-frontend: ## Run frontend unit tests
	cd frontend && npm run test

lint: ## Lint backend (ruff) and frontend (eslint)
	cd backend && uv run ruff check .
	cd frontend && npm run lint

fmt: ## Auto-format backend (ruff) and frontend (prettier)
	cd backend && uv run ruff check . --fix
	cd frontend && npm run format

eval: ## Run the evaluation harness / promotion gate (deterministic, free in stub mode)
	cd backend && uv run python -m evaluation.run_eval

up: ## Build and run the full stack with Docker (UI on http://localhost:8080)
	docker compose up --build

down: ## Stop the Docker stack
	docker compose down

clean: ## Remove caches and build output
	rm -rf backend/.pytest_cache backend/.ruff_cache backend/__pycache__ backend/.work frontend/dist

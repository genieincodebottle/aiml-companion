.PHONY: serve test load-test docker-build docker-run lint all clean setup help

# ─── Configuration ───────────────────────────────────────────────
HOST       ?= 0.0.0.0
PORT       ?= 8000
WORKERS    ?= 1
IMAGE_NAME ?= model-server

# ─── Targets ─────────────────────────────────────────────────────

setup: ## Create demo model (run once after clone)
	python scripts/create_demo_model.py

serve: setup ## Start the model server locally
	uvicorn src.app.main:app --host $(HOST) --port $(PORT) --workers $(WORKERS)

test: ## Run unit tests
	pytest tests/ -v --tb=short

lint: ## Lint source and test code
	ruff check src/ tests/ --ignore E501

load-test: ## Run Locust load tests (headless)
	bash scripts/run_load_test.sh

docker-build: ## Build Docker image
	docker build -t $(IMAGE_NAME) -f docker/Dockerfile .

docker-run: docker-build ## Build and run Docker container
	docker run --rm -p $(PORT):8000 $(IMAGE_NAME)

clean: ## Remove caches and temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

all: lint test docker-build ## Lint, test, and build Docker image

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: help install customer-up customer-seed check test eval lint clean

## FDE Engagement Starter
## `make check` prints the rubric. On a fresh clone most of it is red. That is correct.

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed -e 's/## //'
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

customer-seed: ## Regenerate the customer's messy data (deterministic)
	python customer/seed.py

customer-up: ## Start the mock legacy dispatch API on :8000
	uvicorn customer.legacy_api.server:app --port 8000 --reload

check: ## Run the rubric - the definition of done
	@python tests/rubric.py

test: ## Run all tests verbosely
	pytest tests/ -v --tb=short

eval: ## Run the evaluation gate
	python eval/run_eval.py

lint: ## Byte-compile everything (no network, no API key)
	python -m compileall -q src customer eval tests deploy
	@echo "compiles clean"

clean: ## Remove caches and generated artifacts
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f audit_log.jsonl

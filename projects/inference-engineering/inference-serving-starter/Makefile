.PHONY: help install serve check test experiments incidents diagnose lint clean

## Inference Serving Starter
## `make check` prints six gates. On a fresh clone one is green. That is correct.

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed -e 's/## //'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

serve: ## Start engine_sim (Llama-3.1-8B bf16 on an L4) on :8100
	python -m engine_sim --gpu l4 --quant bf16 --port 8100

check: ## Run the six gates - the definition of done
	@python tests/rubric.py

test: ## Run all tests verbosely
	pytest tests/ -v --tb=short

experiments: ## Run the quantisation and speculation sweeps (needs bench/ implemented)
	python scripts/run_experiments.py

incidents: ## Regenerate the incident bundles (deterministic)
	python debug_lab/make_incidents.py

diagnose: ## Check your three diagnoses
	python debug_lab/check_diagnosis.py

lint: ## Byte-compile everything
	python -m compileall -q engine_sim bench analysis debug_lab tests scripts
	@echo "compiles clean"

clean: ## Remove caches
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

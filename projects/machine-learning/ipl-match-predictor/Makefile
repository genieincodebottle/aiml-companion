# ============================================================================
# Makefile - IPL Match Predictor
# ============================================================================
# Targets:
#   make run        - run the full ML pipeline (CLI)
#   make notebook   - execute the analysis notebook end-to-end
#   make test       - run the pytest test suite
#   make clean      - remove generated artifacts
#   make install    - install Python dependencies
#   make help       - show available targets
# ============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

PROJECT_ROOT := $(shell pwd)
NOTEBOOK     := notebooks/ipl-dataset-analysis.ipynb
PYTHON       := python
PIP          := pip
PYTEST       := pytest

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@echo "IPL Match Predictor - available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""

.PHONY: install
install: ## Install Python dependencies from requirements.txt
	$(PIP) install -r requirements.txt

.PHONY: run
run: ## Run the full ML pipeline (load, clean, EDA, features, train, evaluate)
	$(PYTHON) main.py

.PHONY: run-stage
run-stage: ## Run a specific stage (usage: make run-stage STAGE=eda)
	$(PYTHON) main.py --stage $(STAGE)

.PHONY: notebook
notebook: ## Execute the analysis notebook and produce an HTML report
	@bash scripts/run_analysis.sh --no-open

.PHONY: test
test: ## Run the pytest test suite
	$(PYTEST) tests/ -v --tb=short

.PHONY: clean
clean: ## Remove generated artifacts and caches
	rm -rf artifacts/results/ipl-dataset-analysis-executed.ipynb
	rm -rf artifacts/results/ipl-dataset-analysis.html
	rm -rf artifacts/results/evaluation_report.md
	rm -rf artifacts/figures/*.png artifacts/figures/*.svg
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
# ============================================================================
# Algorithm Showdown - Makefile
# Medical Diagnostic Classifier (UCI Breast Cancer, 6 algorithms, SHAP)
# ============================================================================

.PHONY: train evaluate explain test all clean help

PYTHON ?= python

# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------

train:  ## Train 6 algorithms and save best model to artifacts/models/
	@mkdir -p artifacts/models artifacts/results
	$(PYTHON) -m src.train

evaluate:  ## Evaluate saved model with confusion matrix and threshold analysis
	$(PYTHON) -m src.evaluate

explain:  ## Generate SHAP explainability report
	$(PYTHON) -m src.explain

# --------------------------------------------------------------------------
# Testing
# --------------------------------------------------------------------------

test:  ## Run pytest test suite
	$(PYTHON) -m pytest tests/ -v

# --------------------------------------------------------------------------
# Composite targets
# --------------------------------------------------------------------------

all: train evaluate explain  ## Run full pipeline: train -> evaluate -> explain

pipeline: all test  ## Run full pipeline and tests

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

clean:  ## Remove generated artifacts (models, caches)
	rm -rf artifacts/models/*.joblib
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache

install:  ## Install project dependencies
	pip install -r requirements.txt

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

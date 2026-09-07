# =============================================================================
# Makefile - Visual Defect Triage
# =============================================================================
#   make demo      End-to-end run on synthetic embeddings. No downloads.
#   make test      Unit tests
#   make serve     Start the API (needs the model extras)
#   make install   Install dependencies
#   make clean     Remove caches
# =============================================================================

.PHONY: demo test serve install clean help all

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

all: test demo

demo:
	@echo ">>> Generating synthetic embeddings..."
	$(PYTHON) -m scripts.make_synthetic_data
	@echo ">>> Running the full triage pipeline..."
	$(PYTHON) -m src.run_pipeline
	@echo ""
	@echo "  Report:      artifacts/report.md"
	@echo "  Slices:      artifacts/slice_report.csv"
	@echo "  Thresholds:  artifacts/threshold_sweep.csv"

test:
	@echo ">>> Running unit tests..."
	$(PYTEST) tests/ -v --tb=short

serve:
	$(PYTHON) -m uvicorn api.main:app --port 8000 --reload

install:
	uv pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo "make demo | test | serve | install | clean"

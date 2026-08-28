# =============================================================================
# Makefile - CIFAR-10 Progressive Classifier
# =============================================================================
# Targets:
#   make train        Run the training pipeline
#   make diagnostics  Run the diagnostics toolkit
#   make test         Run unit tests with pytest
#   make all          Run tests, train, and diagnostics (full pipeline)
#   make install      Install Python dependencies
#   make clean        Remove cached files and __pycache__
#   make help         Show available targets
# =============================================================================

.PHONY: train diagnostics test all install clean help

PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

# Default target
all: test train diagnostics
	@echo ""
	@echo "========================================"
	@echo "  Full pipeline complete."
	@echo "  Checkpoints:    artifacts/checkpoints/"
	@echo "  Results:        artifacts/results/"
	@echo "  Experiment log: docs/experiment_log.csv"
	@echo "========================================"

train:
	@echo ">>> Training CIFAR-10 progressive classifier..."
	$(PYTHON) -m src.train

diagnostics:
	@echo ">>> Running diagnostics toolkit..."
	$(PYTHON) -m src.diagnostics

test:
	@echo ">>> Running unit tests..."
	$(PYTEST) tests/ -v --tb=short

install:
	@echo ">>> Installing dependencies..."
	uv pip install -r requirements.txt

clean:
	@echo ">>> Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

help:
	@echo "Available targets:"
	@echo "  make train        - Run the training pipeline"
	@echo "  make diagnostics  - Run the diagnostics toolkit"
	@echo "  make test         - Run unit tests with pytest"
	@echo "  make all          - Run tests, train, and diagnostics (full pipeline)"
	@echo "  make install      - Install Python dependencies"
	@echo "  make clean        - Remove cached files and __pycache__"
	@echo "  make help         - Show this help message"

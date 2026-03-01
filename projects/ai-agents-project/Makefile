# ============================================
# Multi-Agent Research System — Makefile
# ============================================
# Usage:
#   make run       — Run the multi-agent research pipeline
#   make evaluate  — Run single-agent vs multi-agent evaluation
#   make test      — Run unit tests with pytest
#   make ui        — Launch the Streamlit UI
#   make all       — Run tests, then the pipeline, then evaluation

.PHONY: all run evaluate test ui clean help

# Default target
all: test run evaluate

# --- Run the multi-agent research pipeline ---
run:
	@echo "Running multi-agent research pipeline..."
	python -m src.agents

# --- Run the evaluation framework ---
evaluate:
	@echo "Running evaluation: single-agent vs multi-agent..."
	python -m evaluation.run_eval

# --- Launch the Streamlit UI ---
ui:
	@echo "Launching Streamlit UI..."
	streamlit run app.py

# --- Run tests ---
test:
	@echo "Running tests..."
	pytest tests/ -v

# --- Clean generated artifacts ---
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# --- Help ---
help:
	@echo "Available targets:"
	@echo "  make run       — Run the multi-agent research pipeline"
	@echo "  make evaluate  — Run single-agent vs multi-agent evaluation"
	@echo "  make test      — Run unit tests with pytest"
	@echo "  make ui        — Launch the Streamlit UI"
	@echo "  make all       — Run tests, then pipeline, then evaluation"
	@echo "  make clean     — Remove __pycache__ and .pyc files"
	@echo "  make help      — Show this help message"

# ============================================================
# Makefile for RAG Expert Assistant
# ============================================================

.PHONY: run evaluate ab-test security test all clean help

# Default target
help:
	@echo "RAG Expert Assistant - Available targets:"
	@echo ""
	@echo "  make run        - Run the RAG pipeline (ingest, chunk, embed, retrieve, generate)"
	@echo "  make evaluate   - Run RAGAS evaluation suite"
	@echo "  make ab-test    - Run A/B comparison (naive vs optimized)"
	@echo "  make security   - Run security test suite (injection + PII)"
	@echo "  make test       - Run all unit tests with pytest"
	@echo "  make all        - Run full pipeline: test -> run -> evaluate -> security"
	@echo "  make clean      - Remove generated artifacts and caches"
	@echo ""

# Run the RAG pipeline
run:
	python -m src.rag_pipeline

# Run RAGAS evaluation
evaluate:
	python -m src.evaluate

# Run A/B comparison
ab-test:
	python -m src.ab_comparison

# Run security tests
security:
	python -m src.security.sanitizer

# Run unit tests
test:
	pytest tests/ -v

# Run everything
all: test run evaluate security
	@echo ""
	@echo "============================================================"
	@echo "  All targets completed successfully."
	@echo "============================================================"

# Clean generated files
clean:
	rm -rf chroma_db/
	rm -rf __pycache__/ src/__pycache__/ src/security/__pycache__/ tests/__pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned generated files."

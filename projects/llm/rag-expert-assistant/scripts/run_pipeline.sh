#!/usr/bin/env bash
# ============================================================
# Run the RAG Expert Assistant pipeline
# Usage: bash scripts/run_pipeline.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Check for .env file
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy .env.example to .env and add your API keys."
    echo "  cp .env.example .env"
    exit 1
fi

# Check for virtual environment
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "WARNING: No virtual environment detected. Consider activating one:"
    echo "  python -m venv venv && source venv/bin/activate"
fi

echo "============================================================"
echo "  RAG Expert Assistant - Pipeline Runner"
echo "============================================================"
echo ""

# Step 1: Run the RAG pipeline (ingest, chunk, embed, retrieve, generate)
echo "[1/2] Running RAG pipeline..."
python -m src.rag_pipeline
echo ""

# Step 2: Run security tests
echo "[2/2] Running security tests..."
python -m src.security.sanitizer
echo ""

echo "============================================================"
echo "  Pipeline complete. Check output above for results."
echo "============================================================"

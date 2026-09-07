#!/usr/bin/env bash
# ============================================================
# Run RAG evaluation suite (RAGAS + A/B comparison)
# Usage: bash scripts/run_evaluation.sh
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
echo "  RAG Expert Assistant - Evaluation Suite"
echo "============================================================"
echo ""

# Step 1: RAGAS evaluation
echo "[1/2] Running RAGAS evaluation (requires Google API key)..."
python -m src.evaluate
echo ""

# Step 2: A/B comparison (naive vs optimized)
echo "[2/2] Running A/B comparison..."
python -m src.ab_comparison
echo ""

echo "============================================================"
echo "  Evaluation complete. Reports saved to artifacts/results/"
echo "============================================================"

#!/usr/bin/env bash
# ============================================
# Run the Evaluation Framework
# ============================================
# Usage: bash scripts/run_evaluation.sh
#   or:  make evaluate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# --- Check for .env file ---
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found."
    echo "Copy .env.example to .env and fill in your API keys:"
    echo "  cp .env.example .env"
    exit 1
fi

# --- Check for virtual environment ---
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "WARNING: No virtual environment detected."
    echo "Consider activating one: source venv/bin/activate"
fi

# --- Run evaluation ---
echo "=========================================="
echo " Agent Evaluation: Single vs Multi-Agent"
echo "=========================================="
echo ""
echo "Running head-to-head comparison on test questions..."
echo "This will make API calls and may take a few minutes."
echo ""

python -m evaluation.run_eval

echo ""
echo "Evaluation complete. See artifacts/results/evaluation_results.md for reference data."

#!/usr/bin/env bash
# ============================================
# Run the Multi-Agent Research Pipeline
# ============================================
# Usage: bash scripts/run_agents.sh
#   or:  make run

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

# --- Run the pipeline ---
echo "=========================================="
echo " Multi-Agent Research Pipeline"
echo "=========================================="
echo ""
echo "Starting 4-agent pipeline: researcher -> analyst -> writer -> fact-checker"
echo ""

python -m src.agents

echo ""
echo "Pipeline complete. Check artifacts/results/ for output."

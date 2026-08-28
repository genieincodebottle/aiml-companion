#!/usr/bin/env bash
# =============================================================================
# run_training.sh - One-command full training pipeline
# =============================================================================
# Runs the CIFAR-10 progressive training pipeline end-to-end:
#   1. Installs dependencies
#   2. Runs unit tests
#   3. Trains the model
#   4. Runs diagnostics
#
# Usage:
#   bash scripts/run_training.sh
#   bash scripts/run_training.sh --skip-tests
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  CIFAR-10 Progressive Training Pipeline"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# --- Parse arguments ---
SKIP_TESTS=false
for arg in "$@"; do
    case $arg in
        --skip-tests) SKIP_TESTS=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- Step 1: Check dependencies ---
echo "[1/4] Checking dependencies..."
if ! python -c "import torch" 2>/dev/null; then
    echo "  Installing requirements..."
    uv pip install -r requirements.txt
else
    echo "  Dependencies already installed."
fi
echo ""

# --- Step 2: Run tests ---
if [ "$SKIP_TESTS" = false ]; then
    echo "[2/4] Running unit tests..."
    python -m pytest tests/ -v --tb=short
    echo ""
else
    echo "[2/4] Skipping tests (--skip-tests flag)."
    echo ""
fi

# --- Step 3: Train the model ---
echo "[3/4] Training CIFAR-10 progressive classifier..."
echo "  Config: configs/base.yaml"
echo "  Checkpoints: artifacts/checkpoints/"
echo ""
python -m src.train
echo ""

# --- Step 4: Run diagnostics ---
echo "[4/4] Running diagnostics toolkit..."
python -m src.diagnostics
echo ""

echo "============================================================"
echo "  Pipeline Complete"
echo "============================================================"
echo "  Checkpoints:     artifacts/checkpoints/"
echo "  Results:         artifacts/results/"
echo "  Experiment log:  docs/experiment_log.csv"
echo "============================================================"

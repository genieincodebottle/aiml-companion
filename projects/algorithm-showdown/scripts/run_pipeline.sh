#!/usr/bin/env bash
# ============================================================================
# Algorithm Showdown - Full Pipeline Runner
# Executes: train -> evaluate -> explain
#
# Usage:
#   bash scripts/run_pipeline.sh          # Run full pipeline
#   bash scripts/run_pipeline.sh train    # Run train only
#   bash scripts/run_pipeline.sh evaluate # Run evaluate only
#   bash scripts/run_pipeline.sh explain  # Run explain only
# ============================================================================

set -euo pipefail

# Resolve project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_step() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] === $1 ===${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')] $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR: $1${NC}"
}

# Ensure artifact directories exist
mkdir -p artifacts/models artifacts/results

run_train() {
    log_step "STEP 1/3: Training 6 algorithms"
    python -m src.train
    log_success "Training complete. Model saved to artifacts/models/"
}

run_evaluate() {
    log_step "STEP 2/3: Evaluating best model"
    python -m src.evaluate
    log_success "Evaluation complete."
}

run_explain() {
    log_step "STEP 3/3: Generating SHAP explanations"
    python -m src.explain
    log_success "Explainability report complete."
}

run_all() {
    log_step "Starting full pipeline"
    echo ""
    run_train
    echo ""
    run_evaluate
    echo ""
    run_explain
    echo ""
    log_success "Full pipeline finished successfully."
}

# Parse command-line argument
STAGE="${1:-all}"

case "${STAGE}" in
    train)
        run_train
        ;;
    evaluate)
        run_evaluate
        ;;
    explain)
        run_explain
        ;;
    all)
        run_all
        ;;
    *)
        log_error "Unknown stage: ${STAGE}"
        echo "Usage: $0 {train|evaluate|explain|all}"
        exit 1
        ;;
esac

#!/usr/bin/env bash
# ============================================================================
# run_analysis.sh - Execute the IPL Dataset Analysis notebook end-to-end
# ============================================================================
#
# Usage:
#   bash scripts/run_analysis.sh            # run with default settings
#   bash scripts/run_analysis.sh --no-open  # run without opening the HTML output
#
# Prerequisites:
#   - Python virtual environment activated with dependencies installed
#   - jupyter and nbconvert available on PATH

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NOTEBOOK="${PROJECT_ROOT}/notebooks/ipl-dataset-analysis.ipynb"
OUTPUT_DIR="${PROJECT_ROOT}/artifacts/results"
FIGURES_DIR="${PROJECT_ROOT}/artifacts/figures"

# Parse arguments
OPEN_AFTER=true
for arg in "$@"; do
    case $arg in
        --no-open) OPEN_AFTER=false ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "=========================================="
echo " IPL Dataset Analysis - Pipeline Runner"
echo "=========================================="
echo ""

if ! command -v jupyter &>/dev/null; then
    echo "ERROR: 'jupyter' not found. Install with: pip install jupyter"
    exit 1
fi

if [ ! -f "${NOTEBOOK}" ]; then
    echo "ERROR: Notebook not found at ${NOTEBOOK}"
    exit 1
fi

# Ensure output directories exist
mkdir -p "${OUTPUT_DIR}" "${FIGURES_DIR}"

# ---------------------------------------------------------------------------
# Execute notebook
# ---------------------------------------------------------------------------
echo "[1/3] Executing notebook: ${NOTEBOOK}"
echo "      This may take a few minutes..."
echo ""

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output-dir="${OUTPUT_DIR}" \
    --output="ipl-dataset-analysis-executed.ipynb" \
    "${NOTEBOOK}"

echo ""
echo "[2/3] Converting executed notebook to HTML..."

jupyter nbconvert \
    --to html \
    --output-dir="${OUTPUT_DIR}" \
    --output="ipl-dataset-analysis.html" \
    "${OUTPUT_DIR}/ipl-dataset-analysis-executed.ipynb"

echo ""
echo "[3/3] Done."
echo ""
echo "Outputs:"
echo "  Executed notebook : ${OUTPUT_DIR}/ipl-dataset-analysis-executed.ipynb"
echo "  HTML report       : ${OUTPUT_DIR}/ipl-dataset-analysis.html"
echo ""

# ---------------------------------------------------------------------------
# Optionally open the HTML report
# ---------------------------------------------------------------------------
if [ "${OPEN_AFTER}" = true ]; then
    HTML_PATH="${OUTPUT_DIR}/ipl-dataset-analysis.html"
    if command -v xdg-open &>/dev/null; then
        xdg-open "${HTML_PATH}"
    elif command -v open &>/dev/null; then
        open "${HTML_PATH}"
    elif command -v start &>/dev/null; then
        start "${HTML_PATH}"
    else
        echo "Open ${HTML_PATH} in your browser to view the report."
    fi
fi

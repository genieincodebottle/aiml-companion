#!/bin/bash
# Run the full due diligence pipeline
# Usage: ./scripts/run_pipeline.sh "Tesla" [standard|quick|deep]

set -e

COMPANY="${1:?Usage: $0 <company_name> [depth]}"
DEPTH="${2:-standard}"
OUTPUT="artifacts/reports/${COMPANY// /_}_$(date +%Y%m%d_%H%M%S).md"

echo "============================================="
echo "  Due Diligence Agent"
echo "  Company: $COMPANY"
echo "  Depth: $DEPTH"
echo "  Output: $OUTPUT"
echo "============================================="

# Check for API key
if [ -z "$GOOGLE_API_KEY" ] && [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "ERROR: GOOGLE_API_KEY not set. Get one at https://aistudio.google.com/apikey"
    exit 1
fi

mkdir -p artifacts/reports

python main.py --company "$COMPANY" --depth "$DEPTH" --output "$OUTPUT"

echo ""
echo "Report saved to: $OUTPUT"

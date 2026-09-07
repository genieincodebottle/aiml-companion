#!/usr/bin/env bash
# Run Locust load tests against the model server.
# Usage: bash scripts/run_load_test.sh [--host HOST] [--users N] [--rate N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

TARGET_HOST="${HOST:-http://localhost:8000}"
USERS="${USERS:-100}"
SPAWN_RATE="${SPAWN_RATE:-10}"
RUN_TIME="${RUN_TIME:-5m}"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) TARGET_HOST="$2"; shift 2 ;;
        --users) USERS="$2"; shift 2 ;;
        --rate) SPAWN_RATE="$2"; shift 2 ;;
        --time) RUN_TIME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Running load test against ${TARGET_HOST}"
echo "Users: ${USERS}, Spawn rate: ${SPAWN_RATE}/s, Duration: ${RUN_TIME}"

cd "$PROJECT_ROOT"

locust \
    -f tests/load/locustfile.py \
    --host "$TARGET_HOST" \
    --users "$USERS" \
    --spawn-rate "$SPAWN_RATE" \
    --run-time "$RUN_TIME" \
    --headless

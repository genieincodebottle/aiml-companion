#!/usr/bin/env bash
# Start the ML model server using uvicorn.
# Usage: bash scripts/run_server.sh [--port PORT]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
WORKERS="${WORKERS:-1}"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Starting model server on ${HOST}:${PORT} (workers=${WORKERS})"
cd "$PROJECT_ROOT"

uvicorn src.app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    --workers "$WORKERS"

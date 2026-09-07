#!/bin/bash
# Hook: PreToolUse (Bash) - Block accidental deletion of SQLite DBs and audit logs.
# Reads JSON from stdin with tool_input.command

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

# Block direct deletion of SQLite databases
if echo "$COMMAND" | grep -qE '(rm|del).*\.(db|sqlite)'; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Blocked: use `python scripts/clean_data.py` instead of deleting DB files directly."}}'
  exit 0
fi

# Block deletion of audit logs
if echo "$COMMAND" | grep -qE '(rm|del).*audit_log'; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Blocked: audit logs are retained for compliance (7 years). Do not delete."}}'
  exit 0
fi

exit 0
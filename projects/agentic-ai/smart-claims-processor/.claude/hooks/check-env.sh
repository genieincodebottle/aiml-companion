#!/bin/bash
# Hook: UserPromptSubmit (once per session) - Warn if .env is missing or has no API key.

if [ ! -f .env ]; then
  echo "Warning: .env file not found. Run: cp .env.example .env" >&2
  exit 2
fi

# Check that at least one LLM API key is set
GROQ_KEY=$(grep -E '^GROQ_API_KEY=.+' .env 2>/dev/null | cut -d= -f2-)
GOOGLE_KEY=$(grep -E '^GOOGLE_API_KEY=.+' .env 2>/dev/null | cut -d= -f2-)

if [ -z "$GROQ_KEY" ] && [ -z "$GOOGLE_KEY" ]; then
  echo "Warning: No LLM API key found in .env. Set GROQ_API_KEY or GOOGLE_API_KEY." >&2
  exit 2
fi

exit 0
#!/usr/bin/env python3
"""PreToolUse hook: block writes to secrets and local database state.

Claude Code passes the tool call as JSON on stdin. Exit code 2 rejects the call,
and the message printed to stderr is fed back to Claude so it knows why. This is
defense in depth: CLAUDE.md asks for the same thing in prose, this enforces it.
"""
import json
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0  # Never break the tool call over a parse error.

    path = (data.get("tool_input") or {}).get("file_path", "").replace("\\", "/")
    if not path:
        return 0

    low = path.lower()
    if low.endswith(".env") or "/.env" in low or ".env." in low:
        print("Blocked: never write .env files. Use .env.example and keep real keys local.", file=sys.stderr)
        return 2
    if "/backend/databases/" in low:
        print("Blocked: backend/databases/ is local state, not source. Do not edit it.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
---
name: test
description: Run pytest tests - all tests, a specific file, or by keyword match
argument-hint: "[agent-name or test file] (optional, runs all if omitted)"
allowed-tools: Bash(pytest *)
---

# Run Tests

Run the project test suite. Tests use mocked LLMs and default to US country profile.

If `$ARGUMENTS` is provided:
- If it looks like a file path (contains `/` or `.py`), run that file directly
- Otherwise treat it as a keyword filter: `pytest tests/ -q -k "$ARGUMENTS"`

If no arguments, run the full suite:

```bash
pytest tests/ -q
```

Report pass/fail counts and any failures with file:line references.
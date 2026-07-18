#!/usr/bin/env python3
"""PostToolUse hook: format the file that was just edited, if a formatter is installed.

Claude Code passes the tool call as JSON on stdin. This runs after every Edit/Write,
so the tool never leaves unformatted code behind. It exits quietly when a formatter is
missing, so the hook is safe to commit even if a teammate has not installed the tools.
"""
import json
import os
import shutil
import subprocess
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    path = (data.get("tool_input") or {}).get("file_path", "")
    if not path or not os.path.isfile(path):
        return 0

    ext = os.path.splitext(path)[1].lower()
    if ext == ".py" and shutil.which("ruff"):
        subprocess.run(["ruff", "format", path], capture_output=True)
    elif ext in {".js", ".jsx", ".ts", ".tsx", ".json", ".css"} and shutil.which("prettier"):
        subprocess.run(["prettier", "--write", path], capture_output=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
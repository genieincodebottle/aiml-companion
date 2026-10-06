#!/usr/bin/env python
"""Zero-install entry point.

    python run.py all

Puts src/ on the path and calls the same CLI, so nothing needs `pip install -e .`.
    python run.py --help
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MISSING_HINT = """
Missing dependency: {name}

Install everything this project needs with:

    uv pip install -r requirements.txt

(from the project root: {root})
"""

if __name__ == "__main__":
    try:
        from star_schema.cli import main
    except ModuleNotFoundError as e:  # a dependency, not our package
        print(MISSING_HINT.format(name=e.name, root=SRC.parent), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())

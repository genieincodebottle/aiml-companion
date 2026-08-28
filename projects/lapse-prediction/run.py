#!/usr/bin/env python
"""Zero-install entry point.

    python run.py train --model ordinal_chain

Works straight from a fresh clone with no `pip install` of this project at all.
It puts `src/` on the path and calls the same CLI. Use this if the `lapse`
console script is not on your PATH, or if `pip install -e .` failed (which it
does on some Windows setups, with a stale entry-point .exe).

Everything else is identical:
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

    pip install -r requirements.txt

(from the project root: {root})
"""

if __name__ == "__main__":
    try:
        from lapse_prediction.cli import main
    except ModuleNotFoundError as e:  # a dependency, not our package
        print(MISSING_HINT.format(name=e.name, root=SRC.parent), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main())

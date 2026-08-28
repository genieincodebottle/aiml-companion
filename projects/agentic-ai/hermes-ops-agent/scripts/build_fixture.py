#!/usr/bin/env python3
"""Compile fixtures/state.fixture.sql into fixtures/state.fixture.db.

You rarely need to run this by hand: `main.py` and the test suite both build
the fixture automatically if it is missing. It exists for `make fixture` and
for forcing a rebuild after editing the SQL.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fixture import build, FixtureError  # noqa: E402


def main() -> int:
    try:
        db = build()
    except FixtureError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"built {db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

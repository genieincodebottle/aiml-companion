"""Run notebook cells in one shared namespace, the way a notebook runs them.

Usage: python run_steps.py step4 step5 ...   (names of files in cells/, without .py)
Prints each cell's output under a header. Exits non-zero on the first error.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(names):
    ns = {"__name__": "__main__"}
    for name in names:
        path = HERE / "cells" / (name + ".py")
        print("=" * 8, name)
        t = time.perf_counter()
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
        print(f"[{name} finished in {time.perf_counter() - t:.1f} s]")
    return ns


if __name__ == "__main__":
    main(sys.argv[1:])

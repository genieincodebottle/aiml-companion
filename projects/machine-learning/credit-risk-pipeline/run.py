#!/usr/bin/env python
"""Zero-install entry point.

    python run.py              # the whole pipeline, start to finish
    python run.py train        # stop after training
    python run.py --help       # every stage, with descriptions

Why this file exists, when main.py is right there: the documented way in used
to be `make run`, and `make` is not installed on a default Windows box. The
Makefile also hardcoded `.venv/Scripts/pip`, a Windows path inside a tool that
only runs on Unix, so it worked on neither platform. This script needs nothing
but Python, runs from any working directory, and names the missing package when
a dependency is absent instead of throwing a bare ImportError.

Stages run as a prefix: asking for `train` runs clean, eda and features first,
because each one consumes what the previous one produced.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STAGES = ("clean", "eda", "features", "train", "evaluate", "serve")

MISSING_HINT = """
Missing dependency: {name}

Install everything this project needs with:

    pip install -r requirements.txt

(from the project root: {root})
"""

USAGE = """usage: python run.py [stage] [--verbose]

Stages (each one runs the stages before it):
    clean       load the German Credit data and clean it        (~2s)
    eda         exploratory charts into artifacts/figures/      (~5s)
    features    engineer domain features                        (~1s)
    train       LogisticRegression + GradientBoosting, 5-fold CV (~15s)
    evaluate    cost-sensitive threshold, SHAP, written report   (~8s)
    serve       FastAPI prediction endpoint (needs a trained model)

With no stage, the full pipeline runs: about 30 seconds end to end.
"""


def main() -> int:
    args = [a for a in sys.argv[1:]]
    if "--help" in args or "-h" in args:
        print(USAGE)
        return 0

    verbose = "--verbose" in args
    positional = [a for a in args if not a.startswith("-")]

    if len(positional) > 1:
        print(f"Expected at most one stage, got: {positional}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    stage = positional[0] if positional else None
    if stage is not None and stage not in STAGES:
        print(f"Unknown stage: {stage!r}", file=sys.stderr)
        print(f"Choose one of: {', '.join(STAGES)}", file=sys.stderr)
        return 2

    # main.py owns the pipeline. This file only makes it reachable.
    forwarded = ["main.py"]
    if stage:
        forwarded += ["--stage", stage]
    if verbose:
        forwarded += ["--verbose"]
    sys.argv = forwarded

    try:
        import main as pipeline
    except ModuleNotFoundError as e:
        print(MISSING_HINT.format(name=e.name, root=ROOT), file=sys.stderr)
        return 2

    pipeline.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

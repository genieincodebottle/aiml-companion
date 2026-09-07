#!/usr/bin/env python
"""One entry point for this project, on any operating system.

The repo shipped two bash scripts, `scripts/run_pipeline.sh` and
`scripts/run_evaluation.sh`. Windows has no bash, so on the platform a large
share of learners use, the documented way to run this project did not run.

Both scripts also checked only that a `.env` file EXISTS. That is not the same
as having a key: `.env.example` ships

    GOOGLE_API_KEY=your-google-api-key-here

and a placeholder is a perfectly truthy string. Copy the example, run the
script, and the check passes, the pipeline starts, and it dies somewhere deep
inside an HTTP 400 about an invalid API key. `has_real_key` below rejects
placeholders so the failure arrives at the top with an instruction attached.

    python run.py test        # 32 tests, no API key needed
    python run.py security    # injection + PII suite, no API key needed
    python run.py pipeline    # index the corpus and answer one question
    python run.py eval        # RAGAS scores on the fixed eval set
    python run.py ab          # measured naive vs optimized comparison
"""

import argparse
import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

# Substrings that mark a value as a placeholder rather than a credential.
_PLACEHOLDER_MARKERS = ("your-", "your_", "xxx", "changeme", "replace-me", "here")


def has_real_key(name: str = "GOOGLE_API_KEY") -> bool:
    """True only for something that looks like an actual credential."""
    value = (os.getenv(name) or "").strip().strip('"').strip("'")
    if len(value) < 12:
        return False
    return not any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS)


def require_key() -> None:
    if has_real_key():
        return
    print(
        "\nThis command calls the Gemini API and no usable GOOGLE_API_KEY was "
        "found.\n\n"
        "  1. Get a free key at https://aistudio.google.com/app/apikey\n"
        "  2. cp .env.example .env      (Windows: copy .env.example .env)\n"
        "  3. Replace 'your-google-api-key-here' with the real key\n\n"
        "Copying .env.example without editing it is not enough: the "
        "placeholder\n"
        "is a non-empty string, so it passes an 'is the key set?' check and "
        "then\nfails later as an HTTP 400.\n\n"
        "No key yet? `python run.py test` and `python run.py security` run the "
        "full\ntest suite and the security suite without one.\n",
        file=sys.stderr,
    )
    sys.exit(1)


def _run(args: list) -> int:
    return subprocess.call([sys.executable] + args)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "command",
        choices=["test", "security", "pipeline", "eval", "ab"],
        help="test/security need no API key; pipeline/eval/ab do",
    )
    args = ap.parse_args()

    if args.command == "test":
        return _run(["-m", "pytest", "tests/", "src/security/test_security.py", "-v"])

    if args.command == "security":
        return _run(["-m", "src.security.sanitizer"])

    require_key()

    if args.command == "pipeline":
        return _run(["-m", "src.rag_pipeline"])
    if args.command == "eval":
        return _run(["-m", "src.evaluate"])
    if args.command == "ab":
        print("This runs 10 questions through two full pipelines and scores "
              "both with RAGAS.\nIt makes a few hundred API calls and takes "
              "several minutes.\n")
        return _run(["-m", "src.ab_comparison"])

    return 1


if __name__ == "__main__":
    sys.exit(main())

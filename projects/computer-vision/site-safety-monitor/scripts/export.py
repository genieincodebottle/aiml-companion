"""PyTorch to ONNX to TensorRT, with the engine count checked rather than
assumed."""
from __future__ import annotations

import re
import subprocess
import sys

#: A boundary between engines copies the activation out of the accelerator
#: and back. Measured on this graph at 640x640.
ACTIVATION_MB = 3.28
COPY_MS = 0.546


def parse_log(log: str) -> dict:
    """Separated from the subprocess call so it is testable on a fixture.

    The recorded log in tests/fixtures is a real partitioned build. Without
    it this check would only ever be exercised on the day it fires.
    """
    # Count completions only. "Building engine" and "Engine built" both
    # appear for every subgraph, so matching either double counts.
    engines = len(re.findall(r"Engine built", log))
    fallbacks = set(re.findall(r"falling back to (\w+)", log))
    boundaries = max(engines - 1, 0)
    return {
        "engines": engines,
        "fallbacks": fallbacks,
        "boundaries": boundaries,
        # Each boundary copies the activation BOTH ways.
        "copy_overhead_ms": boundaries * 2 * COPY_MS,
        "partitioned": engines > 1,
    }


def export_and_check(onnx_path: str, engine_path: str, log: str | None = None):
    """A partitioned graph is not an error, so nothing fails on its own.

    trtexec happily reports success while producing three engines. The
    only way to notice is to read the log for the partition count.
    """
    if log is None:
        log = subprocess.run(
            ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}",
             "--fp16", "--verbose"],
            capture_output=True, text=True,
        ).stderr

    info = parse_log(log)
    if info["partitioned"]:
        print(f"REFUSING: graph split into {info['engines']} engines",
              file=sys.stderr)
        print(f"  unsupported operators: {info['fallbacks']}", file=sys.stderr)
        print(f"  copy overhead: {info['copy_overhead_ms']:.2f} ms per frame",
              file=sys.stderr)
        sys.exit(1)
    return info


if __name__ == "__main__":
    export_and_check(sys.argv[1], sys.argv[2])

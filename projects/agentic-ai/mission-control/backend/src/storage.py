"""Crash-safe file writes shared by checkpoints and the prompt registry."""

from __future__ import annotations

import os
import time
from pathlib import Path


def atomic_write(path: Path, text: str, attempts: int = 20) -> None:
    """Write `text` to `path` so a reader never sees a half-written file.

    Write to a temp file, then rename it over the target. On Windows that rename fails
    with "Access is denied" while ANY reader has the target open (the console reads a
    case's checkpoint while its worker writes it), so the rename is retried briefly.
    POSIX renames over open files and succeeds on the first try.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(attempts):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.01 * (attempt + 1))

"""Write the generated world to disk. Raw CSVs go to one folder and the answer
key to another, so the build path can be pointed at the first and never see the
second."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from star_schema.config import load_config, resolve
from star_schema.data.world import build_world
from star_schema.evaluation.answer_key import write_key


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" plus an explicit "\n" terminator keeps the bytes identical on Windows.
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_world(cfg: dict | None = None, raw_dir: Path | None = None,
                key_dir: Path | None = None) -> dict[str, str]:
    """Generate the world and write it. Returns {file name: sha256} for the raw files."""
    cfg = cfg or load_config()
    raw_dir = raw_dir or resolve(cfg, "raw_dir")
    key_dir = key_dir or resolve(cfg, "key_dir")
    world = build_world(cfg)
    hashes = {}
    for table, (header, rows) in world["raw"].items():
        path = raw_dir / f"{table}.csv"
        write_csv(path, header, rows)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_key(key_dir, world)
    return hashes

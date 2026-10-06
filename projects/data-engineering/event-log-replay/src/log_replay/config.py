"""Config loading and the on-disk locations every command shares."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "conf" / "config.yaml"
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
LOG_DIR = DATA_DIR / "log"
MANIFEST_PATH = DATA_DIR / "manifest.json"
ANSWER_KEY_PATH = DATA_DIR / "answer_key.json"


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)

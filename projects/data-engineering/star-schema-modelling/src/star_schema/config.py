"""Configuration. One YAML file, loaded once, plus the project paths."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "conf" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(cfg: dict, key: str) -> Path:
    """Turn a relative path from the config into an absolute one under ROOT."""
    return ROOT / cfg["paths"][key]

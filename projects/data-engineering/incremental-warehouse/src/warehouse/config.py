"""Config loading and the folders the commands write to."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"


def load_config(path=None):
    path = Path(path) if path else ROOT / "conf" / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg["pipeline"]["lookback_minutes"] <= cfg["source"]["late_max_minutes"]:
        raise ValueError("pipeline.lookback_minutes must exceed source.late_max_minutes, "
                         "or late commits fall outside the window by construction")
    return cfg

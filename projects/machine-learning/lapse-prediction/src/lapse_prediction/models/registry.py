"""Model persistence with provenance.

A pickled estimator on its own is a liability: six months later nobody can say
what data trained it, which code produced it, or how well it scored. Every save
here carries a metadata sidecar -- git sha, config, data fingerprint, feature
list, validation metrics -- and `load()` gives both back together.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from lapse_prediction.config import CFG, Config
from lapse_prediction.utils.logging import get_logger

log = get_logger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def data_fingerprint(df: pd.DataFrame) -> str:
    """Cheap, stable identifier for the training data: shape + hashed content."""
    h = pd.util.hash_pandas_object(df, index=False).sum()
    return f"{len(df)}x{df.shape[1]}:{h & 0xFFFFFFFF:08x}"


@dataclass
class ModelCard:
    """What a reviewer, an auditor, or you-in-six-months needs to know."""
    name: str
    version: str
    created_utc: str
    git_sha: str
    python: str
    n_train_rows: int
    train_date_min: str
    train_date_max: str
    data_fingerprint: str
    features: list[str]
    class_names: list[str]
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def build_card(name: str, model, train: pd.DataFrame, metrics: dict | None = None,
               cfg: Config = CFG, notes: str = "") -> ModelCard:
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    features = list(getattr(model, "cols", None) or
                    getattr(getattr(model, "mx", None), "dummies", []) or [])
    return ModelCard(
        name=name, version=version,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(), python=platform.python_version(),
        n_train_rows=len(train),
        train_date_min=str(train["due_date"].min().date()),
        train_date_max=str(train["due_date"].max().date()),
        data_fingerprint=data_fingerprint(train[["policy_id", "due_date", "days_to_pay"]]),
        features=features, class_names=cfg.class_names,
        config={"grace_days": cfg.grace_days, "buckets": [list(b) for b in cfg.buckets],
                "seed": cfg.seed},
        metrics=metrics or {}, notes=notes)


def save(model, card: ModelCard, store: str | Path | None = None) -> Path:
    """Write <store>/<name>/<version>/{model.joblib,model_card.json} and move
    the `latest` pointer. Versioned directories mean a rollback is a one-line
    change, not a retrain."""
    root = Path(store or CFG.model_store) / card.name / card.version
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, root / "model.joblib")
    (root / "model_card.json").write_text(card.to_json())
    pointer = Path(store or CFG.model_store) / card.name / "LATEST"
    pointer.write_text(card.version)
    log.info("saved model %s v%s -> %s", card.name, card.version, root)
    return root


def load(name: str, version: str = "latest", store: str | Path | None = None):
    """Returns (model, card_dict)."""
    base = Path(store or CFG.model_store) / name
    if version == "latest":
        pointer = base / "LATEST"
        if not pointer.exists():
            raise FileNotFoundError(f"no saved versions of model {name!r} under {base}")
        version = pointer.read_text().strip()
    root = base / version
    model = joblib.load(root / "model.joblib")
    card = json.loads((root / "model_card.json").read_text())
    log.info("loaded model %s v%s", name, version)
    return model, card


def list_versions(name: str, store: str | Path | None = None) -> list[str]:
    base = Path(store or CFG.model_store) / name
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())

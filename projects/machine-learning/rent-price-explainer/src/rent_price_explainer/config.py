"""YAML-backed configuration with validation."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "conf" / "config.yaml"


@dataclass(frozen=True)
class Config:
    n_listings: int = 6000
    junk_features: int = 8
    raw_path: str = "data/listings.parquet"

    test_size: float = 0.25
    split_seed: int = 42

    vif_threshold: float = 5.0
    alpha: float = 0.05
    cooks_multiplier: float = 4.0

    gbm_log_target: bool = True
    shap_sample: int = 800

    artifacts: str = "artifacts"
    model_store: str = "models"

    seed: int = 42
    log_level: str = "INFO"

    def validate(self) -> "Config":
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be in (0, 1)")
        if self.vif_threshold <= 1:
            raise ValueError("a VIF threshold at or below 1 is meaningless")
        if not 0 < self.alpha < 0.5:
            raise ValueError("alpha must be a sensible significance level")
        if self.junk_features < 1:
            raise ValueError(
                "junk_features must be >= 1 -- the R-squared inflation "
                "demonstration is a core part of this project")
        return self


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else Path(os.environ.get("RENT_CONFIG", DEFAULT_CONFIG))
    if not p.exists():
        return Config().validate()
    raw = yaml.safe_load(p.read_text()) or {}
    d, s = raw.get("data", {}), raw.get("split", {})
    dg, m = raw.get("diagnostics", {}), raw.get("model", {})
    pa, r = raw.get("paths", {}), raw.get("run", {})
    return Config(
        n_listings=d.get("n_listings", 6000),
        junk_features=d.get("junk_features", 8),
        raw_path=d.get("raw_path", "data/listings.parquet"),
        test_size=s.get("test_size", 0.25),
        split_seed=s.get("seed", 42),
        vif_threshold=dg.get("vif_threshold", 5.0),
        alpha=dg.get("alpha", 0.05),
        cooks_multiplier=dg.get("cooks_multiplier", 4.0),
        gbm_log_target=m.get("gbm_log_target", True),
        shap_sample=m.get("shap_sample", 800),
        artifacts=pa.get("artifacts", "artifacts"),
        model_store=pa.get("model_store", "models"),
        seed=r.get("seed", 42),
        log_level=r.get("log_level", "INFO"),
    ).validate()


CFG = load_config()

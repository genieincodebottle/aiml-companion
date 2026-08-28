"""YAML-backed configuration with validation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "conf" / "config.yaml"


@dataclass(frozen=True)
class Config:
    n_tickets: int = 9000
    raw_path: str = "data/tickets.parquet"
    dependency_strength: float = 0.85
    boilerplate_rate: float = 0.55

    test_size: float = 0.25
    split_seed: int = 42

    alpha: float = 0.3
    calibration_method: str = "isotonic"
    calibration_cv: int = 3

    auto_route_threshold: float = 0.80
    human_capacity_share: float = 0.25

    n_calibration_bins: int = 12
    sweep_points: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)

    artifacts: str = "artifacts"
    model_store: str = "models"

    seed: int = 42
    log_level: str = "INFO"

    def validate(self) -> "Config":
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be in (0, 1)")
        if not 0.0 <= self.dependency_strength <= 1.0:
            raise ValueError("dependency_strength must be a probability")
        if self.alpha <= 0:
            raise ValueError(
                "alpha must be > 0: zero smoothing lets one unseen token drive "
                "the whole posterior to zero, which is the bug Laplace fixes")
        if not 0 < self.auto_route_threshold < 1:
            raise ValueError("auto_route_threshold must be in (0, 1)")
        if self.n_calibration_bins < 2:
            raise ValueError("need at least 2 bins to measure calibration")
        if not self.sweep_points:
            raise ValueError(
                "sweep_points drives the headline experiment and cannot be empty")
        return self


def load_config(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else Path(os.environ.get("TRIAGE_CONFIG", DEFAULT_CONFIG))
    if not p.exists():
        return Config().validate()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    d, s = raw.get("data", {}), raw.get("split", {})
    m, rt = raw.get("model", {}), raw.get("routing", {})
    e, pa = raw.get("evaluation", {}), raw.get("paths", {})
    r = raw.get("run", {})
    return Config(
        n_tickets=d.get("n_tickets", 9000),
        raw_path=d.get("raw_path", "data/tickets.parquet"),
        dependency_strength=d.get("dependency_strength", 0.85),
        boilerplate_rate=d.get("boilerplate_rate", 0.55),
        test_size=s.get("test_size", 0.25),
        split_seed=s.get("seed", 42),
        alpha=m.get("alpha", 0.3),
        calibration_method=m.get("calibration_method", "isotonic"),
        calibration_cv=m.get("calibration_cv", 3),
        auto_route_threshold=rt.get("auto_route_threshold", 0.80),
        human_capacity_share=rt.get("human_capacity_share", 0.25),
        n_calibration_bins=e.get("n_calibration_bins", 12),
        sweep_points=tuple(e.get("sweep_points", (0.0, 0.2, 0.4, 0.6, 0.8, 0.95))),
        artifacts=pa.get("artifacts", "artifacts"),
        model_store=pa.get("model_store", "models"),
        seed=r.get("seed", 42),
        log_level=r.get("log_level", "INFO"),
    ).validate()


CFG = load_config()

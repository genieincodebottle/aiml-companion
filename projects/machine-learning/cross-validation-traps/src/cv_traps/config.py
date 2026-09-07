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
    n_customers: int = 2600
    n_periods: int = 12
    n_noise: int = 150
    raw_path: str = "data/panel.parquet"

    # The three dials that switch the traps on and off. At group_effect=0 and
    # drift=0 the process is exchangeable and stationary, which is the one
    # world where naive KFold is actually correct.
    group_effect: float = 2.50
    drift: float = 0.90
    missing_rate: float = 0.08
    label_noise: float = 0.05

    n_folds: int = 5
    n_repeats: int = 20
    select_k: int = 20
    holdout_periods: int = 3
    holdout_customer_share: float = 0.30
    #: Row counts for the preprocessing trap's sample-size curve.
    preprocess_sizes: tuple[int, ...] = (200, 400, 800, 2000, 7000)
    n_candidates: int = 30
    candidate_size: int = 25

    seed: int = 42
    artifacts: str = "artifacts"
    log_level: str = "INFO"

    def validate(self) -> "Config":
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.n_periods - self.holdout_periods < 3:
            raise ValueError(
                "holdout_periods leaves fewer than 3 training periods, so "
                "forward-chaining validation has nothing to chain over")
        if not 0 < self.holdout_customer_share < 1:
            raise ValueError("holdout_customer_share must be in (0, 1)")
        if self.select_k >= self.n_noise:
            raise ValueError(
                "select_k must be smaller than n_noise, otherwise selection "
                "cannot discard anything and the preprocessing trap is inert")
        if not 0.0 <= self.label_noise < 0.5:
            raise ValueError(
                "label_noise must be in [0, 0.5); at 0.5 the label is a coin "
                "flip and nothing is learnable")
        if self.group_effect < 0 or self.drift < 0:
            raise ValueError("group_effect and drift are magnitudes, not signs")
        return self


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load settings, failing loudly when an explicit path is wrong.

    Only the built-in default may be absent. A path given through `--config` or
    `CVTRAPS_CONFIG` that does not exist used to fall back to the defaults in
    silence, so a typo ran the full project panel while the operator believed
    they were running their own small one. That is the same failure the data
    cache manifest exists to prevent, and it deserves the same treatment.
    """
    explicit = path if path is not None else os.environ.get("CVTRAPS_CONFIG")
    p = Path(explicit) if explicit else Path(DEFAULT_CONFIG)
    if not p.exists():
        if explicit and path is not None:
            raise FileNotFoundError(_missing(p))
        return Config().validate()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    d = raw.get("data", {})
    v = raw.get("validation", {})
    pa = raw.get("paths", {})
    r = raw.get("run", {})
    return Config(
        n_customers=d.get("n_customers", 2600),
        n_periods=d.get("n_periods", 12),
        n_noise=d.get("n_noise", 150),
        raw_path=d.get("raw_path", "data/panel.parquet"),
        group_effect=d.get("group_effect", 2.50),
        drift=d.get("drift", 0.90),
        missing_rate=d.get("missing_rate", 0.08),
        label_noise=d.get("label_noise", 0.05),
        n_folds=v.get("n_folds", 5),
        n_repeats=v.get("n_repeats", 20),
        select_k=v.get("select_k", 20),
        holdout_periods=v.get("holdout_periods", 3),
        holdout_customer_share=v.get("holdout_customer_share", 0.30),
        preprocess_sizes=tuple(v.get("preprocess_sizes",
                                     (200, 400, 800, 2000, 7000))),
        n_candidates=v.get("n_candidates", 30),
        candidate_size=v.get("candidate_size", 25),
        seed=r.get("seed", 42),
        artifacts=pa.get("artifacts", "artifacts"),
        log_level=r.get("log_level", "INFO"),
    ).validate()


def _missing(p) -> str:
    return (f"config file not found: {p}\n"
            "Nothing was loaded, so the run would have silently used the "
            "built-in defaults instead of your settings.")


def check_environment_config() -> None:
    """Validate CVTRAPS_CONFIG, if set, without doing it at import time.

    Import-time validation put a traceback on an `import` line, which is the
    least readable place an error can surface. The CLI calls this once, inside
    the handler that turns a bad setting into a one-line message.
    """
    env = os.environ.get("CVTRAPS_CONFIG")
    if env and not Path(env).exists():
        raise FileNotFoundError(_missing(Path(env)))


CFG = load_config()

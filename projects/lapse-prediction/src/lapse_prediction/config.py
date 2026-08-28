"""Configuration: YAML-backed, with the same object shape the code already uses.

`CFG` stays importable as a module-level singleton for convenience, but it is
loaded from conf/config.yaml rather than hardcoded, and `load_config(path)`
lets a pipeline run against an alternative config (a different grace period,
a finer hazard granularity) without touching source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "conf" / "config.yaml"


@dataclass(frozen=True)
class Config:
    grace_days: int = 45
    buckets: tuple = (("d0_7", 7), ("d8_15", 15), ("d16_30", 30), ("d31_grace", 45))
    lapse_label: str = "lapsed"

    hazard_horizon_days: int = 365
    hazard_period_days: int = 7

    valid_months: int = 3
    test_months: int = 3

    capacity_pct: float = 0.20
    contact_lead_days: int = 5
    assumed_save_rate: float = 0.25

    raw_data: str = "data/dues.parquet"
    modelling_table: str = "data/modelling_table.parquet"
    model_store: str = "models"
    artifacts: str = "artifacts"

    seed: int = 42
    log_level: str = "INFO"

    # ------------------------------------------------------------------ derived
    @property
    def n_classes(self) -> int:
        return len(self.buckets) + 1

    @property
    def class_names(self) -> list[str]:
        return [b[0] for b in self.buckets] + [self.lapse_label]

    @property
    def lapse_index(self) -> int:
        return len(self.buckets)

    @property
    def edges(self) -> list[float]:
        return [float(ub) for _, ub in self.buckets]

    def bucket_of(self, days_to_pay) -> int:
        """days_to_pay None/NaN or > grace -> lapse bucket."""
        if days_to_pay is None or days_to_pay != days_to_pay or days_to_pay > self.grace_days:
            return self.lapse_index
        for i, (_, ub) in enumerate(self.buckets):
            if days_to_pay <= ub:
                return i
        return self.lapse_index

    # ------------------------------------------------------------- validation
    def validate(self) -> "Config":
        ubs = [ub for _, ub in self.buckets]
        if ubs != sorted(ubs):
            raise ValueError(f"bucket upper bounds must be ascending, got {ubs}")
        if ubs[-1] != self.grace_days:
            raise ValueError(
                f"last bucket edge ({ubs[-1]}) must equal grace_days "
                f"({self.grace_days}) -- otherwise the lapse tail is mis-defined")
        if self.hazard_horizon_days < self.grace_days:
            raise ValueError("hazard horizon must cover at least the grace period")
        if not 0 < self.capacity_pct <= 1:
            raise ValueError("capacity_pct must be in (0, 1]")
        return self


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Read YAML into a validated Config. Falls back to dataclass defaults if
    the file is absent, so the package still imports in a bare environment."""
    p = Path(path) if path else Path(os.environ.get("LAPSE_CONFIG", DEFAULT_CONFIG))
    if not p.exists():
        return Config().validate()
    raw: dict[str, Any] = yaml.safe_load(p.read_text()) or {}
    b, s, sp = raw.get("business", {}), raw.get("survival", {}), raw.get("split", {})
    d, pa, r = raw.get("decisioning", {}), raw.get("paths", {}), raw.get("run", {})
    return Config(
        grace_days=b.get("grace_days", 45),
        buckets=tuple(tuple(x) for x in b.get("buckets", Config.buckets)),
        lapse_label=b.get("lapse_label", "lapsed"),
        hazard_horizon_days=s.get("horizon_days", 365),
        hazard_period_days=s.get("period_days", 7),
        valid_months=sp.get("valid_months", 3),
        test_months=sp.get("test_months", 3),
        capacity_pct=d.get("capacity_pct", 0.20),
        contact_lead_days=d.get("contact_lead_days", 5),
        assumed_save_rate=d.get("assumed_save_rate", 0.25),
        raw_data=pa.get("raw_data", "data/dues.parquet"),
        modelling_table=pa.get("modelling_table", "data/modelling_table.parquet"),
        model_store=pa.get("model_store", "models"),
        artifacts=pa.get("artifacts", "artifacts"),
        seed=r.get("seed", 42),
        log_level=r.get("log_level", "INFO"),
    ).validate()


CFG = load_config()

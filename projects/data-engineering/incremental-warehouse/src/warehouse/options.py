"""The switches that separate the correct pipeline from each naive variant.

Every naive variant in `naive.py` is the correct pipeline with exactly one
switch flipped, so a measured difference has a single cause.
"""
from dataclasses import dataclass, replace

_CHOICES = {
    "load_mode": ("overwrite", "append"),
    "scd_type": (1, 2),
    "clock": ("run_date", "wall"),
    "checks_mode": ("gate", "report"),
}


@dataclass(frozen=True)
class PipelineOptions:
    lookback_minutes: int = 180
    detect_deletes: bool = True
    load_mode: str = "overwrite"
    scd_type: int = 2
    clock: str = "run_date"
    checks_mode: str = "gate"
    retries: int = 2
    retry_delay_seconds: float = 0.05
    retry_backoff: float = 2.0

    def __post_init__(self):
        for name, allowed in _CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")

    @classmethod
    def from_cfg(cls, cfg, **overrides):
        return replace(cls(**cfg["pipeline"]), **overrides)

    def but(self, **overrides):
        return replace(self, **overrides)

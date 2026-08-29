"""Data access. The one place that knows where bytes come from."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cv_traps.config import CFG, Config
from cv_traps.data import schema
from cv_traps.utils.logging import get_logger

log = get_logger(__name__)


def save(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except (ImportError, ValueError):
            log.warning("pyarrow unavailable, falling back to csv")
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def read(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" \
        else pd.read_csv(path)


def load_panel(*, refresh: bool = False, cfg: Config = CFG,
               validate: bool = True) -> pd.DataFrame:
    """Load the panel, generating and caching it on first use.

    The cache carries a manifest of the settings that produced it. Without one,
    `run.py data --group-effect 0.0` leaves a control panel on disk that every
    later command reads in silence, reporting no group leakage at all while the
    config still says 2.50. That is not a smaller version of the experiment, it
    is the opposite of it, and nothing on screen would say so.
    """
    path = Path(cfg.raw_path)
    if path.exists() and not refresh:
        try:
            df = read(path)
        except Exception as e:                      # noqa: BLE001
            # Deliberately broad. Every reader failure has the same remedy,
            # and the list of ways a cache file can be unreadable (truncated,
            # wrong format, permission, unsupported engine) is longer than any
            # tuple worth maintaining here.
            log.warning("could not read the cached panel at %s (%s: %s) -- "
                        "regenerating it", path, type(e).__name__, e)
            df = None
        stale = "unreadable cache" if df is None else _staleness(df, cfg, path)
        if stale:
            log.warning("cached panel does not match the config (%s) -- "
                        "regenerating. Delete %s to start clean.", stale, path)
        else:
            log.info("loaded %d rows from %s", len(df), path)
            if validate:
                schema.validate(df)
            return df

    from cv_traps.data.generate import generate
    log.info("generating panel (%d customers, %d periods, group_effect=%.2f, "
             "drift=%.2f)", cfg.n_customers, cfg.n_periods, cfg.group_effect,
             cfg.drift)
    df = generate(n_customers=cfg.n_customers, n_periods=cfg.n_periods,
                  n_noise=cfg.n_noise, seed=cfg.seed,
                  group_effect=cfg.group_effect, drift=cfg.drift,
                  missing_rate=cfg.missing_rate, label_noise=cfg.label_noise)
    save(df, path)
    _write_manifest(path, cfg)
    log.info("cached panel to %s", path)
    if validate:
        schema.validate(df)
    log.info("panel: %s", schema.summarise(df))
    return df


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _manifest(cfg: Config) -> dict:
    """The settings that change what the data IS, not merely where it lives."""
    return {"n_customers": cfg.n_customers, "n_periods": cfg.n_periods,
            "n_noise": cfg.n_noise, "seed": cfg.seed,
            "group_effect": cfg.group_effect, "drift": cfg.drift,
            "missing_rate": cfg.missing_rate, "label_noise": cfg.label_noise}


def _write_manifest(path: Path, cfg: Config) -> None:
    _manifest_path(path).write_text(json.dumps(_manifest(cfg), indent=2),
                                    encoding="utf-8")


def _staleness(df: pd.DataFrame, cfg: Config, path: Path | None = None) -> str:
    """Why the cache cannot be trusted for this config, or an empty string."""
    if path is not None:
        mpath = _manifest_path(path)
        if not mpath.exists():
            return ("no manifest beside the cache, so the settings that built "
                    "it are unknown")
        try:
            recorded = json.loads(mpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return "the cache manifest is unreadable"
        differing = [f"{k}: cached {recorded.get(k)!r}, config wants {v!r}"
                     for k, v in _manifest(cfg).items() if recorded.get(k) != v]
        if differing:
            return "; ".join(differing)

    if df["customer_id"].nunique() != cfg.n_customers:
        return (f"{df['customer_id'].nunique()} customers cached, "
                f"{cfg.n_customers} requested")
    n_noise = len([c for c in df.columns if c.startswith("noise_")])
    if n_noise != cfg.n_noise:
        return f"{n_noise} noise columns cached, {cfg.n_noise} requested"
    return ""

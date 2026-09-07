"""Data access -- the single seam between this package and a real data source.

Swap `generate()` for a warehouse query or a listings CSV and nothing
downstream changes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rent_price_explainer.config import CFG, Config
from rent_price_explainer.data import schema
from rent_price_explainer.utils.logging import get_logger

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
    """Read a cached listings file, parquet or csv."""
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet"         else pd.read_csv(path)


def _staleness(df: pd.DataFrame, cfg: Config) -> list[str]:
    """Reasons the cached file does not answer the current config."""
    reasons = []
    if len(df) != cfg.n_listings:
        reasons.append(f"{len(df)} rows cached, {cfg.n_listings} requested")
    n_junk = len([c for c in df.columns if c.startswith("junk_")])
    if n_junk != cfg.junk_features:
        reasons.append(f"{n_junk} junk columns cached, {cfg.junk_features} requested")
    return reasons


def load_listings(path: str | Path | None = None, *, refresh: bool = False,
                  validate: bool = True, cfg: Config = CFG) -> pd.DataFrame:
    path = Path(path or cfg.raw_path)

    if path.exists() and not refresh:
        try:
            cached = read(path)
        except Exception as e:
            # A half-written or corrupted cache (an interrupted first run is the
            # usual cause) should not surface as a pyarrow traceback.
            log.warning("could not read %s (%s: %s) -- regenerating it.",
                        path, type(e).__name__, str(e)[:80])
            cached, stale = None, ["unreadable cache"]
        else:
            stale = _staleness(cached, cfg)
        if stale:
            # Handing back data that does not match the requested config is a
            # silent-wrong-answer bug: nothing errors, the numbers are just not
            # the ones asked for. Regenerate instead, and say so loudly.
            if cached is not None:
                log.warning("cached listings do not match the config (%s) -- "
                            "regenerating. Delete %s to start clean.",
                            "; ".join(stale), path)
            refresh = True
        else:
            log.info("loaded listings from %s", path)
            df = cached

    if refresh or not path.exists():
        from rent_price_explainer.data.generate import generate
        log.info("generating synthetic market (%d listings)", cfg.n_listings)
        df = generate(n=cfg.n_listings, seed=cfg.seed,
                      junk_features=cfg.junk_features)
        save(df, path)
        log.info("cached listings to %s", path)

    if validate:
        schema.validate(df)
    log.info("listings: %s", schema.summarise(df))
    return df

"""Data access. The ONE place that knows where bytes come from.

`load_ledger()` is the seam between this package and your warehouse: point it
at a parquet/csv extract, or replace the `source="synthetic"` branch with a
SQL client. Everything downstream sees a validated DataFrame and nothing else.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lapse_prediction.config import CFG, Config
from lapse_prediction.data import schema
from lapse_prediction.utils.logging import get_logger

log = get_logger(__name__)


def save(df: pd.DataFrame, path: str | Path) -> Path:
    """Write parquet, falling back to CSV when pyarrow is unavailable."""
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
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(
        path, parse_dates=["due_date"])
    return df


def load_ledger(path: str | Path | None = None, *, refresh: bool = False,
                source: str = "synthetic", validate: bool = True,
                cfg: Config = CFG, **gen_kwargs) -> pd.DataFrame:
    """Load the renewal ledger, generating and caching a synthetic one if absent.

    Replace the `source` branch with your warehouse client to go to production;
    the validation and logging around it should stay exactly as they are.
    """
    path = Path(path or cfg.raw_data)
    if path.exists() and not refresh:
        df = read(path)
        log.info("loaded ledger from %s", path)
    elif source == "synthetic":
        from lapse_prediction.data.generate import generate
        log.info("no cached ledger at %s -- generating synthetic book", path)
        df = generate(seed=cfg.seed, **gen_kwargs)
        save(df, path)
        log.info("cached ledger to %s", path)
    else:
        raise FileNotFoundError(
            f"no ledger at {path} and source={source!r} has no loader wired up")

    df["due_date"] = pd.to_datetime(df["due_date"])
    if validate:
        schema.validate(df)
    log.info("ledger: %s", schema.summarise(df))
    return df

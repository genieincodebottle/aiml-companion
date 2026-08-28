"""Data access. The one place that knows where bytes come from."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from support_ticket_triage.config import CFG, Config
from support_ticket_triage.data import schema
from support_ticket_triage.utils.logging import get_logger

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
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def load_tickets(*, refresh: bool = False, cfg: Config = CFG,
                 validate: bool = True) -> pd.DataFrame:
    """Load the inbox, generating and caching it on first use.

    The cache carries a manifest of the settings that produced it. Without one,
    `run.py data --dependency 0.0` leaves a control dataset on disk that every
    later command reads in silence, reporting a lift of 1.00 and a blind-survey
    recall of 0% while the config still says 0.85. That is not a smaller
    version of the experiment, it is the opposite of it, and nothing on screen
    would say so.
    """
    path = Path(cfg.raw_path)
    if path.exists() and not refresh:
        df = read(path)
        stale = _staleness(df, cfg, path)
        if stale:
            log.warning("cached tickets do not match the config (%s) -- "
                        "regenerating. Delete %s to start clean.", stale, path)
        else:
            log.info("loaded %d tickets from %s", len(df), path)
            if validate:
                schema.validate(df)
            return df

    from support_ticket_triage.data.generate import generate
    log.info("generating inbox (%d tickets, dependency_strength=%.2f)",
             cfg.n_tickets, cfg.dependency_strength)
    df = generate(n_tickets=cfg.n_tickets, seed=cfg.seed,
                  dependency_strength=cfg.dependency_strength,
                  boilerplate_rate=cfg.boilerplate_rate)
    save(df, path)
    _write_manifest(path, cfg)
    log.info("cached tickets to %s", path)
    if validate:
        schema.validate(df)
    log.info("inbox: %s", schema.summarise(df))
    return df


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _manifest(cfg: Config) -> dict:
    """The settings that change what the data IS, not merely where it lives."""
    return {"n_tickets": cfg.n_tickets, "seed": cfg.seed,
            "dependency_strength": cfg.dependency_strength,
            "boilerplate_rate": cfg.boilerplate_rate}


def _write_manifest(path: Path, cfg: Config) -> None:
    _manifest_path(path).write_text(json.dumps(_manifest(cfg), indent=2),
                                    encoding="utf-8")


def _staleness(df: pd.DataFrame, cfg: Config, path: Path | None = None) -> str:
    """Why the cache cannot be trusted for this config, or an empty string.

    Row count is the obvious check. The vocabulary width is the one that
    actually bites: edit the generator to add tokens and every cached file
    silently has the wrong number of feature columns. Catching it here turns a
    confusing schema error into an automatic regeneration.
    """
    from support_ticket_triage.data.generate import VOCAB

    # The settings check comes first, because it is the one that silently
    # inverts the experiment rather than merely resizing it.
    if path is not None:
        mpath = _manifest_path(path)
        if not mpath.exists():
            return ("no manifest beside the cache, so the settings that built "
                    "it are unknown")
        try:
            recorded = json.loads(mpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return "the cache manifest is unreadable"
        wanted = _manifest(cfg)
        differing = [f"{k}: cached {recorded.get(k)!r}, config wants {v!r}"
                     for k, v in wanted.items() if recorded.get(k) != v]
        if differing:
            return "; ".join(differing)

    if len(df) != cfg.n_tickets:
        return f"{len(df)} rows cached, {cfg.n_tickets} requested"
    missing = [t for t in VOCAB if t not in df.columns]
    if missing:
        return (f"vocabulary changed, {len(missing)} tokens absent from the "
                f"cache, starting with {missing[:3]}")
    extra = [c for c in df.columns
             if c not in set(VOCAB) and c not in {"ticket_id", "category",
                                                  "text", "n_tokens"}]
    if extra:
        return f"cache has {len(extra)} tokens the vocabulary no longer contains"
    return ""

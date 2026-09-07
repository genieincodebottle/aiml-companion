"""Inference. Loads a versioned model and scores open dues at login time.

Two entry points:
  score_batch()  -- nightly job over every open due, writes the retention queue
  score_one()    -- single policy, for a synchronous call from the servicing UI
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lapse_prediction.config import CFG, Config
from lapse_prediction.data.io import load_ledger, save
from lapse_prediction.evaluation import metrics
from lapse_prediction.features.build import build
from lapse_prediction.features.labels import add_labels
from lapse_prediction.models import registry
from lapse_prediction.serving import decide
from lapse_prediction.utils.logging import get_logger

log = get_logger(__name__)


class Scorer:
    """Holds a loaded model + its card. Construct once, reuse per request."""

    def __init__(self, name: str = "ordinal_chain", version: str = "latest",
                 cfg: Config = CFG):
        self.model, self.card = registry.load(name, version, cfg.model_store)
        self.cfg = cfg
        self.name, self.version = name, self.card["version"]

    # -------------------------------------------------------------- internals
    def _features(self, ledger: pd.DataFrame) -> pd.DataFrame:
        """History features need each policy's PRIOR dues, so the caller must
        pass the policy's full ledger, not just the open row."""
        return add_labels(build(ledger, self.cfg), self.cfg)

    def _check_features(self, df: pd.DataFrame) -> None:
        expected = self.card.get("features") or []
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(
                f"model {self.name} v{self.version} expects features absent at "
                f"score time: {missing[:10]}")

    # ---------------------------------------------------------------- public
    def score_frame(self, feats: pd.DataFrame) -> pd.DataFrame:
        self._check_features(feats)
        proba = self.model.predict_proba(feats)
        out = decide.score(feats, proba, self.cfg,
                           capacity_pct=self.cfg.capacity_pct,
                           contact_lead_days=self.cfg.contact_lead_days)
        out.insert(1, "model_version", self.version)
        return out

    def score_batch(self, ledger: pd.DataFrame | None = None,
                    as_of: pd.Timestamp | None = None,
                    write: bool = True) -> pd.DataFrame:
        """Score the dues that are still open as of `as_of` -- i.e. the ones a
        retention team can still act on. Settled dues are excluded because
        scoring them tells nobody anything."""
        ledger = load_ledger(cfg=self.cfg) if ledger is None else ledger
        feats = self._features(ledger)
        as_of = pd.Timestamp(as_of) if as_of is not None else feats["due_date"].max()

        open_mask = ((feats["due_date"] <= as_of) &
                     (feats["due_date"] + pd.Timedelta(days=self.cfg.grace_days) > as_of))
        target = feats[open_mask]
        if target.empty:
            log.warning("no open dues as of %s -- nothing to score", as_of.date())
            return pd.DataFrame()

        log.info("scoring %d open dues as of %s", len(target), as_of.date())
        out = self.score_frame(target)
        if write:
            path = Path(self.cfg.artifacts) / "retention_queue.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
            log.info("retention queue -> %s (%d rows)", path, len(out))
            log.info("queue economics: %s",
                     decide.value_of_queue(out, self.cfg.assumed_save_rate))
        return out

    def score_one(self, policy_ledger: pd.DataFrame) -> dict:
        """Score the most recent due for one policy. `policy_ledger` must
        contain that policy's history so the lag features can be built."""
        feats = self._features(policy_ledger)
        row = feats.sort_values("due_date").tail(1)
        proba = self.model.predict_proba(row)[0]
        li = self.cfg.lapse_index
        return {
            "policy_id": str(row["policy_id"].iloc[0]),
            "due_date": str(row["due_date"].iloc[0].date()),
            "model_version": self.version,
            "p_lapse": float(proba[li]),
            "expected_days_if_paid": float(
                metrics.expected_days(proba[None, :], self.cfg)[0]),
            "distribution": {n: float(p) for n, p in
                             zip(self.cfg.class_names, proba)},
        }


def run(name: str = "ordinal_chain", version: str = "latest",
        cfg: Config = CFG) -> pd.DataFrame:
    return Scorer(name, version, cfg).score_batch()


if __name__ == "__main__":
    run()

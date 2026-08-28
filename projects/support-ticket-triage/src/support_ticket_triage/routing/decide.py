"""Turning probabilities into what actually happens to the ticket.

The business rule is one line: auto-route when the model is confident enough,
otherwise put it in the human queue. That rule is where the calibration
argument stops being academic. "Confident enough" is a number, and on
uncalibrated Naive Bayes scores the number does not mean what it says, so the
rule silently auto-routes tickets it should have escalated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from support_ticket_triage.config import CFG, Config


def route(proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray | None = None,
          cfg: Config = CFG) -> pd.DataFrame:
    """Apply the threshold and record what each ticket's fate was."""
    conf = proba.max(axis=1)
    pred = classes[np.argmax(proba, axis=1)]
    auto = conf >= cfg.auto_route_threshold

    out = pd.DataFrame({
        "predicted": pred,
        "confidence": conf.round(4),
        "action": np.where(auto, "auto_route", "human_queue"),
    })
    if y_true is not None:
        out["actual"] = y_true
        out["correct"] = pred == y_true
    return out


def routing_summary(routed: pd.DataFrame, cfg: Config = CFG) -> dict:
    """What the rule cost and what it bought.

    `auto_routed_error_rate` is the number that matters: those are the tickets
    that reach a customer with no human check. A model whose confidence is
    inflated puts far more traffic into that bucket than it has earned.
    """
    auto = routed[routed["action"] == "auto_route"]
    human = routed[routed["action"] == "human_queue"]
    summary = {
        "threshold": cfg.auto_route_threshold,
        "auto_routed_share": round(len(auto) / len(routed), 4),
        "human_queue_share": round(len(human) / len(routed), 4),
    }
    if "correct" in routed.columns:
        summary["auto_routed_error_rate"] = (
            round(float(1 - auto["correct"].mean()), 4) if len(auto) else 0.0)
        summary["human_queue_error_rate"] = (
            round(float(1 - human["correct"].mean()), 4) if len(human) else 0.0)
        summary["errors_reaching_customer"] = int((~auto["correct"]).sum())
        # Did the queue actually earn its cost? If the model is well ordered,
        # the tickets it held back should be much harder than the ones it sent.
        summary["queue_is_harder_than_auto"] = bool(
            len(auto) and len(human)
            and (1 - human["correct"].mean()) > (1 - auto["correct"].mean()))
    over = float((routed["confidence"] >= cfg.auto_route_threshold).mean())
    if over > 1 - cfg.human_capacity_share / 4 and len(routed) > 100:
        summary["capacity_warning"] = (
            f"{over:.1%} of tickets clear the threshold, so the human queue is "
            "nearly empty. On an overconfident model that is not a triumph, it "
            "is the threshold failing to bind.")
    return summary


def threshold_sweep(proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray,
                    points: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
                    cfg: Config = CFG) -> pd.DataFrame:
    """The operating curve: what each threshold does to volume and to errors.

    This is the table an operations lead actually needs, and it only makes sense
    when the probabilities are calibrated. Read it against the uncalibrated
    model and every column is a fiction.
    """
    conf = proba.max(axis=1)
    pred = classes[np.argmax(proba, axis=1)]
    correct = pred == y_true

    rows = []
    for t in points:
        auto = conf >= t
        rows.append({
            "threshold": t,
            "auto_routed_share": round(float(auto.mean()), 4),
            "auto_routed_error_rate": round(
                float(1 - correct[auto].mean()), 4) if auto.any() else 0.0,
            "errors_reaching_customer": int((~correct[auto]).sum()),
            "human_reviews_needed": int((~auto).sum()),
        })
    return pd.DataFrame(rows)

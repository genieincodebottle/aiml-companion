"""Score a sink against the answer key. The only module that reads the key.

Money is integer cents everywhere, so drift is exact and never a float artefact.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import ANSWER_KEY_PATH


def load_key(path: Path = ANSWER_KEY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stream_facts(key: dict) -> dict:
    """The planted-fault counts, without the long id list."""
    return {k: v for k, v in key["planted"].items() if k != "retried_event_ids"}


def account_truth(key: dict, account_id: int) -> dict:
    return {"balance_cents": key["balances"][str(account_id)],
            "profile": key["profiles"].get(str(account_id))}


def score_sink(con, key: dict) -> dict:
    """Compare balances, profiles and per-event effects with the truth."""
    truth_bal = {int(a): v for a, v in key["balances"].items()}
    sink_bal = dict(con.execute("SELECT account_id, balance_cents FROM balances").fetchall())
    diffs = {a: sink_bal.get(a, 0) - t for a, t in truth_bal.items()}
    wrong = sum(1 for d in diffs.values() if d != 0)

    applied = dict(con.execute(
        "SELECT event_id, COUNT(*) FROM effect_audit GROUP BY event_id").fetchall())
    wallet_ids = key["wallet_event_ids"]
    lost = [e for e in wallet_ids if applied.get(e, 0) == 0]
    twice = {e: n for e, n in applied.items() if n > 1}

    sink_prof = {a: (t, lim) for a, t, lim in con.execute(
        "SELECT account_id, tier, daily_limit_cents FROM profiles").fetchall()}
    wrong_prof = sum(
        1 for a, p in key["profiles"].items()
        if sink_prof.get(int(a)) != (p["tier"], p["daily_limit_cents"]))

    anomalies = sorted([(e, 0) for e in lost] + list(twice.items()))
    return {
        "accounts_wrong": wrong,
        "net_drift_cents": sum(diffs.values()),
        "abs_drift_cents": sum(abs(d) for d in diffs.values()),
        "events_lost": len(lost),
        "applied_twice": sum(n - 1 for n in twice.values()),
        "profiles_wrong": wrong_prof,
        "anomalies": anomalies,
    }

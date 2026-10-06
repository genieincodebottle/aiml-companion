"""Generate the wallet event stream, plant the faults, and write the answer key.

The stream is built in event-time order first, and only then are the delivery
faults applied. That order matters. Every fault is a statement about ARRIVAL
("this event reached the log later than it should have"), while the event
itself, and so the truth, never changes.

Planted faults, each at a configured rate with an exact count
  retries       an event is written twice with the same event_id (the producer
                did not see the ack and resent)
  late updates  a profile update arrives after a newer update for the same account
  ties          the last two updates of an account share one event_time, so
                event_time alone cannot say which is newer (version breaks it)

`answer_key` is the one name in the package that stands for the truth. It is
written to data/answer_key.json and read by evaluation/score.py and nothing
else (tests/test_generator.py scans for it).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ANSWER_KEY_PATH, MANIFEST_PATH
from ..log.base import partition_for

TIERS = ["free", "plus", "pro", "business"]


@dataclass
class Stream:
    """What the producer appends, in arrival order, plus the truth about it."""
    records: list[dict] = field(default_factory=list)
    answer_key: dict = field(default_factory=dict)
    featured: dict = field(default_factory=dict)


def _event_id(kind: str, n: int) -> str:
    return f"{kind}-{n:06d}"


def _build_events(cfg: dict, rng: random.Random) -> tuple[list[dict], dict[int, list[dict]]]:
    s = cfg["stream"]
    slots = ["w"] * s["n_wallet_events"] + ["p"] * s["n_profile_events"]
    rng.shuffle(slots)

    events: list[dict] = []
    updates: dict[int, list[dict]] = {}
    version: dict[int, int] = {}
    t = s["first_event_time"]
    for pos, slot in enumerate(slots):
        t += rng.randint(1, 3)
        account = rng.randrange(s["n_accounts"])
        if slot == "w":
            deposit = rng.random() < 0.6
            ev = {
                "type": "wallet_txn", "event_id": _event_id("w", pos), "account_id": account,
                "kind": "deposit" if deposit else "withdrawal",
                "amount_cents": rng.randint(500, 50_000) if deposit else rng.randint(100, 30_000),
                "event_time": t, "producer_attempt": 0,
            }
        else:
            version[account] = version.get(account, 0) + 1
            ev = {
                "type": "profile_updated", "event_id": _event_id("p", pos), "account_id": account,
                "tier": rng.choice(TIERS),
                "daily_limit_cents": rng.randint(1, 1_000_000) * 100,
                "event_time": t, "version": version[account], "producer_attempt": 0,
            }
            updates.setdefault(account, []).append(ev)
        ev["_pos"] = pos
        events.append(ev)
    # A stale update must never equal the truth by accident, or "wrong" would be
    # unmeasurable. Limits are drawn from 1M values, so collisions are checked, not assumed away.
    for evs in updates.values():
        assert len({e["daily_limit_cents"] for e in evs}) == len(evs), "limit collision"
    return events, updates


def _plant_ties(cfg: dict, rng: random.Random, updates: dict[int, list[dict]]) -> list[int]:
    eligible = sorted(a for a, u in updates.items() if len(u) >= 2)
    chosen = sorted(rng.sample(eligible, cfg["stream"]["tie_accounts"]))
    for a in chosen:
        updates[a][-1]["event_time"] = updates[a][-2]["event_time"]
    return chosen


def _plant_late(cfg: dict, rng: random.Random, updates: dict[int, list[dict]],
                key: dict[str, float]) -> list[str]:
    """Move non-final updates so they arrive after the account's newest update."""
    s = cfg["stream"]
    candidates = sorted((e["event_id"], a) for a, u in updates.items() for e in u[:-1])
    n_late = round(s["late_profile_rate"] * s["n_profile_events"])
    picked = sorted(rng.sample(candidates, n_late))
    for event_id, account in picked:
        newest = updates[account][-1]
        key[event_id] = (key[newest["event_id"]] + rng.randint(1, s["late_max_gap"])
                         + rng.random() * 0.5)
    return [event_id for event_id, _ in picked]


def _plant_retries(cfg: dict, rng: random.Random, events: list[dict],
                   key: dict[str, float]) -> tuple[list[dict], dict[int, float], list[str]]:
    """Each retried event is appended again a few records later in the same partition."""
    s = cfg["stream"]
    n_retry = round(s["retry_rate"] * len(events))
    picked = sorted(rng.sample([e["event_id"] for e in events], n_retry))
    by_id = {e["event_id"]: e for e in events}
    dupes, dupe_key = [], {}
    for event_id in picked:
        copy = dict(by_id[event_id])
        copy["producer_attempt"] = 1
        dupes.append(copy)
        dupe_key[id(copy)] = (key[event_id] + rng.randint(1, s["retry_max_gap"])
                              + rng.random() * 0.5)
    return dupes, dupe_key, picked


def _truth(events: list[dict], updates: dict[int, list[dict]], n_accounts: int) -> dict:
    """Each event_id applied once; the profile is the newest by (event_time, version)."""
    balances = {a: 0 for a in range(n_accounts)}
    for e in events:
        if e["type"] == "wallet_txn":
            sign = 1 if e["kind"] == "deposit" else -1
            balances[e["account_id"]] += sign * e["amount_cents"]
    profiles = {}
    for a, evs in updates.items():
        best = max(evs, key=lambda e: (e["event_time"], e["version"]))
        profiles[str(a)] = {"tier": best["tier"], "daily_limit_cents": best["daily_limit_cents"]}
    return {"balances": {str(a): v for a, v in sorted(balances.items())}, "profiles": profiles}


def _out_of_order_count(arrival: list[dict]) -> int:
    """Profile updates (first copy only) that reached the log after a higher version."""
    newest: dict[int, int] = {}
    seen: set[str] = set()
    late = 0
    for e in arrival:
        if e["type"] != "profile_updated" or e["event_id"] in seen:
            continue
        seen.add(e["event_id"])
        a = e["account_id"]
        if e["version"] < newest.get(a, 0):
            late += 1
        newest[a] = max(newest.get(a, 0), e["version"])
    return late


def _pick_featured(arrival: list[dict], n_partitions: int, retried: set[str], late: set[str]) -> dict:
    """The smallest account that shows all three faults, so its trace fits on a screen.

    It must have a retried wallet event (a duplicate), a late profile update
    (out of order), and one un-retried wallet event to pin a consumer crash on.
    """
    per_acct: dict[int, list[dict]] = {}
    for e in arrival:
        per_acct.setdefault(e["account_id"], []).append(e)
    best = None
    for a in sorted(per_acct):
        evs = per_acct[a]
        has_retry = any(e["type"] == "wallet_txn" and e["event_id"] in retried for e in evs)
        has_late = any(e["event_id"] in late for e in evs)
        targets = [e for e in evs if e["type"] == "wallet_txn" and e["event_id"] not in retried]
        if has_retry and has_late and targets:
            if best is None or len(evs) < best[0]:
                best = (len(evs), a, targets[0]["event_id"])
    if best is None:
        raise RuntimeError("no account shows all three faults; raise the fault rates")
    _, account, crash_event = best
    part = partition_for(str(account), n_partitions)
    offset = -1
    for e in arrival:
        if partition_for(str(e["account_id"]), n_partitions) == part:
            offset += 1
            if e["event_id"] == crash_event and e["producer_attempt"] == 0:
                break
    return {"account_id": account, "partition": part, "crash_offset": offset,
            "crash_event_id": crash_event}


def generate(cfg: dict) -> Stream:
    rng = random.Random(cfg["seed"])
    s = cfg["stream"]
    events, updates = _build_events(cfg, rng)
    tie_accounts = _plant_ties(cfg, rng, updates)
    key = {e["event_id"]: float(e["_pos"]) for e in events}
    late_ids = _plant_late(cfg, rng, updates, key)
    dupes, dupe_key, retried_ids = _plant_retries(cfg, rng, events, key)

    placed = [(key[e["event_id"]], i, e) for i, e in enumerate(events)]
    placed += [(dupe_key[id(d)], len(events) + j, d) for j, d in enumerate(dupes)]
    placed.sort(key=lambda x: (x[0], x[1]))
    ordered = [{k: v for k, v in e.items() if k != "_pos"} for _, _, e in placed]

    truth = _truth(events, updates, s["n_accounts"])
    featured = _pick_featured(ordered, s["n_partitions"], set(retried_ids), set(late_ids))
    truth["wallet_event_ids"] = sorted(e["event_id"] for e in events if e["type"] == "wallet_txn")
    truth["planted"] = {
        "distinct_events": len(events),
        "wallet_events": s["n_wallet_events"],
        "profile_events": s["n_profile_events"],
        "log_records": len(ordered),
        "retried_events": len(retried_ids),
        "retried_wallet_events": sum(1 for i in retried_ids if i.startswith("w-")),
        "retried_profile_events": sum(1 for i in retried_ids if i.startswith("p-")),
        "retried_event_ids": retried_ids,
        "late_profile_updates": len(late_ids),
        "out_of_order_arrivals": _out_of_order_count(ordered),
        "tie_accounts": len(tie_accounts),
        "accounts_with_profile": len(updates),
    }
    return Stream(records=ordered, answer_key=truth, featured=featured)


def write_stream(stream: Stream, cfg: dict, log, answer_key_path: Path = ANSWER_KEY_PATH,
                 manifest_path: Path = MANIFEST_PATH) -> None:
    """Append every record to the log (partitioned by account) and write the key."""
    s = cfg["stream"]
    for rec in stream.records:
        key = str(rec["account_id"])
        log.append(partition_for(key, s["n_partitions"]), key, rec)
    if hasattr(log, "flush"):
        log.flush()
    answer_key_path.parent.mkdir(parents=True, exist_ok=True)
    answer_key_path.write_text(
        json.dumps(stream.answer_key, sort_keys=True, separators=(",", ":")),
        encoding="utf-8", newline="\n")
    manifest = {
        "topic": s["topic"], "n_partitions": s["n_partitions"], "seed": cfg["seed"],
        "partition_lengths": [log.end_offset(p) for p in range(s["n_partitions"])],
        "featured": stream.featured,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                             encoding="utf-8", newline="\n")

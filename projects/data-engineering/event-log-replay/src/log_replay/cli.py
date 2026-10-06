"""Command line entry point. `python run.py --help` lists the subcommands."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from .config import ARTIFACTS_DIR, LOG_DIR, MANIFEST_PATH, load_config
from .data.generate import generate, write_stream
from .evaluation.score import account_truth, load_key, stream_facts
from .experiments import STRATEGIES, crash_schedule, load_manifest, make_job, open_log, run_jobs
from .log.base import partition_for
from .report import money, table

DEFAULT_JOBS = min(6, os.cpu_count() or 1)
MONEY_STRATEGIES = ["naive", "commit_first", "dedup_separate_txn", "atomic_offset_only", "atomic"]
REPLAY_STRATEGIES = ["naive", "commit_first", "dedup_separate_txn", "atomic"]
PROFILE_STRATEGIES = ["atomic_last_arrived_wins", "atomic_time_only_guard", "atomic"]


def _need_data() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit("No log yet. Run `python run.py produce` first.")


def _save(name: str, payload: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / name).write_text(json.dumps(payload, indent=2, sort_keys=True),
                                      encoding="utf-8", newline="\n")


def _public(score: dict) -> dict:
    return {k: v for k, v in score.items() if k != "anomalies"}


# ------------------------------------------------------------------ commands

def cmd_produce(args, cfg) -> int:
    from .log.filelog import FileLog
    s = cfg["stream"]
    stream = generate(cfg)
    file_log = FileLog(LOG_DIR, s["topic"], s["n_partitions"], create=True)
    write_stream(stream, cfg, file_log)
    manifest = load_manifest()
    facts = stream_facts(load_key())
    print(f"[OK] wrote {LOG_DIR} and the answer key")
    print(table(["fact", "count"], [[k, f"{v:,}"] for k, v in facts.items()], ["l", "r"]))
    print()
    print(table(["partition", "records"],
                [[p, f"{n:,}"] for p, n in enumerate(manifest["partition_lengths"])]))
    f = manifest["featured"]
    print(f"\nfeatured account {f['account_id']} (partition {f['partition']}); "
          f"the consumer crash schedule pins a crash on {f['crash_event_id']} "
          f"at offset {f['crash_offset']}")
    n = len(crash_schedule(cfg, manifest))
    print(f"crash schedule: {n} crashes at about one per {cfg['crashes']['every']} records per partition")
    _save("stream_facts.json", facts)
    if args.backend == "kafka":
        kafka = open_log(cfg, "kafka", args.bootstrap)
        kafka.recreate_topic()
        src = open_log(cfg, "file")
        for p in range(src.n_partitions):
            for rec in src.read(p, 0, src.end_offset(p)):
                kafka.append(p, rec.key, rec.value)
        kafka.flush()
        print(f"[OK] published the same {sum(manifest['partition_lengths']):,} records to Kafka")
    return 0


def _row(label: str, r: dict) -> list:
    s, st = r["score"], r["stats"]
    return [label, r["crashes"], f"{st['redelivered']:,}", s["accounts_wrong"],
            money(s["net_drift_cents"]), money(s["abs_drift_cents"]),
            s["events_lost"], s["applied_twice"], s["profiles_wrong"]]


SCORE_HEADERS = ["strategy", "crashes", "redelivered", "accounts wrong", "net drift",
                 "abs drift", "events lost", "applied twice", "profiles wrong"]


def _job(cfg, args, strategy, crashes, **kw) -> dict:
    return make_job(cfg, strategy, crashes, backend=args.backend, bootstrap=args.bootstrap, **kw)


def cmd_consume(args, cfg) -> int:
    _need_data()
    crashes = crash_schedule(cfg, load_manifest(), args.crash_every)
    r = run_jobs([_job(cfg, args, args.strategy, crashes, batch=args.batch, export=True)], 1)[0]
    print(f"strategy {args.strategy} | backend {args.backend} | batch {r['batch_size']} | "
          f"{len(crashes)} scheduled crashes\n")
    print(table(SCORE_HEADERS, [_row(args.strategy, r)]))
    return 0


def cmd_compare(args, cfg) -> int:
    _need_data()
    crashes = crash_schedule(cfg, load_manifest())
    names = MONEY_STRATEGIES + [n for n in PROFILE_STRATEGIES if n not in MONEY_STRATEGIES]
    jobs = [_job(cfg, args, "naive", [], label="naive (no crashes)")]
    jobs += [_job(cfg, args, n, crashes, export=True) for n in names]
    res = {r["label"]: r for r in run_jobs(jobs, args.jobs)}
    facts = stream_facts(load_key())
    print(f"backend {args.backend} | {facts['log_records']:,} log records "
          f"({facts['distinct_events']:,} distinct) | {facts['retried_events']:,} producer retries | "
          f"{facts['out_of_order_arrivals']:,} out-of-order profile updates | "
          f"{len(crashes)} consumer crashes | batch {cfg['consumer']['batch_size']}\n")
    print("Balances by strategy (money in currency units, cents underneath)")
    shown = ["naive (no crashes)"] + MONEY_STRATEGIES
    print(table(SCORE_HEADERS, [_row(n, res[n]) for n in shown]))
    print("\nProfiles by policy (all three run on the atomic sink, so balances are correct)")
    print(table(["profile policy", "profiles wrong", "accounts with a profile"],
                [[n, res[n]["score"]["profiles_wrong"], f"{facts['accounts_with_profile']:,}"]
                 for n in PROFILE_STRATEGIES]))
    _save(f"scoreboard_{args.backend}.json", {
        "facts": facts, "crashes": len(crashes),
        "strategies": {n: {"score": _public(r["score"]), "redelivered": r["stats"]["redelivered"],
                           "crashes": r["crashes"]} for n, r in res.items()}})
    return 0


def cmd_replay(args, cfg) -> int:
    _need_data()
    crashes = crash_schedule(cfg, load_manifest())
    names = REPLAY_STRATEGIES if args.strategy == "all" else [args.strategy]
    results = run_jobs([_job(cfg, args, n, crashes, kind="replay") for n in names], args.jobs)
    rows, saved = [], {}
    for p in results:
        saved[p["strategy"]] = p
        verdict = "checksum identical" if p["identical"] else "CHECKSUM DIFFERS"
        rows.append([p["strategy"], verdict, p["balances_differ"], p["profiles_differ"],
                     p["processed_differ"], p["wrong_before"], p["wrong_after"],
                     money(p["net_drift_before"]), money(p["net_drift_after"])])
    print("Rewind to offset zero, consume the whole log again, diff the sink tables\n")
    print(table(["strategy", "verdict", "balance rows differ", "profile rows differ",
                 "dedup rows differ", "wrong before", "wrong after",
                 "net drift before", "net drift after"], rows, ["l", "l"] + ["r"] * 7))
    _save(f"replay_{args.backend}.json", saved)
    return 0


def cmd_batches(args, cfg) -> int:
    _need_data()
    crashes = crash_schedule(cfg, load_manifest())
    sizes = cfg["experiments"]["batch_sizes"]
    results = run_jobs([_job(cfg, args, "atomic", crashes, label=f"atomic b={b}", batch=b)
                        for b in sizes], args.jobs)
    rows, saved = [], {}
    for b, r in zip(sizes, results):
        s, st = r["score"], r["stats"]
        txns = st["polls"] - st["crashes"]
        rows.append([b, f"{txns:,}", st["crashes"], f"{st['records_handled']:,}",
                     f"{st['redelivered']:,}", s["accounts_wrong"], s["profiles_wrong"],
                     money(s["net_drift_cents"])])
        saved[str(b)] = {"transactions": txns, **st, **_public(s)}
    print(f"Atomic sink, {len(crashes)} scheduled crashes, varying events per transaction\n")
    print(table(["batch size", "transactions", "crashes", "records handled", "redelivered",
                 "accounts wrong", "profiles wrong", "net drift"], rows))
    _save(f"batches_{args.backend}.json", saved)
    return 0


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def cmd_inspect(args, cfg) -> int:
    _need_data()
    manifest, key = load_manifest(), load_key()
    log = open_log(cfg, "file")
    feat = manifest["featured"]
    account = feat["account_id"] if args.account is None else args.account
    part = partition_for(str(account), log.n_partitions)
    crashes = set(map(tuple, crash_schedule(cfg, manifest)))

    rows, seen, newest = [], set(), 0
    for rec in log.read(part, 0, log.end_offset(part)):
        if rec.key != str(account):
            continue
        v, flags = rec.value, []
        if v["event_id"] in seen:
            flags.append("DUPLICATE (producer retry)")
        if v["type"] == "profile_updated":
            if v["version"] < newest:
                flags.append("OUT OF ORDER")
            newest = max(newest, v["version"])
        seen.add(v["event_id"])
        if (rec.partition, rec.offset) in crashes:
            flags.append("CONSUMER CRASH HERE")
        what = (f"{v['kind']} {money(v['amount_cents'])}" if v["type"] == "wallet_txn"
                else f"v{v['version']} {v['tier']} limit {money(v['daily_limit_cents'])}")
        rows.append([rec.offset, v["event_id"], v["type"], what, v["event_time"],
                     v["producer_attempt"], ", ".join(flags)])
    print(f"Account {account} lives in partition {part}"
          + ("  (the featured account)" if account == feat["account_id"] else "") + "\n")
    print(table(["offset", "event_id", "type", "event", "event_time", "attempt", "flags"],
                rows, ["r", "l", "l", "l", "r", "r", "l"]))

    truth = account_truth(key, account)
    ids = {r[1] for r in rows}
    sink_dir = ARTIFACTS_DIR / "sinks"
    if not (sink_dir / "naive_balances.csv").exists():
        print("\nNo sink exports yet, running `compare` first.\n")
        cmd_compare(argparse.Namespace(backend="file", bootstrap=None, jobs=DEFAULT_JOBS), cfg)
    out = []
    for name in ["naive", "commit_first", "dedup_separate_txn", "atomic",
                 "atomic_last_arrived_wins"]:
        bal = {int(r["account_id"]): int(r["balance_cents"])
               for r in _read_csv(sink_dir / f"{name}_balances.csv")}
        prof = {int(r["account_id"]): r for r in _read_csv(sink_dir / f"{name}_profiles.csv")}
        eff = {r["event_id"]: int(r["times_applied"]) for r in _read_csv(sink_dir / f"{name}_effects.csv")}
        mine = {e: n for e, n in eff.items() if e in ids}
        p = prof.get(account)
        out.append([name, money(bal.get(account, 0)), money(bal.get(account, 0) - truth["balance_cents"]),
                    ", ".join(f"{e} x{n}" for e, n in sorted(mine.items())) or "-",
                    f"{p['tier']} {money(int(p['daily_limit_cents']))}" if p else "-"])
    tp = truth["profile"]
    print(f"\ntruth: balance {money(truth['balance_cents'])}, profile "
          + (f"{tp['tier']} {money(tp['daily_limit_cents'])}" if tp else "none"))
    print("\nWhat each strategy computed (x0 = lost, x2 = applied twice)\n")
    print(table(["strategy", "balance", "error", "events applied wrongly", "profile"],
                out, ["l", "r", "r", "l", "l"]))
    return 0


def cmd_all(args, cfg) -> int:
    for fn, extra in ((cmd_produce, {}), (cmd_compare, {}),
                      (cmd_replay, {"strategy": "all"}), (cmd_batches, {}),
                      (cmd_inspect, {"account": None})):
        ns = argparse.Namespace(backend="file", bootstrap=None, jobs=DEFAULT_JOBS, **extra)
        print("=" * 100)
        rc = fn(ns, cfg)
        if rc:
            return rc
        print()
    return 0


# -------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run.py", description="Replay the log.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def backend(p):
        p.add_argument("--backend", choices=["file", "kafka"], default="file")
        p.add_argument("--bootstrap", help="Kafka host:port, or set KAFKA_BOOTSTRAP")
        p.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="parallel worker processes, 1 = in this process")

    p = sub.add_parser("produce", help="generate the log and the answer key")
    backend(p)
    p.set_defaults(fn=cmd_produce)

    p = sub.add_parser("consume", help="run one strategy over the log")
    p.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    p.add_argument("--crash-every", type=int, help="records per partition between crashes, 0 = none")
    p.add_argument("--batch", type=int, help="records per poll and per transaction")
    backend(p)
    p.set_defaults(fn=cmd_consume)

    p = sub.add_parser("compare", help="run every strategy and print the scoreboard")
    backend(p)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("replay", help="rewind to offset zero, replay, diff the sink")
    p.add_argument("--strategy", choices=sorted(STRATEGIES) + ["all"], default="all")
    backend(p)
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("batches", help="batch size vs redelivered records, atomic sink")
    backend(p)
    p.set_defaults(fn=cmd_batches)

    p = sub.add_parser("inspect", help="one account: its events, flags and each strategy's result")
    p.add_argument("--account", type=int, help="default is the featured account")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("all", help="produce, compare, replay, batches, inspect")
    p.set_defaults(fn=cmd_all)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    return args.fn(args, load_config())

"""Command line entry point.

    triage data          build or refresh the synthetic inbox
    triage independence  measure how false the assumption is, then find it blind
    triage strategies    the four multiclass strategies, one identical split
    triage calibrate     what calibration repairs, and what it leaves alone
    triage sweep         the headline: accuracy vs ECE as the assumption breaks
    triage route         the operating curve an ops lead would actually use
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from support_ticket_triage.config import CFG, load_config
from support_ticket_triage.utils.logging import configure, get_logger

log = get_logger(__name__)
pd.set_option("display.width", 170, "display.max_columns", 40)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "triage", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--log-level")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="build or refresh the inbox")
    d.add_argument("--n", type=int)
    d.add_argument("--refresh", action="store_true")
    d.add_argument("--dependency", type=float,
                   help="override dependency_strength (0 = assumption is true)")

    for name, helptext in (
            ("independence", "measure the assumption violation"),
            ("strategies", "native vs ovr vs ovo vs softmax"),
            ("calibrate", "repair the probabilities, keep the ranking"),
            ("sweep", "what the assumption costs as dependence rises"),
            ("route", "the threshold operating curve")):
        sp = sub.add_parser(name, help=helptext)
        # Attached per subcommand as well as globally, so both
        # `run.py --no-write sweep` and `run.py sweep --no-write` work. Argparse
        # would otherwise reject the second, which is the one people type.
        sp.add_argument("--no-write", action="store_true",
                        help="do not write CSVs to artifacts/")
    d.add_argument("--no-write", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-write", action="store_true", dest="no_write_global",
                   help="do not write CSVs to artifacts/")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else CFG
    configure(args.log_level or cfg.log_level)
    write = not (args.no_write or getattr(args, 'no_write_global', False))

    if args.cmd == "data":
        import dataclasses

        from support_ticket_triage.data.io import load_tickets
        changes = {}
        if args.n:
            changes["n_tickets"] = args.n
        if args.dependency is not None:
            changes["dependency_strength"] = args.dependency
        if changes:
            cfg = dataclasses.replace(cfg, **changes).validate()
        load_tickets(refresh=args.refresh or bool(changes), cfg=cfg)

    elif args.cmd == "independence":
        from support_ticket_triage.pipelines.compare import run_independence
        out = run_independence(cfg, write=write)
        print("\n=== THE PLANTED PAIRS (lift of 1.0 = the assumption holds) ===")
        print(out["planted"].to_string(index=False))
        print("\n=== WHAT A BLIND SURVEY FINDS (no answer key) ===")
        print(out["survey"].to_string(index=False))
        r = out["recovery"]
        print(f"\nOf the {r['surveyed']} pairs this blind sweep flagged, "
              f"{r['found']} are genuinely planted "
              f"({r['precision']:.0%} precision),")
        print(f"covering {r['recall']:.0%} of the {r['planted']} planted pairs.")
        print("Widen the list and recall rises while precision falls. That is "
              "the same trade you\nface on real data, where nothing tells you "
              "which flags were real.")
        print("\nnext: `python run.py strategies`, then `sweep` for the payoff.")

    elif args.cmd == "strategies":
        from support_ticket_triage.pipelines.compare import run_strategies
        out = run_strategies(cfg, write=write)
        print("\n=== FOUR MULTICLASS STRATEGIES (held out) ===")
        print(out["strategies"].to_string(index=False))
        print("\n=== PER CLASS, native_nb (watch the 3% class) ===")
        print(out["per_class"]["native_nb"].to_string(index=False))

    elif args.cmd == "calibrate":
        from support_ticket_triage.pipelines.compare import run_calibration
        out = run_calibration(cfg, write=write)
        print("\n=== BEFORE AND AFTER CALIBRATION ===")
        print(out["table"].to_string(index=False))
        print("\n=== DID THE DECISIONS MOVE? ===")
        for k, v in out["preserved"].items():
            print(f"  {k:28s} {v}")
        print("\n=== WHERE THE CONFIDENCE MASS SAT ===")
        print(out["shift"].to_string(index=False))
        print("\n=== RELIABILITY, UNCALIBRATED ===")
        print(out["reliability_raw"].to_string(index=False))

    elif args.cmd == "sweep":
        from support_ticket_triage.pipelines.compare import headline, run_sweep
        table = run_sweep(cfg, write=write)
        print("\n=== AS THE ASSUMPTION BREAKS ===")
        print(table.to_string(index=False))
        h = headline(table)
        print(f"\nshared decline, both models   : {h['shared_accuracy_decline']:.4f}"
              "   <- the data got harder, nobody's fault")
        print(f"cost of the assumption, worst : {h['assumption_cost_worst']:.4f}"
              "   <- what Naive Bayes actually gives up")
        print(f"dependence lift at that point : {h['lift_at_worst']:.2f}"
              "     <- 1.00 would mean the assumption holds")
        print(f"\n{h['verdict']}")

    elif args.cmd == "route":
        from support_ticket_triage.pipelines.compare import run_calibration
        out = run_calibration(cfg, write=write)
        print("\n=== WHAT THE ROUTING RULE DOES ===")
        print(out["routing"].to_string(index=False))
        print("\n=== OPERATING CURVE (calibrated model) ===")
        print(out["sweep"].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

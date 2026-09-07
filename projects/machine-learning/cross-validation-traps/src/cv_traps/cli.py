"""Command line entry point.

    cvtraps data           build or refresh the synthetic panel
    cvtraps truth          what the schemes are being scored against
    cvtraps preprocessing  fitting a transformer before the split
    cvtraps grouped        the same customer on both sides of a fold
    cvtraps temporal       training on the future to predict the past
    cvtraps selection      the winner's curse, and what nested CV costs
    cvtraps variance       the trap that is noise rather than bias
    cvtraps sweep          all four biased traps ranked by measured optimism
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from cv_traps.config import check_environment_config, load_config
from cv_traps.utils.logging import configure, get_logger

log = get_logger(__name__)
pd.set_option("display.width", 200, "display.max_columns", 40)

TRAPS = ("truth", "preprocessing", "grouped", "temporal", "selection",
         "variance", "sweep")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "cvtraps", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--log-level")
    p.add_argument("--no-write", action="store_true", dest="no_write_global",
                   help="do not write CSVs to artifacts/")
    sub = p.add_subparsers(dest="cmd", required=True)

    # The control dials belong to every command, not just `data`. Building a
    # control panel and then running a trap against the default config cannot
    # work: the trap sees a cache whose manifest disagrees with the config,
    # correctly calls it stale, and rebuilds the non-control panel underneath
    # you. Passing the same override to both keeps the two in agreement.
    dials = argparse.ArgumentParser(add_help=False)
    dials.add_argument("--group-effect", type=float,
                       help="0.0 makes customers exchangeable (group trap off)")
    dials.add_argument("--drift", type=float,
                       help="0.0 makes the process stationary (time trap off)")

    d = sub.add_parser("data", parents=[dials], help="build or refresh the panel")
    d.add_argument("--n", type=int, help="number of customers")
    d.add_argument("--refresh", action="store_true")
    d.add_argument("--no-write", action="store_true", help=argparse.SUPPRESS)

    helptext = {
        "truth": "the holdout every scheme is measured against",
        "preprocessing": "transformer fitted before vs inside the split",
        "grouped": "customer leakage across folds",
        "temporal": "training on the future",
        "selection": "the winner's curse and nested CV",
        "variance": "how far one CV number moves on reshuffling alone",
        "sweep": "all four biased traps, ranked",
    }
    for name in TRAPS:
        sp = sub.add_parser(name, parents=[dials], help=helptext[name])
        # Attached per subcommand as well as globally, so both
        # `run.py --no-write sweep` and `run.py sweep --no-write` work.
        # Argparse would otherwise reject the second, which is the one people
        # actually type.
        sp.add_argument("--no-write", action="store_true",
                        help="do not write CSVs to artifacts/")
    return p


def _print(table: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except (ValueError, FileNotFoundError) as e:
        # A misconfigured run is an ordinary mistake, not a crash. A traceback
        # here buries a message that was written to be read.
        print(f"\nerror: {e}\n", file=sys.stderr)
        return 2


def _main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    check_environment_config()
    cfg = load_config(args.config) if args.config else load_config()
    configure(args.log_level or cfg.log_level)
    write = not (args.no_write or getattr(args, "no_write_global", False))

    import dataclasses
    changes = {}
    if getattr(args, "n", None):
        changes["n_customers"] = args.n
    if args.group_effect is not None:
        changes["group_effect"] = args.group_effect
    if args.drift is not None:
        changes["drift"] = args.drift
    if changes:
        cfg = dataclasses.replace(cfg, **changes).validate()
        log.info("running the control: %s",
                 ", ".join(f"{k}={v}" for k, v in changes.items()))

    if args.cmd == "data":
        from cv_traps.data.io import load_panel
        load_panel(refresh=getattr(args, "refresh", False) or bool(changes),
                   cfg=cfg)
        return 0

    if args.cmd == "truth":
        from cv_traps.data.io import load_panel
        from cv_traps.evaluation import truth as truth_mod
        dev, out = truth_mod.split_panel(load_panel(cfg=cfg), cfg)
        print("\n=== WHAT 'CORRECT' MEANS HERE ===")
        for k, v in truth_mod.summarise(dev, out).items():
            print(f"  {k:28s} {v}")
        print("\nEvery scheme is scored by how far its estimate sits from a "
              "model\nfitted on development and tested on this holdout: unseen "
              "customers,\nfuture periods. That is what deployment is.")
        return 0

    from cv_traps.pipelines import traps

    if args.cmd == "preprocessing":
        out = traps.run_preprocessing(cfg, write=write)
        _print(out["table"], "FITTING THE TRANSFORMER BEFORE VS INSIDE THE SPLIT")
        _print(out["curve"], "THE SAME LEAK, ACROSS SAMPLE SIZES")
        print("\nPositive optimism means the scheme flattered the model.")
        print("`self_deception` is leaked CV minus honest CV on the identical")
        print("rows and folds, so it isolates the leak from everything else.")
        print("next: `python run.py grouped`")

    elif args.cmd == "grouped":
        out = traps.run_grouped(cfg, write=write)
        _print(out["table"], "THE SAME CUSTOMER ON BOTH SIDES OF A FOLD")
        _print(out["leakage"]["stratified_kfold"],
               "WHAT STRATIFIED KFOLD ACTUALLY DID, FOLD BY FOLD")
        _print(out["curve"], "THE SAME LEAK, ACROSS SAMPLE SIZES")
        print("\nUnlike the selection traps, this one does not shrink as rows")
        print("are added. It is a property of the split, not of the noise.")

    elif args.cmd == "temporal":
        out = traps.run_temporal(cfg, write=write)
        _print(out["table"], "TRAINING ON THE FUTURE TO PREDICT THE PAST")

    elif args.cmd == "selection":
        out = traps.run_selection(cfg, write=write)
        _print(out["candidates"].head(8),
               f"THE SEARCH: TOP 8 OF {cfg.n_candidates} "
               "INTERCHANGEABLE CANDIDATES, GROUPED FOLDS")
        _print(out["table"], "WHAT EACH WAY OF REPORTING IT IS WORTH")
        _print(out["curve"], "THE SAME CURSE, ACROSS SAMPLE SIZES")
        print(f"\nwinner: {out['winner']}")

    elif args.cmd == "variance":
        out = traps.run_variance(cfg, write=write)
        _print(out["table"], "THE SAME CV, RESHUFFLED")
        v = out["verdict"]
        print(f"\ngap between the two models   : {v['gap_between_models']:.4f}")
        print(f"widest spread within one model: "
              f"{v['widest_single_model_spread']:.4f}")
        print(f"A beat B in {v['times_A_beat_B_out_of_repeats']} of "
              f"{v['n_repeats']} repeats")
        print("\nA gap smaller than the spread is not a finding, it is a "
              "reshuffle."
              if v["gap_smaller_than_spread"] else
              "\nHere the gap does exceed the spread, so the ranking survives "
              "reshuffling.")

    elif args.cmd == "sweep":
        table = traps.run_sweep(cfg, write=write)
        _print(table, "EVERY TRAP, RANKED BY MEASURED OPTIMISM")
        print(f"\n{traps.headline(table)['verdict']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

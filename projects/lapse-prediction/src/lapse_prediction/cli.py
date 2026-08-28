"""Command-line entry point.

    lapse data      --n 20000 [--refresh]      build/refresh the ledger
    lapse train     [--model ordinal_chain]    train, evaluate, persist
    lapse benchmark [--only a,b]               algorithm bake-off
    lapse score     [--version latest]         write the retention queue
    lapse models                               list persisted versions
"""
from __future__ import annotations

import argparse
import json
import sys

from lapse_prediction.config import CFG, load_config
from lapse_prediction.utils.logging import configure, get_logger

log = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("lapse", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to a YAML config (default conf/config.yaml)")
    p.add_argument("--log-level", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="build or refresh the renewal ledger")
    d.add_argument("--n", type=int, default=20_000, help="synthetic policies")
    d.add_argument("--refresh", action="store_true")

    t = sub.add_parser("train", help="train, evaluate and persist a model")
    t.add_argument("--model", default="ordinal_chain",
                   choices=["ordinal_chain", "bucket"])
    t.add_argument("--n", type=int, default=20_000)
    t.add_argument("--refresh", action="store_true")
    t.add_argument("--no-persist", action="store_true")
    t.add_argument("--no-calibrate", action="store_true")

    b = sub.add_parser("benchmark", help="compare every algorithm family")
    b.add_argument("--n", type=int, default=12_000)
    b.add_argument("--only", help="comma-separated model names")
    b.add_argument("--refresh", action="store_true")

    s = sub.add_parser("score", help="score open dues into the retention queue")
    s.add_argument("--model", default="ordinal_chain")
    s.add_argument("--version", default="latest")

    m = sub.add_parser("models", help="list persisted model versions")
    m.add_argument("--model", default="ordinal_chain")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else CFG
    configure(args.log_level or cfg.log_level)

    if args.cmd == "data":
        from lapse_prediction.pipelines.train import prepare
        prepare(cfg, n_policies=args.n, refresh=args.refresh)

    elif args.cmd == "train":
        from lapse_prediction.evaluation.report import gate
        from lapse_prediction.pipelines.train import run
        out = run(args.model, cfg, n_policies=args.n, refresh=args.refresh,
                  calibrate=not args.no_calibrate, persist=not args.no_persist)
        ok, fails = gate(out["metrics"]["headline"])
        if not ok:
            log.error("RELEASE GATE FAILED: %s", "; ".join(fails))
            return 1
        log.info("release gate passed")

    elif args.cmd == "benchmark":
        from lapse_prediction.pipelines.benchmark import run
        run(cfg, n_policies=args.n,
            only=set(args.only.split(",")) if args.only else None,
            refresh=args.refresh)

    elif args.cmd == "score":
        from lapse_prediction.serving.predict import Scorer
        Scorer(args.model, args.version, cfg).score_batch()

    elif args.cmd == "models":
        from lapse_prediction.models.registry import list_versions
        versions = list_versions(args.model, cfg.model_store)
        print(json.dumps({"model": args.model, "versions": versions}, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line entry point.

    rent data                 build/refresh the synthetic market
    rent diagnose             fit the naive OLS and run all seven checks
    rent compare              the full argument: naive -> specified -> GBM
    rent recover              how close did each model get to the truth?
    rent explain              SHAP bridge + the interpretability ledger
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from rent_price_explainer.config import CFG, load_config
from rent_price_explainer.utils.logging import configure, get_logger

log = get_logger(__name__)
pd.set_option("display.width", 170, "display.max_columns", 30)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("rent", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--log-level")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="build or refresh the listings table")
    d.add_argument("--n", type=int)
    d.add_argument("--refresh", action="store_true")

    dg = sub.add_parser("diagnose", help="run the assumption checks")
    dg.add_argument("--strict", action="store_true",
                    help="exit non-zero if any check fails (for CI); off by "
                         "default because the naive model is SUPPOSED to fail")
    c = sub.add_parser("compare", help="the full naive -> specified -> GBM story")
    c.add_argument("--no-write", action="store_true")
    sub.add_parser("recover", help="score coefficients against the known truth")
    sub.add_parser("explain", help="SHAP bridge and interpretability ledger")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else CFG
    configure(args.log_level or cfg.log_level)

    if args.cmd == "data":
        import dataclasses

        from rent_price_explainer.data.io import load_listings
        if args.n:
            cfg = dataclasses.replace(cfg, n_listings=args.n).validate()
        load_listings(refresh=args.refresh, cfg=cfg)

    elif args.cmd == "diagnose":
        from rent_price_explainer.data.io import load_listings
        from rent_price_explainer.diagnostics import assumptions
        from rent_price_explainer.models.linear import NaiveOLS
        from rent_price_explainer.pipelines.compare import split

        train, _ = split(load_listings(cfg=cfg), cfg)
        m = NaiveOLS().fit(train)
        d = assumptions.run_all(m.res, m._prepare(train))
        print(assumptions.summary_frame(d["checks"]).to_string(index=False))
        print(f"\n{d['n_failed']} of {len(d['checks'])} checks FAILED "
              "-- which is the EXPECTED result for the naive model.")
        print("\nworst VIFs:\n" + d["vif_table"].head(6).to_string(index=False))
        print("\nnext: `python run.py compare` fixes exactly what failed here.")
        # Exit 0 by default: a failing naive model is the point of this command,
        # so `make diagnose` and CI should not treat it as a broken build.
        return 1 if (args.strict and d["n_failed"]) else 0

    elif args.cmd == "compare":
        from rent_price_explainer.pipelines.compare import run
        out = run(cfg, write=not args.no_write)
        print("\n=== ACCURACY (held out) ===")
        print(out["accuracy"].to_string(index=False))
        print("\n=== RECOVERY vs the known truth ===")
        print(out["recovery_summary"].to_string(index=False))

    elif args.cmd == "recover":
        from rent_price_explainer.data.io import load_listings
        from rent_price_explainer.evaluation import recovery
        from rent_price_explainer.models.linear import InteractionOLS
        from rent_price_explainer.pipelines.compare import split

        train, _ = split(load_listings(cfg=cfg), cfg)
        m = InteractionOLS().fit(train)
        r = recovery.score_recovery(m, m.name)
        print(r.to_string(index=False))
        print("\nsummary:", recovery.recovery_summary(r))
        print("\ncollinearity damage (bootstrap spread of one coefficient):")
        print(recovery.collinearity_damage(train).to_string(index=False))

    elif args.cmd == "explain":
        from rent_price_explainer.data.io import load_listings
        from rent_price_explainer.explain import shap_report
        from rent_price_explainer.models.gbm import GBM
        from rent_price_explainer.models.linear import InteractionOLS
        from rent_price_explainer.pipelines.compare import split

        train, test = split(load_listings(cfg=cfg), cfg)
        ols = InteractionOLS().fit(train)
        gbm = GBM(log_target=cfg.gbm_log_target, seed=cfg.seed).fit(train)
        print(shap_report.compare_attributions(ols, gbm, test).to_string(index=False))
        print("\n=== what each model can tell a regulator ===")
        print(shap_report.interpretability_ledger(ols, gbm).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

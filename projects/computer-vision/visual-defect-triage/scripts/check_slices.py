"""Fail the build when any slice regresses, even if the average improves.

A candidate can gain half a point overall while losing four points on the class
the customer cares about. An average-only check waves that straight through, so
this compares every slice against the committed baseline.
"""
import argparse
import csv
import sys
from pathlib import Path

ART = Path("artifacts")
BASELINE = ART / "slice_baseline.csv"
CURRENT = ART / "slice_report.csv"


def read(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["slice"]: float(r["accuracy"]) for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-regression", type=float, default=0.02)
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    current = read(CURRENT)
    if not current:
        print("no slice_report.csv, run src.run_pipeline first", file=sys.stderr)
        return 1

    if args.update_baseline or not BASELINE.exists():
        with BASELINE.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slice", "accuracy"])
            for k, v in sorted(current.items()):
                w.writerow([k, v])
        print(f"baseline written with {len(current)} slices")
        return 0

    baseline = read(BASELINE)
    failures = []
    for name, was in baseline.items():
        now = current.get(name)
        if now is None:
            failures.append(f"slice {name} disappeared from the report")
        elif was - now > args.max_regression:
            failures.append(f"{name}: {was:.4f} -> {now:.4f} "
                            f"(-{was - now:.4f}, limit {args.max_regression})")

    if failures:
        print("SLICE REGRESSION", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(f"all {len(baseline)} slices within {args.max_regression} of baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Jev decision lab. Decide whether a typed decision model fits your task.

    uv run python run.py compare       one holistic question against five narrow ones
    uv run python run.py calibration   what the confidence numbers are actually worth
    uv run python run.py threshold     the band you can automate, and what it covers
    uv run python run.py primitives    one Noul, one Choice and one Score, printed raw

Runs offline with no API key. Add --backend live, with TYPESAFE_API_KEY set and
`typesafe-sdk` installed, to run the same harness against the real model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backends import get_backend  # noqa: E402
from src.calibrate import operating_point, reliability  # noqa: E402
from src.experiment import collect, compare, state_of  # noqa: E402
from src.questions import PRIMITIVE_DEMO  # noqa: E402
from src.task import SIGNALS, make_tickets, split  # noqa: E402

BANNER = """\
  ----------------------------------------------------------------------------
  Backend: simulated. No network, no key, no cost.

  These tickets are generated and their labels are computed from the five
  signals, so decomposition wins here by construction. That is the harness
  working, not a finding about any real model. Run --backend live against your
  own labelled data for a number you can act on.
  ----------------------------------------------------------------------------
"""


def cmd_compare(args, backend) -> int:
    tickets = make_tickets(args.n, seed=args.seed)
    arms = compare(backend, tickets, target=args.target)
    train, test = split(tickets)

    print(f"{len(tickets)} tickets, {len(train)} to fit on, {len(test)} held out.")
    print(f"Escalation rate {sum(t.escalate for t in tickets) / len(tickets):.1%}.\n")

    print(f"{'arm':<12}{'questions':>10}{'accuracy':>10}{'ECE raw':>9}{'+temp':>9}{'+platt':>9}")
    print("-" * 59)
    for arm in arms:
        print(
            f"{arm.name:<12}{arm.questions_per_call:>10}{arm.accuracy:>10.3f}"
            f"{arm.ece_raw:>9.3f}{arm.ece_temperature:>9.3f}{arm.ece_platt:>9.3f}"
        )
    print()
    print("Lower ECE is better. Temperature only rescales, so it cannot correct a")
    print("model that leaned the wrong way. Platt scaling adds the shift that can.")

    print("\nWhat the fitted regression says each signal is worth:")
    decomposed = arms[1]
    if decomposed.model:
        for name, weight in sorted(
            zip(decomposed.model.features, decomposed.model.weights), key=lambda r: -abs(r[1])
        ):
            print(f"  {name:<10}{weight:>7.2f}   {SIGNALS[name]}")

    print(f"\nBand that clears {args.target:.0%} accuracy:")
    for arm in arms:
        if arm.point:
            print(
                f"  {arm.name:<12}confidence >= {arm.point.confidence:.2f}, "
                f"covers {arm.point.coverage:.1%} of tickets at {arm.point.accuracy_in_band:.1%}"
            )
        else:
            print(f"  {arm.name:<12}no band reaches {args.target:.0%}. Everything goes to a human.")
    return 0


def cmd_calibration(args, backend) -> int:
    tickets = make_tickets(args.n, seed=args.seed)
    arms = compare(backend, tickets, target=args.target)

    for arm in arms:
        print(
            f"\n{arm.name}, after Platt scaling "
            f"(slope {arm.platt.slope:.2f}, shift {arm.platt.shift:+.2f}; "
            f"temperature alone would have been {arm.temperature:.2f})"
        )
        print(f"{'predicted':>12}{'count':>8}{'said':>8}{'happened':>10}{'gap':>8}")
        print("-" * 46)
        for b in reliability(arm.probs, arm.labels):
            if not b.count:
                continue
            gap = b.mean_confidence - b.observed_rate
            print(
                f"{b.low:.1f}-{b.high:.1f}".rjust(12)
                + f"{b.count:>8}{b.mean_confidence:>8.2f}{b.observed_rate:>10.2f}{gap:>+8.2f}"
            )
        print(f"  ECE {arm.ece_platt:.3f}  (raw {arm.ece_raw:.3f}, temperature only {arm.ece_temperature:.3f})")

    print("\nA positive gap is a promise the model did not keep on that row.")
    return 0


def cmd_threshold(args, backend) -> int:
    tickets = make_tickets(args.n, seed=args.seed)
    arms = compare(backend, tickets, target=args.target)

    print(f"{'arm':<12}{'target':>8}{'cut':>8}{'covered':>10}{'accuracy':>10}{'to humans':>12}")
    print("-" * 60)
    for arm in arms:
        for target in (0.90, 0.95, 0.99):
            point = operating_point(arm.probs, arm.labels, target)
            if point:
                print(
                    f"{arm.name:<12}{target:>8.0%}{point.confidence:>8.2f}"
                    f"{point.coverage:>10.1%}{point.accuracy_in_band:>10.1%}"
                    f"{point.total - point.handled:>12}"
                )
            else:
                print(f"{arm.name:<12}{target:>8.0%}{'-':>8}{'0.0%':>10}{'-':>10}{len(arm.labels):>12}")
    print("\nCoverage is the share you automate. The rest is the queue you still staff.")
    return 0


def cmd_primitives(args, backend) -> int:
    ticket = make_tickets(args.n, seed=args.seed)[0]
    print(f"{ticket.id}: {ticket.text}\n")
    answers = backend.ask(state_of(ticket), PRIMITIVE_DEMO.questions)
    for key, a in answers.items():
        if a.kind == "noul":
            print(f"  {key:<12} noul   {a.noul:.3f}")
        elif a.kind == "choice":
            dist = ", ".join(f"{k} {v:.2f}" for k, v in sorted(a.probabilities.items(), key=lambda r: -r[1]))
            print(f"  {key:<12} choice {a.choice}  (confidence {a.confidence:.2f}; {dist})")
        else:
            print(f"  {key:<12} score  {a.score:.2f} on a 0 to {len(PRIMITIVE_DEMO.questions[key].criteria) - 1} scale  (confidence {a.confidence:.2f})")
    return 0


COMMANDS = {
    "compare": cmd_compare,
    "calibration": cmd_calibration,
    "threshold": cmd_threshold,
    "primitives": cmd_primitives,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--backend", choices=("simulated", "live"), default="simulated")
    parser.add_argument("--n", type=int, default=600, help="how many tickets to generate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target", type=float, default=0.95, help="accuracy the automated band must clear")
    args = parser.parse_args()

    try:
        backend = get_backend(args.backend)
    except RuntimeError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    if args.backend == "simulated":
        print(BANNER)
    return COMMANDS[args.command](args, backend)


if __name__ == "__main__":
    raise SystemExit(main())

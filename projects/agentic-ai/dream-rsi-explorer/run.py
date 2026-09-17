#!/usr/bin/env python
"""Dream-RSI explorer CLI. Every command calls the live Gemini API.

    uv run python run.py explore     one real search with the hand-written policy (24 calls)
    uv run python run.py compare     Dream-RSI vs Recursive Fixed Exploration (a few hundred calls)

Run `uv sync` once first. Needs GEMINI_API_KEY in a .env file in this folder. See the README.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Lets `python path/to/run.py` work from any folder, not only from this one.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents import GeminiAgent  # noqa: E402
from src.develop import GeminiDeveloper  # noqa: E402
from src.explore import explore  # noqa: E402
from src.loop import Settings, fresh_check, run_arm  # noqa: E402
from src.policy import HAND_WRITTEN  # noqa: E402

ARTIFACTS = Path(__file__).parent / "artifacts"
DEFAULT_MODEL = "gemini-3.5-flash"
log = logging.getLogger("run")


def cmd_explore(args, client) -> int:
    print(f"Running one search with up to {args.max_calls} Gemini calls. This usually takes under a minute.\n", flush=True)
    result = explore(HAND_WRITTEN, GeminiAgent(client, args.model), seed=args.seed, max_calls=args.max_calls,
                     workers=args.workers)
    valid = sum(1 for n in result.tree.nodes.values() if n.artefact is not None)
    print(f"\npolicy          {HAND_WRITTEN.label()}")
    print(f"rounds          {result.rounds}")
    print(f"Gemini calls    {result.calls}  ({valid} valid layouts)")
    print(f"best score      {result.best_score:.4f}   (best known for 26 circles is about 2.635)")
    print("best by round   " + " ".join(f"{b:.3f}" for b in result.best_by_round))
    path = ARTIFACTS / "explore_tree.json"
    result.tree.save(path)
    print(f"tree saved to   artifacts/{path.name}")
    print("\n" + client.usage_line(args.model))
    return 0


def cmd_compare(args, client) -> int:
    settings = Settings(
        rounds=args.rounds,
        versions=args.versions,
        max_calls=args.max_calls,
        beta1=args.beta1,
        beta2=args.beta2,
        seed=args.seed,
        fresh_seeds=args.fresh_seeds,
        workers=args.workers,
        artifacts=ARTIFACTS / "trees",
    )
    searches = settings.rounds + (settings.rounds - 1) + 2 * settings.fresh_seeds
    upper = searches * settings.max_calls + settings.rounds * settings.versions
    print(f"Up to {upper} Gemini calls ({searches} searches of up to {settings.max_calls} calls, "
          f"plus up to {settings.rounds * settings.versions} policy-writing calls).")
    print("Expect several minutes. Progress is printed as it goes.\n", flush=True)

    agent = GeminiAgent(client, args.model)
    developer = GeminiDeveloper(client, args.model, beta1=settings.beta1, beta2=settings.beta2)

    # Both arms start with the same policy, so they share one round-0 search.
    log.info("round 0, shared by both arms, exploring with %s", HAND_WRITTEN.label())
    round0 = explore(HAND_WRITTEN, agent, seed=settings.seed * 1000, max_calls=settings.max_calls,
                     workers=settings.workers)
    fixed = run_arm("fixed", HAND_WRITTEN, agent, settings, first_round=round0)
    dreamt = run_arm("dream", HAND_WRITTEN, agent, settings, developer=developer, first_round=round0)

    print("\nOnline rounds")
    print(f"{'round':<7}{'fixed best':>11}{'calls':>7}   {'dream best':>11}{'calls':>7}   policy used by dream")
    for f, d in zip(fixed.rounds, dreamt.rounds):
        print(f"{f.round:<7}{f.best_score:>11.4f}{f.calls:>7}   {d.best_score:>11.4f}{d.calls:>7}   {d.policy.label()}")
    print(f"{'total':<7}{fixed.best_score:>11.4f}{fixed.total_calls:>7}   {dreamt.best_score:>11.4f}{dreamt.total_calls:>7}")

    for d in dreamt.rounds:
        print(f"\nDreaming after round {d.round}")
        for note in d.dream_notes:
            print("  " + note)

    log.info("fresh check, hand-written policy")
    hand = fresh_check(HAND_WRITTEN, agent, settings)
    log.info("fresh check, final dream policy")
    final = fresh_check(dreamt.final_policy, agent, settings)
    predicted = dreamt.rounds[-1].chosen_replay_value

    print(f"\nFresh check, {settings.fresh_seeds} real searches per policy on unseen seeds")
    print(f"value = best - {settings.beta1} * calls + {settings.beta2} * calls per round")
    print(f"{'':<14}{'policy':<32}{'best':>8}{'calls':>8}{'value':>9}")
    for name, c in (("hand-written", hand), ("dream final", final)):
        print(f"{name:<14}{c.policy.label():<32}{c.mean_best:>8.4f}{c.mean_calls:>8.1f}{c.mean_value:>9.4f} +/- {c.value_se:.4f}")
    if predicted is not None:
        print(f"\nReplay predicted {predicted:.4f} for the final policy. Real runs gave {final.mean_value:.4f}.")
    print("\n" + client.usage_line(args.model))

    out = ARTIFACTS / "compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items()},
                "arms": {
                    arm.name: {
                        "total_calls": arm.total_calls,
                        "best_score": arm.best_score,
                        "final_policy": arm.final_policy.to_dict(),
                        "rounds": [
                            {
                                "round": r.round,
                                "policy": r.policy.to_dict(),
                                "best": r.best_score,
                                "calls": r.calls,
                                "dream_notes": r.dream_notes,
                                "chosen_replay_value": r.chosen_replay_value,
                            }
                            for r in arm.rounds
                        ],
                    }
                    for arm in (fixed, dreamt)
                },
                "fresh_check": {name: asdict(c) for name, c in (("hand_written", hand), ("dream_final", final))},
                "usage": {"calls": client.calls, "input_tokens": client.input_tokens,
                          "output_tokens": client.output_tokens, "failures": client.failures},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print("Full results saved to artifacts/compare.json")
    return 0


def add_global_options(parser: argparse.ArgumentParser, defaults: bool) -> None:
    """Options that work before or after the command name.

    `python run.py --seed 3 explore` and `python run.py explore --seed 3` both
    work. The subcommand copy uses SUPPRESS, so it only overrides when given.
    """
    d = (lambda v: v) if defaults else (lambda v: argparse.SUPPRESS)
    parser.add_argument("--model", default=d(DEFAULT_MODEL), help=f"Gemini model (default {DEFAULT_MODEL})")
    parser.add_argument("--seed", type=int, default=d(7))
    parser.add_argument("--max-calls", type=int, default=d(24), help="Gemini-call cap for one search (default 24)")
    parser.add_argument("--workers", type=int, default=d(8), help="parallel Gemini calls (lower this if you hit rate limits)")
    parser.add_argument("-q", "--quiet", action="store_true", default=d(False), help="hide progress lines")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run.py", description="Dream-RSI: exploration policies that learn from replay.")
    add_global_options(parser, defaults=True)
    shared = argparse.ArgumentParser(add_help=False)
    add_global_options(shared, defaults=False)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("explore", parents=[shared], help="one real search with the hand-written policy")
    p = sub.add_parser("compare", parents=[shared], help="Dream-RSI vs Recursive Fixed Exploration")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--versions", type=int, default=4, help="policy versions Gemini writes per dreaming phase")
    p.add_argument("--beta1", type=float, default=0.004, help="replay penalty per call")
    p.add_argument("--beta2", type=float, default=0.002, help="replay reward per call per round")
    p.add_argument("--fresh-seeds", type=int, default=3, help="real searches per policy in the final check")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    for noisy in ("httpx", "google_genai", "google_genai.models", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        from src.gemini import GeminiClient

        client = GeminiClient()
    except ModuleNotFoundError:
        print("google-genai is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"{exc}\n\nGet a free key at https://aistudio.google.com/apikey", file=sys.stderr)
        return 2

    return {"explore": cmd_explore, "compare": cmd_compare}[args.command](args, client)


if __name__ == "__main__":
    sys.exit(main())

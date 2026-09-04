#!/usr/bin/env python3
"""Run this project without installing it, and without an API key.

`make` is not present on a default Windows install, and the full app needs two
terminals, Node, and a Gemini key before anything happens on screen. This gives
you one command that runs the whole moderation graph in about a second:

    python run.py demo

Commands
--------
    demo        Moderate five posts through the full six-agent pipeline.
    fast        The same posts through the single-pass fast path, for contrast.
    keywords    Show what the keyword layer alone thinks, and where it is wrong.
    test        Run the test suite.
    serve       Start the API (this one does want a key for real analysis).

Everything except `serve` runs offline against a deterministic stand-in for the
model. The stand-in is a rule engine, not a model, and the output says so.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

# The database and the Chroma store are opened on paths relative to the working
# directory ("./databases/..."), the way main.py is started from inside
# backend/. Running this script from the project root therefore created a
# SECOND, empty databases/ directory next to the real one. Move first, so every
# path resolves exactly as it does for the server.
os.chdir(BACKEND)


# --------------------------------------------------------------------------
# The five posts. These are chosen to take DIFFERENT routes through the graph.
# A demo where every input ends the same way teaches nothing about routing.
# --------------------------------------------------------------------------
SAMPLES = [
    {
        "label": "plainly fine",
        "text": "Thanks for writing this up, the section on retries was really helpful.",
        "expect": "approved after five agents, and no human ever sees it",
    },
    {
        "label": "the greeting that used to be flagged",
        "text": "Hello everyone, I am grateful for the offer. Let us discuss the strategy at the office.",
        "expect": "approved. This scored 0.65 and was flagged before the word-boundary fix",
    },
    {
        "label": "plainly abusive",
        "text": "You are all worthless idiots and I hope every one of you gets hurt.",
        "expect": "flagged by agent one and sent straight to the moderator queue",
    },
    {
        "label": "toxic words, innocent use",
        "text": "This documentary about the massacre at Wounded Knee is essential viewing for history students.",
        "expect": "the keyword layer scores it 0.30; the context pass approves it anyway",
    },
    {
        "label": "repeat offender, mild post",
        "text": "You are being an idiot about this.",
        "expect": "suspended. The same words from a new account are only approved",
        "violations": 4,
        "warnings": 3,
        "reputation": 0.25,
    },
    {
        "label": "same mild post, verified account",
        "text": "You are being an idiot about this.",
        "expect": "held for a human, because a 50k-follower verified account is a HITL trigger",
        "verified": True,
        "followers": 50000,
    },
]


def _banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def _quiet_logs() -> None:
    """The agents log heavily at INFO. Useful when debugging, noise in a demo."""
    import logging

    logging.getLogger().setLevel(logging.WARNING)
    for name in list(logging.root.manager.loggerDict):
        logging.getLogger(name).setLevel(logging.WARNING)


def cmd_demo(args) -> int:
    os.environ.setdefault("MODERATION_OFFLINE", "1")
    os.environ.setdefault("ENABLE_FAST_MODE", "false")

    from src.agents.workflow import create_moderation_workflow, process_content
    from src.core.state_factory import build_initial_state, demo_user
    from src.database.moderation_db import ModerationDatabase

    if not args.verbose:
        _quiet_logs()

    _banner("SIX-AGENT MODERATION PIPELINE (offline, no API key)")
    print("Replies come from a deterministic rule engine, not a model.")
    print("Toxicity numbers are this project's own keyword detector.\n")

    db = ModerationDatabase()
    graph = create_moderation_workflow(
        db, use_checkpointer=False, enable_fast_mode=False
    )

    rows = []
    for sample in SAMPLES:
        if sample.get("violations"):
            username, user_id = "repeat_offender", "demo-repeat"
        elif sample.get("verified"):
            username, user_id = "verified_creator", "demo-verified"
        else:
            username, user_id = "demo_user", "demo-clean"

        profile = demo_user(
            username=username,
            user_id=user_id,
            total_violations=sample.get("violations", 0),
            previous_warnings=sample.get("warnings", 0),
            reputation_score=sample.get("reputation", 0.75),
            verified=sample.get("verified", False),
            follower_count=sample.get("followers", 120),
        )
        state = build_initial_state(sample["text"], user_profile=profile)
        final = process_content(graph, state)

        rows.append(
            {
                "label": sample["label"],
                "expect": sample["expect"],
                "text": sample["text"],
                "status": final.get("status"),
                "score": final.get("toxicity_score"),
                "action": final.get("moderation_action"),
                "hitl": bool(final.get("hitl_required")),
                "agents": len(final.get("agent_decisions") or []),
                "reason": (final.get("action_reason") or "").strip(),
            }
        )

    _banner("RESULTS")
    for row in rows:
        score = row["score"]
        print(f"\n{row['label'].upper()}")
        print(f"  post      {row['text'][:70]}")
        print(f"  expected  {row['expect']}")
        print(
            f"  outcome   status={row['status']} "
            f"score={score if score is None else format(score, '.2f')} "
            f"hitl={'yes' if row['hitl'] else 'no'} agents={row['agents']}"
        )
        if row["reason"]:
            print(f"  told user {row['reason'][:100]}")

    _banner("WHAT TO NOTICE")
    print("1. Six posts, six different routes. Look at the agent counts: one post")
    print("   used a single agent, one used six. The graph is not a queue that")
    print("   every item walks end to end.")
    print("2. The abusive post got the FEWEST agents. Agent one flagged it and the")
    print("   graph exited to the moderator queue, because five more model calls")
    print("   cannot improve on a decision a human has to make anyway.")
    print("3. The last two posts are the SAME SENTENCE. One was suspended and one")
    print("   was held for a human. Only the account attached to it differed.")
    print("4. Every toxicity score here was computed before a model was consulted.")
    print("   The model's job in this system is context, not detection.")
    print("\nRun 'python run.py keywords' to see the layer that gets this wrong,")
    print("and 'python run.py fast' for the single-pass path.")
    return 0


def cmd_fast(args) -> int:
    os.environ.setdefault("MODERATION_OFFLINE", "1")
    os.environ["ENABLE_FAST_MODE"] = "true"
    os.environ["FAST_MODE_MAX_LENGTH"] = "200"

    from src.agents.workflow import create_moderation_workflow, process_content
    from src.core.state_factory import build_initial_state, demo_user
    from src.database.moderation_db import ModerationDatabase

    if not args.verbose:
        _quiet_logs()

    _banner("FAST PATH: one LLM call instead of six agents")
    print("Short comments skip the full pipeline. Same posts, different route.\n")

    db = ModerationDatabase()
    graph = create_moderation_workflow(
        db, use_checkpointer=False, enable_fast_mode=True
    )
    for sample in SAMPLES:
        state = build_initial_state(sample["text"], user_profile=demo_user())
        final = process_content(graph, state)
        agents = len(final.get("agent_decisions") or [])
        score = final.get("toxicity_score")
        print(
            f"  {sample['label'][:38]:<40} status={final.get('status'):<14} "
            f"score={score if score is None else format(score, '.2f'):<6} agents={agents}"
        )
    print("\nOne agent each, against one to six on the full path. That is the")
    print("trade: latency and cost against depth of review.")
    print("\nNow compare the OUTCOMES with 'python run.py demo'. The repeat")
    print("offender is approved here and suspended there, and the verified")
    print("account is approved here and held for a human there. The fast path")
    print("never loads reputation or evaluates HITL triggers, so the account")
    print("behind the post is invisible to it. That is not a bug, it is the")
    print("cost of the shortcut, and it is why routing short comments here is")
    print("a policy decision rather than an optimisation.")
    return 0


def cmd_keywords(args) -> int:
    from src.ml.keyword_detectors import keyword_toxicity_detection

    _banner("WHAT THE KEYWORD LAYER ALONE SEES")
    print("This layer runs before any model. It is fast, free and deterministic,")
    print("and it cannot tell use from mention.\n")

    cases = [
        ("Hello everyone, I am grateful for the offer.", "greeting"),
        ("The class passed the assessment.", "school report"),
        ("This documentary about the massacre is essential viewing.", "history"),
        ("We should hack together a demo and take out a subscription.", "shop talk"),
        ("You are all worthless idiots.", "actual insult"),
        ("I will find you and hurt you badly, you pathetic loser.", "actual threat"),
    ]
    for text, kind in cases:
        result = keyword_toxicity_detection(text)
        verdict = "FLAG" if result["toxicity_score"] >= 0.3 else "ok  "
        print(
            f"  {verdict}  {result['toxicity_score']:.2f}  {kind:<14} "
            f"{','.join(result['categories'])}"
        )
        print(f"          {text}")

    print("\nThe first two used to fire and no longer do: 'Hello' contains 'hell'")
    print("and 'class' contains 'ass', and this layer was matching substrings.")
    print("\nBut look at the ordering that remains. 'Shop talk' scores 0.60 while")
    print("an actual insult scores 0.40, because 'hack' and 'take out' are both")
    print("on the threat list and both have an everyday meaning. Word boundaries")
    print("fixed a mechanical bug. They cannot fix ambiguity, and no keyword list")
    print("can. That residue is the entire reason there is a model behind this.")
    return 0


def cmd_test(args) -> int:
    return subprocess.call(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--ignore=tests/test_e2e.py"],
        cwd=str(BACKEND),
    )


def cmd_serve(args) -> int:
    print("Starting the API on http://localhost:8000 (docs at /docs).")
    if not os.getenv("GOOGLE_API_KEY"):
        print("No GOOGLE_API_KEY found, so the API will serve offline replies too.")
    return subprocess.call(
        [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"],
        cwd=str(BACKEND),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show agent logs")
    subs = parser.add_subparsers(dest="command")
    for name, fn, help_text in (
        ("demo", cmd_demo, "moderate five posts through the six-agent pipeline"),
        ("fast", cmd_fast, "the same posts through the single-pass fast path"),
        ("keywords", cmd_keywords, "what the keyword layer alone sees"),
        ("test", cmd_test, "run the test suite"),
        ("serve", cmd_serve, "start the API"),
    ):
        sub = subs.add_parser(name, help=help_text)
        sub.set_defaults(func=fn)
        sub.add_argument("-v", "--verbose", action="store_true", help="show agent logs")

    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

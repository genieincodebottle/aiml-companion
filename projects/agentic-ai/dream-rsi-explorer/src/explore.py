"""Stage 1, online exploration. The current policy steers the real discovery agent.

Every attempt costs one Gemini call and becomes a node in the discovery tree. The
tree is the output that matters: it becomes a replay simulator for stage 3.

Attempts inside one round run in parallel, the way the paper's batch decisions
intend. Results are added to the tree in submission order, so the tree layout
does not depend on which call happened to finish first.
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .agents import DiscoveryAgent
from .policy import PolicySpec, View, decide
from .task import evaluate
from .tree import ROOT, DiscoveryTree

log = logging.getLogger(__name__)


@dataclass
class ExploreResult:
    tree: DiscoveryTree
    calls: int
    rounds: int
    best_score: float
    best_by_round: list[float]


def _attempt(agent: DiscoveryAgent, parent, rng: random.Random):
    try:
        layout = agent.propose(parent, rng)
        return layout, evaluate(layout)
    except Exception as exc:  # a live model can fail in many ways; record it, keep going
        log.warning("agent call failed: %s", exc)
        return None, None


def explore(
    spec: PolicySpec, agent: DiscoveryAgent, *, seed: int, max_calls: int, workers: int = 8
) -> ExploreResult:
    tree = DiscoveryTree()
    best_by_round: list[float] = []
    calls = 0
    rounds = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rnd in range(spec.max_rounds):
            leaves = [(n, tree.nodes[n].score) for n in tree.leaves()]
            decision = decide(spec, View(round=rnd, leaves=leaves, best_by_round=best_by_round))
            if not decision or calls >= max_calls:
                break

            jobs = []
            for node_id, attempts in decision.items():
                parent = None if node_id == ROOT else tree.nodes[node_id]
                start = len(tree.nodes[node_id].children)
                for k in range(attempts):
                    if calls + len(jobs) >= max_calls:
                        break
                    # Keyed by position in the search (path from the root plus child
                    # index), so a re-run with the same seed asks the same questions.
                    rng = random.Random(f"{seed}:{tree.path(node_id)}:{start + k}")
                    jobs.append((node_id, pool.submit(_attempt, agent, parent, rng)))

            for node_id, future in jobs:
                layout, result = future.result()
                if result is None:
                    tree.add(node_id, 0.0, None, "agent call failed", rnd)
                else:
                    tree.add(node_id, result.score, layout if result.valid else None, result.diagnostics, rnd)
            calls += len(jobs)
            rounds += 1
            best_by_round.append(tree.best().score)
            log.info("  round %d  %d attempts  best %.4f", rnd, len(jobs), best_by_round[-1])

    return ExploreResult(tree, calls, rounds, tree.best().score, best_by_round)

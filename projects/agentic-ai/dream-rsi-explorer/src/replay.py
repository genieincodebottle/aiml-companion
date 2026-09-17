"""Stages 2 and 3, a recorded tree used as a replay simulator.

A policy walks a recorded tree exactly as it would walk a live search. The only
difference is where children come from. Picking a node returns the next child
that was recorded for it, in creation order. When nothing recorded is left, it
returns nothing. Replay never calls an agent and never invents a result.

Replay score, from the paper:

    V = best score found - beta1 * N + beta2 * N / max(1, k)

N is the number of revealed attempts, each of which stood for a real agent call.
k is the number of completed rounds, so N / k rewards running attempts in
parallel, which finishes a search in fewer rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .policy import PolicySpec, View, decide
from .tree import ROOT, DiscoveryTree


@dataclass
class ReplayResult:
    value: float
    best_score: float
    revealed: int
    rounds: int
    best_by_round: list[float] = field(default_factory=list)
    calls_after_last_gain: int = 0
    empty_picks: int = 0


def replay(spec: PolicySpec, tree: DiscoveryTree, *, beta1: float, beta2: float) -> ReplayResult:
    revealed_children: dict[int, list[int]] = {ROOT: []}
    best = 0.0
    best_by_round: list[float] = []
    revealed = 0
    rounds = 0
    last_gain_at = 0
    empty_picks = 0

    for rnd in range(spec.max_rounds):
        leaves = [
            (n, tree.nodes[n].score)
            for n, kids in revealed_children.items()
            if n != ROOT and not kids
        ]
        decision = decide(spec, View(round=rnd, leaves=leaves, best_by_round=best_by_round))
        if not decision:
            break

        revealed_this_round = 0
        for node_id, attempts in decision.items():
            recorded = tree.nodes[node_id].children
            shown = revealed_children[node_id]
            new = recorded[len(shown): len(shown) + attempts]
            if len(new) < attempts:
                empty_picks += attempts - len(new)
            for child in new:
                shown.append(child)
                revealed_children[child] = []
                revealed += 1
                revealed_this_round += 1
                if tree.nodes[child].score > best + 1e-12:
                    best = tree.nodes[child].score
                    last_gain_at = revealed

        rounds += 1
        best_by_round.append(best)
        if revealed_this_round == 0:
            # Nothing recorded is reachable any more. A live run would have
            # called the agent here; replay has no result to give.
            break

    value = best - beta1 * revealed + beta2 * revealed / max(1, rounds)
    return ReplayResult(
        value=value,
        best_score=best,
        revealed=revealed,
        rounds=rounds,
        best_by_round=best_by_round,
        calls_after_last_gain=revealed - last_gain_at,
        empty_picks=empty_picks,
    )


@dataclass
class PoolScore:
    spec: PolicySpec
    value: float
    best_score: float
    revealed: float
    waste: float          # share of revealed attempts after the last improvement
    empty_picks: float    # picks that found no recorded child, a coverage signal


def score_on_pool(spec: PolicySpec, pool: list[DiscoveryTree], *, beta1: float, beta2: float) -> PoolScore:
    results = [replay(spec, t, beta1=beta1, beta2=beta2) for t in pool]
    n = len(results)
    return PoolScore(
        spec=spec,
        value=sum(r.value for r in results) / n,
        best_score=sum(r.best_score for r in results) / n,
        revealed=sum(r.revealed for r in results) / n,
        waste=sum(r.calls_after_last_gain / max(1, r.revealed) for r in results) / n,
        empty_picks=sum(r.empty_picks for r in results) / n,
    )

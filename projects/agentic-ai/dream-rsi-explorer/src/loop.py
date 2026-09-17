"""The recursive loop, and the baseline it has to beat.

Dream-RSI, round t:
    1. explore for real with policy t, record the tree
    2. add the tree to the replay pool
    3. dream: Gemini writes policy versions, replay scores them, keep the best

Recursive Fixed Exploration runs the same loop with the same model and the same
seeds, and skips step 3. It is the fair baseline, because the only difference
between the two arms is whether the policy learns. Round 0 uses the same policy
in both arms, so both arms share one round-0 search instead of paying for two.

A live model is not deterministic, so two arms on the same seed still see
different answers. That noise is why the final word goes to a fresh check: real
searches on seeds neither arm used during the loop.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .agents import DiscoveryAgent
from .develop import dream
from .explore import ExploreResult, explore
from .policy import PolicySpec
from .tree import DiscoveryTree

log = logging.getLogger(__name__)


@dataclass
class Settings:
    rounds: int = 3
    versions: int = 4           # M, policy versions per dreaming phase
    max_calls: int = 24         # Gemini-call cap for one online search
    beta1: float = 0.004        # cost per revealed attempt
    beta2: float = 0.002        # reward per attempt per round
    seed: int = 7
    fresh_seeds: int = 3        # real searches per policy in the final check
    workers: int = 8            # parallel Gemini calls within a round
    artifacts: Path | None = None


@dataclass
class RoundRecord:
    round: int
    policy: PolicySpec
    best_score: float
    calls: int
    dream_notes: list[str] = field(default_factory=list)
    chosen_replay_value: float | None = None   # what replay predicted for the next policy


@dataclass
class ArmResult:
    name: str
    rounds: list[RoundRecord]
    final_policy: PolicySpec

    @property
    def total_calls(self) -> int:
        return sum(r.calls for r in self.rounds)

    @property
    def best_score(self) -> float:
        return max(r.best_score for r in self.rounds)


def run_arm(
    name: str,
    initial: PolicySpec,
    agent: DiscoveryAgent,
    settings: Settings,
    developer=None,
    first_round: ExploreResult | None = None,
) -> ArmResult:
    spec = initial
    pool: list[DiscoveryTree] = []
    records: list[RoundRecord] = []

    for t in range(settings.rounds):
        if t == 0 and first_round is not None:
            result = first_round
        else:
            log.info("%s round %d, exploring with %s", name, t, spec.label())
            result = explore(
                spec, agent, seed=settings.seed * 1000 + t, max_calls=settings.max_calls, workers=settings.workers
            )
        pool.append(result.tree)
        if settings.artifacts:
            result.tree.save(settings.artifacts / name / f"round_{t}.json")
        record = RoundRecord(t, spec, result.best_score, result.calls)

        if developer is not None:
            log.info("%s round %d, dreaming over %d recorded trees", name, t, len(pool))
            d = dream(spec, pool, developer, versions=settings.versions, beta1=settings.beta1, beta2=settings.beta2)
            record.dream_notes = d.notes
            record.chosen_replay_value = max(v.value for v in d.tried)
            spec = d.chosen
        records.append(record)

    return ArmResult(name, records, spec)


@dataclass
class FreshCheck:
    policy: PolicySpec
    mean_best: float
    mean_calls: float
    mean_value: float     # the replay objective, measured on real runs
    value_se: float       # standard error of mean_value across seeds


def fresh_check(spec: PolicySpec, agent: DiscoveryAgent, settings: Settings) -> FreshCheck:
    """Real searches on seeds that no arm used during the loop, run side by side.

    Reports the same objective dreaming optimised, computed from real runs,
    best - beta1 * calls + beta2 * calls / rounds. Comparing best score alone
    would ignore the cost side of the trade-off the policy was trained on.
    """
    seeds = [900_000 + settings.seed * 1000 + i for i in range(settings.fresh_seeds)]
    with ThreadPoolExecutor(max_workers=max(1, len(seeds))) as pool:
        results = list(
            pool.map(
                lambda s: explore(spec, agent, seed=s, max_calls=settings.max_calls, workers=settings.workers), seeds
            )
        )
    values = [
        r.best_score - settings.beta1 * r.calls + settings.beta2 * r.calls / max(1, r.rounds) for r in results
    ]
    n = len(results)
    mean_value = sum(values) / n
    variance = sum((v - mean_value) ** 2 for v in values) / max(1, n - 1)
    return FreshCheck(
        spec,
        mean_best=sum(r.best_score for r in results) / n,
        mean_calls=sum(r.calls for r in results) / n,
        mean_value=mean_value,
        value_se=(variance / n) ** 0.5,
    )

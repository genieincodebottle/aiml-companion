"""Stage 3, dreaming. Gemini writes policy versions, replay scores them.

The loop follows the paper. Start from the current policy. The policy-development
agent reads the replay results and writes the next version. After M versions,
keep the one with the highest average replay score across the pool of recorded
trees. The discovery agent is never called in this file, which is the whole point.

`GeminiDeveloper` is the policy-development agent. It reads the replay statistics
for every version tried so far and writes the next one. The spec it returns is
clamped to `policy.BOUNDS` before it is scored, so a model can never push a knob
outside the limits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .policy import BOUNDS, ON_STALL, PolicySpec
from .replay import PoolScore, score_on_pool
from .tree import DiscoveryTree


@dataclass
class DreamResult:
    chosen: PolicySpec
    tried: list[PoolScore]
    notes: list[str]


POLICY_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "width": {"type": "integer"},
        "attempts_per_node": {"type": "integer"},
        "restarts": {"type": "integer"},
        "patience": {"type": "integer"},
        "on_stall": {"type": "string", "enum": list(ON_STALL)},
        "stall_boost": {"type": "integer"},
        "max_stalled_rounds": {"type": "integer"},
        "max_rounds": {"type": "integer"},
    },
    # Reason first. Structured decoding writes fields in schema order, so the
    # model explains before it commits to numbers.
    "required": ["reason", *BOUNDS.keys(), "on_stall"],
}


class GeminiDeveloper:
    name = "gemini"

    def __init__(self, client, model: str, *, beta1: float, beta2: float) -> None:
        self.client = client
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2

    def propose(self, best: PoolScore, tried: list[PoolScore]) -> tuple[PolicySpec, str]:
        history = [
            {
                "spec": t.spec.to_dict(),
                "replay_value": round(t.value, 4),
                "avg_best_score": round(t.best_score, 4),
                "avg_calls": round(t.revealed, 1),
                "share_of_calls_after_last_gain": round(t.waste, 2),
                "picks_with_no_recorded_child": round(t.empty_picks, 2),
            }
            for t in tried
        ]
        prompt = (
            "You improve an exploration policy for a search that packs circles in a square. "
            "Each attempt costs one agent call. Policies are scored by replaying recorded "
            f"searches, value = best score - {self.beta1} * calls + {self.beta2} * calls per round.\n\n"
            "Knobs. width = leaves to continue from each round, best first. attempts_per_node = "
            "parallel attempts per picked leaf. restarts = fresh starts per round. patience = rounds "
            "without improvement before the search counts as stalled. on_stall = stop, widen "
            "(add stall_boost to width) or restart (add stall_boost to restarts). "
            "max_stalled_rounds = stalled rounds allowed before stopping. max_rounds = hard cap.\n"
            f"Bounds {json.dumps(BOUNDS)}\n\n"
            "Replay can only reveal attempts that were actually recorded, so a policy that asks "
            "for much more than the recorded search tried gets no credit for it.\n\n"
            f"Policies tried so far, best first\n{json.dumps(history, indent=1)}\n\n"
            "Propose one new policy that should score higher. Change one or two knobs."
        )
        data = self.client.generate_json(
            model=self.model, prompt=prompt, schema=POLICY_SCHEMA, max_output_tokens=1024, temperature=0.7
        )
        return PolicySpec.from_dict(data), str(data.get("reason", ""))[:300]


def dream(
    current: PolicySpec,
    pool: list[DiscoveryTree],
    developer,
    *,
    versions: int,
    beta1: float,
    beta2: float,
) -> DreamResult:
    tried = [score_on_pool(current, pool, beta1=beta1, beta2=beta2)]
    notes = [f"v0 current {current.label()}  value={tried[0].value:.4f}"]
    seen = {current}

    for m in range(1, versions + 1):
        best = max(tried, key=lambda t: t.value)
        ranked = sorted(tried, key=lambda t: -t.value)
        try:
            spec, why = developer.propose(best, ranked)
        except Exception as exc:  # a live developer can fail; keep what we have
            notes.append(f"v{m} developer failed: {exc}")
            continue
        if spec in seen:
            notes.append(f"v{m} duplicate of an earlier version, skipped")
            continue
        seen.add(spec)
        result = score_on_pool(spec, pool, beta1=beta1, beta2=beta2)
        tried.append(result)
        notes.append(f"v{m} {spec.label()}  value={result.value:.4f}  ({why})")

    chosen = max(tried, key=lambda t: t.value).spec
    return DreamResult(chosen=chosen, tried=tried, notes=notes)

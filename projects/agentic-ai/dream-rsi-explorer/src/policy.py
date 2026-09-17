"""The exploration policy, the only component that changes between rounds.

In the paper the policy is executable code and an LLM rewrites it. Here the
policy is a small spec interpreted by one fixed function, `decide`. That is a
deliberate simplification. It lets a live LLM edit the policy without this
project ever executing model-written code, and every edit is clamped to bounds.

The decision interface follows the paper. At each round the policy sees the
tree built so far and picks a batch from {root} + leaves. Picking the root means
starting fresh. The number attached to each pick is how many attempts to run
from that node in parallel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

from .tree import ROOT

ON_STALL = ("stop", "widen", "restart")

# (min, max) for every integer knob. A developer, rule-based or LLM, can only
# move a knob inside these limits.
BOUNDS: dict[str, tuple[int, int]] = {
    "width": (1, 6),
    "attempts_per_node": (1, 4),
    "restarts": (0, 4),
    "patience": (1, 6),
    "stall_boost": (0, 6),
    "max_stalled_rounds": (1, 6),
    "max_rounds": (2, 20),
}


@dataclass(frozen=True)
class PolicySpec:
    width: int = 2                # leaves to continue from, best first
    attempts_per_node: int = 2    # parallel attempts from each picked leaf
    restarts: int = 1             # fresh starts from the root each round
    patience: int = 6             # rounds without improvement before "stalled"
    on_stall: str = "stop"        # what to do once stalled
    stall_boost: int = 0          # extra width or restarts while stalled
    max_stalled_rounds: int = 6   # give up after this many stalled rounds in a row
    max_rounds: int = 6

    def clamp(self) -> "PolicySpec":
        values = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in BOUNDS:
                lo, hi = BOUNDS[f.name]
                v = min(hi, max(lo, int(v)))
            values[f.name] = v
        if values["on_stall"] not in ON_STALL:
            values["on_stall"] = "stop"
        return PolicySpec(**values)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PolicySpec":
        known = {f.name for f in fields(cls)}
        return replace(cls(), **{k: v for k, v in data.items() if k in known}).clamp()

    def label(self) -> str:
        return (
            f"w{self.width} a{self.attempts_per_node} r{self.restarts} "
            f"p{self.patience} {self.on_stall}+{self.stall_boost} "
            f"s{self.max_stalled_rounds} R{self.max_rounds}"
        )


# The hand-written rule most search loops start with. Best-first, a few branches
# in parallel, a fixed number of rounds, and it never adapts. It is also the
# policy Recursive Fixed Exploration keeps forever. One search with it makes
# 1 + 3 + 5 + 5 + 5 + 5 = 24 Gemini calls.
#
# It is deliberately broad. Replay can only credit behaviour the recorded trees
# contain, so a narrow starting rule leaves dreaming little to learn from.
HAND_WRITTEN = PolicySpec()


@dataclass(frozen=True)
class View:
    """What a policy is allowed to see. The same view in a real run and in replay."""

    round: int
    leaves: list[tuple[int, float]]   # (node id, score) for every current leaf
    best_by_round: list[float]        # best score seen after each completed round


def stalled_rounds(best_by_round: list[float]) -> int:
    """How many of the latest rounds brought no improvement."""
    count = 0
    for i in range(len(best_by_round) - 1, 0, -1):
        if best_by_round[i] > best_by_round[i - 1] + 1e-12:
            break
        count += 1
    return count


def decide(spec: PolicySpec, view: View) -> dict[int, int]:
    """Return {node id: attempts}. An empty dict means stop."""
    if view.round >= spec.max_rounds:
        return {}

    stalled = stalled_rounds(view.best_by_round)
    is_stalled = stalled >= spec.patience
    width, restarts = spec.width, spec.restarts

    if is_stalled:
        if spec.on_stall == "stop" or stalled - spec.patience + 1 > spec.max_stalled_rounds:
            return {}
        if spec.on_stall == "widen":
            width += spec.stall_boost
        elif spec.on_stall == "restart":
            restarts += spec.stall_boost

    decision: dict[int, int] = {}
    ranked = sorted(view.leaves, key=lambda item: (-item[1], item[0]))
    for node_id, _ in ranked[:width]:
        decision[node_id] = spec.attempts_per_node
    if restarts > 0 or not ranked:
        decision[ROOT] = max(1, restarts)
    return decision

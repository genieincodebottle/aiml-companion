"""A discovery tree: the record of one search.

Each non-root node is one attempt by the discovery agent. It keeps the parent it
grew from, its artefact, its score and the evaluator's diagnostics. Children are
stored in the order they were created, and replay depends on that order.

The root is not an attempt. Expanding the root means starting fresh.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = 0


@dataclass
class Node:
    id: int
    parent: int | None
    score: float
    artefact: dict | None
    diagnostics: str
    round: int
    children: list[int] = field(default_factory=list)


class DiscoveryTree:
    def __init__(self) -> None:
        self.nodes: dict[int, Node] = {
            ROOT: Node(id=ROOT, parent=None, score=0.0, artefact=None, diagnostics="root", round=-1)
        }

    def add(self, parent: int, score: float, artefact: dict | None, diagnostics: str, round: int) -> int:
        if parent not in self.nodes:
            raise KeyError(f"unknown parent node {parent}")
        node_id = len(self.nodes)
        self.nodes[node_id] = Node(node_id, parent, score, artefact, diagnostics, round)
        self.nodes[parent].children.append(node_id)
        return node_id

    def path(self, node_id: int) -> str:
        """Child indices from the root, e.g. "2.0.1". Stable across policies."""
        steps = []
        while node_id != ROOT:
            parent = self.nodes[node_id].parent
            steps.append(self.nodes[parent].children.index(node_id))
            node_id = parent
        return ".".join(str(s) for s in reversed(steps))

    def leaves(self) -> list[int]:
        return [n.id for n in self.nodes.values() if n.id != ROOT and not n.children]

    def best(self) -> Node:
        return max(self.nodes.values(), key=lambda n: (n.score, -n.id))

    @property
    def attempts(self) -> int:
        return len(self.nodes) - 1

    def to_dict(self) -> dict:
        return {"nodes": [asdict(n) for n in self.nodes.values()]}

    @classmethod
    def from_dict(cls, data: dict) -> "DiscoveryTree":
        tree = cls()
        tree.nodes = {n["id"]: Node(**n) for n in data["nodes"]}
        return tree

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DiscoveryTree":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

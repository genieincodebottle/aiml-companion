"""The discovery agent. It stays fixed for the whole run; only the policy changes.

`GeminiAgent` shows the model a parent layout, its score and the evaluator's
notes, and asks for better circle centres as JSON. The harness then fits exact
radii to those centres.

That split is deliberate. On the first live run, a model asked for complete
layouts (centres and radii) returned 5 invalid answers out of 6, every one from
an overlap between 0.02 and 0.16. Exact tangency is arithmetic, so it lives in
code. The model decides the arrangement, which is the part it is good at.

Scoring always happens outside the agent, in `task.evaluate`, so the agent can
never grade its own work.
"""

from __future__ import annotations

import random
from typing import Protocol

from .task import N_CIRCLES, fit_radii
from .tree import Node


class DiscoveryAgent(Protocol):
    name: str

    def propose(self, parent: Node | None, rng: random.Random) -> dict: ...


CENTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "idea": {"type": "string"},
        "centers": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
    },
    # Idea first, so the model states its plan before writing 26 coordinates.
    "required": ["idea", "centers"],
}


class GeminiAgent:
    name = "gemini"

    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    def propose(self, parent: Node | None, rng: random.Random) -> dict:
        if parent is None or parent.artefact is None:
            task = (
                f"Propose centres for {N_CIRCLES} circles inside the unit square [0,1] x [0,1]. "
                "The largest non-overlapping radii are fitted to your centres, and the score "
                "is the sum of those radii."
            )
        else:
            layout = {
                "centers": [[round(x, 5), round(y, 5)] for x, y in parent.artefact["centers"]],
                "radii": [round(r, 5) for r in parent.artefact["radii"]],
            }
            task = (
                f"Improve this layout of {N_CIRCLES} circles in the unit square. The largest "
                "non-overlapping radii are fitted to whatever centres you return, and the score "
                "is their sum. Return all centres for a layout that should score higher.\n"
                f"Current score: {parent.score}\n"
                f"Evaluator notes: {parent.diagnostics}\n"
                f"Current layout: {layout}"
            )
        rules = (
            f"Return exactly {N_CIRCLES} [x, y] centres inside the square. Circles placed too "
            "close together get tiny radii, so spread them well. State the one idea you tried "
            "in a short sentence."
        )
        data = self.client.generate_json(
            model=self.model,
            # The variation number makes parallel attempts from the same parent differ.
            prompt=f"{task}\n\n{rules}\nVariation: {rng.randint(0, 10**6)}",
            schema=CENTERS_SCHEMA,
            max_output_tokens=2048,
            temperature=1.0,
        )
        centers = [(min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))) for x, y in data["centers"]]
        return {"centers": centers, "radii": fit_radii(centers)}

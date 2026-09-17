"""Test doubles. The project itself always calls live Gemini; the tests never do.

`FakeGeminiClient` answers both kinds of request the project makes, circle
centres and policy specs, deterministically from the prompt text. That keeps the
suite fast, free and repeatable while exercising the real prompt, schema and
parsing code paths.
"""

from __future__ import annotations

import hashlib
import random
import threading

from src.task import N_CIRCLES, fit_radii


def random_layout(rng: random.Random) -> dict:
    centers = [(rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)) for _ in range(N_CIRCLES)]
    return {"centers": centers, "radii": fit_radii(centers)}


class FakeGeminiClient:
    def __init__(self, policy_reply: dict | None = None, fail_every: int = 0) -> None:
        self.prompts: list[str] = []
        self.policy_reply = policy_reply
        self.fail_every = fail_every
        self.calls = 0
        self._lock = threading.Lock()

    def generate_json(self, *, model, prompt, schema, max_output_tokens, temperature):
        with self._lock:
            self.calls += 1
            self.prompts.append(prompt)
            n = self.calls
        if self.fail_every and n % self.fail_every == 0:
            raise RuntimeError("503 UNAVAILABLE (fake)")
        rng = random.Random(hashlib.sha256(prompt.encode()).hexdigest())
        if "centers" in schema["properties"]:
            return {"idea": "spread them out", "centers": random_layout(rng)["centers"]}
        if self.policy_reply is not None:
            return dict(self.policy_reply)
        return {
            "reason": "try a narrower search",
            "width": rng.randint(1, 3),
            "attempts_per_node": rng.randint(1, 3),
            "restarts": rng.randint(0, 2),
            "patience": rng.randint(1, 6),
            "on_stall": rng.choice(["stop", "widen", "restart"]),
            "stall_boost": rng.randint(0, 2),
            "max_stalled_rounds": rng.randint(1, 6),
            "max_rounds": rng.randint(2, 8),
        }

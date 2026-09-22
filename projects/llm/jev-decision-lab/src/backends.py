"""Where answers come from. One interface, two implementations.

`LiveBackend` calls the real System One API. `SimulatedBackend` needs no
credentials and no network, so the harness runs anywhere.

Read this before you trust anything the simulated backend prints. It is a stand
in, built to exercise the harness, and it is not Jev. What it does model is one
specific mechanism, and that mechanism is the reason decomposition helps:

    A single holistic question forces the model to guess how much each piece of
    evidence should count. It has no way to know your weighting, so it applies a
    sensible outsider's one. Five narrow questions ask only what is in the text,
    and leave the weighting to a regression fitted on your labels.

So the simulated model reads every narrow signal from the text with the same
accuracy in both question sets. Nothing is handicapped. The holistic answer is
worse only because it has to commit to weights it was never told, and the
probabilities it returns are pushed through a temperature, which is the
miscalibration independent audits keep finding on hosted decision models.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from math import exp
from pathlib import Path

from .questions import ChoiceQ, NoulQ, Question, ScoreQ
from .task import PHRASES


@dataclass(frozen=True)
class Answer:
    """One typed answer. Only the field matching `kind` is populated."""

    kind: str
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None


class Backend:
    name = "backend"

    def ask(self, state: dict, questions: dict[str, Question]) -> dict[str, Answer]:
        raise NotImplementedError


def _logistic(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + exp(-x))


# What an outsider would reasonably guess each signal is worth. Compare with
# task.WEIGHTS, which is what this desk's history actually says. The gap between
# the two tables is the whole argument for decomposing.
GENERIC_WEIGHTS: dict[str, float] = {
    "outage": 1.9,
    "deadline": 0.7,
    "legal": 0.6,
    "repeat": 0.5,
    "refund": 1.6,
}
GENERIC_BIAS = -1.9

# Above 1 makes a probability more extreme than it should be. Audits of hosted
# decision models report both directions, so both appear here.
HOLISTIC_TEMPERATURE = 2.1
SIGNAL_TEMPERATURE = 0.75


class SimulatedBackend(Backend):
    """Deterministic, offline, and not evidence about any real model."""

    name = "simulated"

    def __init__(self, noise: float = 0.06, seed: int = 11) -> None:
        self.noise = noise
        self.seed = seed

    def _jitter(self, *parts: str) -> float:
        """Stable pseudo-noise in [-1, 1], so a ticket answers the same way every run."""
        digest = hashlib.sha256("|".join((str(self.seed), *parts)).encode()).digest()
        return (int.from_bytes(digest[:4], "big") / 0xFFFFFFFF) * 2 - 1

    def _read_signal(self, text: str, signal: str, ticket_id: str) -> float:
        """How strongly the text carries one signal. The same reading feeds both question sets."""
        present = any(phrase in text for phrase in PHRASES[signal])
        base = 2.6 if present else -2.6
        return _logistic((base + self.noise * 10 * self._jitter(ticket_id, signal)) * SIGNAL_TEMPERATURE)

    def ask(self, state: dict, questions: dict[str, Question]) -> dict[str, Answer]:
        text = str(state.get("ticket", ""))
        ticket_id = str(state.get("id", ""))
        out: dict[str, Answer] = {}

        for key, q in questions.items():
            if isinstance(q, NoulQ):
                if key in PHRASES:
                    out[key] = Answer(kind="noul", noul=self._read_signal(text, key, ticket_id))
                else:
                    # The holistic question. Same readings, guessed weights.
                    total = GENERIC_BIAS + sum(
                        GENERIC_WEIGHTS[s] * self._read_signal(text, s, ticket_id) for s in PHRASES
                    )
                    out[key] = Answer(kind="noul", noul=_logistic(total * HOLISTIC_TEMPERATURE))
            elif isinstance(q, ChoiceQ):
                options = list(q.criteria)
                scores = {o: self._jitter(ticket_id, key, o) for o in options}
                total = sum(exp(v * 2) for v in scores.values())
                probs = {o: exp(v * 2) / total for o, v in scores.items()}
                pick = max(probs, key=probs.get)
                out[key] = Answer(
                    kind="choice", choice=pick, probabilities=probs, confidence=probs[pick]
                )
            elif isinstance(q, ScoreQ):
                # A Score is not a level index. It is the probability-weighted mean of
                # the level numbers, so it lands between levels and carries a decimal.
                weights = [exp(self._jitter(ticket_id, key, str(i)) * 2) for i in range(len(q.criteria))]
                total = sum(weights)
                mean = sum(i * w for i, w in enumerate(weights)) / total
                out[key] = Answer(
                    kind="score",
                    score=round(mean, 2),
                    confidence=0.5 + 0.4 * abs(self._jitter(ticket_id, key, "c")),
                )
            else:
                raise TypeError(f"unknown question type for {key!r}")

        return out


def _read_text_any_encoding(path: Path) -> str:
    """PowerShell's `echo ... > .env` writes UTF-16, and Notepad adds a byte-order
    mark. Either one hides the key from a plain UTF-8 read, and the error you get
    is a missing key rather than a bad file, which is a miserable thing to debug."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def find_api_key(start: Path | None = None) -> str | None:
    """Environment first, then a .env in this folder or any folder above it."""
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    here = (start or Path(__file__)).resolve()
    for folder in [here, *here.parents]:
        env = folder / ".env"
        if env.is_file():
            for line in _read_text_any_encoding(env).splitlines():
                key, _, value = line.partition("=")
                if key.strip().removeprefix("export ").strip() == "TYPESAFE_API_KEY" and value.strip():
                    return value.strip().strip('"').strip("'")
    return None


class LiveBackend(Backend):
    """The real API. Needs `pip install typesafe-sdk` and TYPESAFE_API_KEY."""

    name = "live"

    def __init__(self) -> None:
        key = find_api_key()
        if not key:
            raise RuntimeError(
                "No TYPESAFE_API_KEY found.\n"
                "Create a file named .env in the jev-decision-lab folder with one line:\n"
                "  TYPESAFE_API_KEY=your-key-here\n"
                "Or drop --backend live to run offline, which needs no key at all."
            )
        try:
            from typesafe_sdk import TypeSafeClient  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The typesafe-sdk package is not installed. Run `pip install typesafe-sdk`, "
                "or drop --backend live to run offline."
            ) from exc

        self._client = TypeSafeClient(api_key=key)

    def ask(self, state: dict, questions: dict[str, Question]) -> dict[str, Answer]:
        from .questions import to_sdk  # noqa: PLC0415

        response = self._client.system_one(state=state, questions=to_sdk(questions))
        out: dict[str, Answer] = {}

        for key in questions:
            if key in getattr(response, "nouls", {}):
                a = response.nouls[key]
                out[key] = Answer(kind="noul", noul=float(a.noul))
            elif key in getattr(response, "choices", {}):
                a = response.choices[key]
                out[key] = Answer(
                    kind="choice",
                    choice=a.choice,
                    probabilities=dict(getattr(a, "probabilities", {}) or {}),
                    confidence=getattr(a, "confidence", None),
                )
            elif key in getattr(response, "scores", {}):
                a = response.scores[key]
                out[key] = Answer(
                    kind="score", score=float(a.score), confidence=getattr(a, "confidence", None)
                )
            else:
                raise KeyError(f"no answer returned for question {key!r}")

        return out

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            close()


def get_backend(name: str) -> Backend:
    if name == "live":
        return LiveBackend()
    if name == "simulated":
        return SimulatedBackend()
    raise ValueError(f"unknown backend {name!r}")

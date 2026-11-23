"""Episodic runbook memory - the agent learns across incidents.

Without this, every incident starts amnesiac: the same fault gets re-diagnosed
from first principles forever, and a hard-won lesson from Tuesday is worthless on
Wednesday. This module closes that loop. When an incident ends, its postmortem is
distilled into one `Episode`; when a new incident starts, the most similar past
episodes are retrieved and handed to the Diagnostician as prior art.

## Why lexical retrieval instead of embeddings

Retrieval here is BM25-flavoured lexical scoring over the incident's symptom text,
implemented in ~40 lines of standard library. That is a deliberate choice, not a
shortcut, and it is worth understanding because the same tradeoff shows up in real
RAG systems:

- Incident telemetry is **high-signal keyword text** (`OutOfMemoryError`,
  `HikariPool-1`, `SSLHandshakeException`). These are near-identifiers. Lexical
  matching is *strong* on identifiers and semantic embeddings are comparatively
  weak on them - the classic embedding failure mode is deciding that two different
  exception classes are 0.94 similar because both are "a Java error".
- It has **no extra dependency, no API cost, no cold start**, and it is
  deterministic, so tests can assert on retrieval results.
- It is **inspectable**: `explain=True` returns which terms matched, so a learner
  can see exactly why an episode was recalled.

The honest limitation: lexical retrieval misses paraphrase. "heap exhausted" will
not match "out of memory". A production system would run both and fuse the
rankings (hybrid search). `EXERCISES.md` has that as a guided exercise, and
`Episode.symptom_terms` is the seam to hook it into.

## Poisoning

Memory is an attack surface: OWASP calls this memory poisoning. An agent that
writes whatever it concluded into a store it later trusts can talk itself into a
wrong belief permanently. Two mitigations here:

- Only **terminal** incidents are recorded, with their real outcome and halt
  reason attached, so a failed run is stored as a failure and retrieved as a
  cautionary example rather than as advice.
- Every episode records `schema_version`; entries written under an older contract
  are ignored on read rather than silently misinterpreted.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..models.schemas import SCHEMA_VERSION

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Words that carry no diagnostic signal. Kept short on purpose: over-aggressive
# stopword lists throw away terms like "pool" and "timeout" that matter here.
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by for from had has have if in into is it its
    of on or that the this to was were will with not no yes we you your
    service error errors log logs metric metrics incident
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, minus stopwords and single characters."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Episode:
    """One remembered incident: what it looked like, and what actually worked."""

    episode_id: str
    scenario_id: str
    service: str
    #: Free text describing what was observed - the retrieval key.
    symptom: str
    root_cause: str
    #: The action sequence that was actually executed, in order.
    actions: list[str] = field(default_factory=list)
    #: The action(s) that produced progress. Empty when nothing worked.
    effective_actions: list[str] = field(default_factory=list)
    outcome: str = "resolved"  # resolved | escalated | failed | cancelled
    halt_reason: str | None = None
    iterations: int = 0
    #: One self-contained, reusable sentence from the postmortem.
    lesson: str = ""
    recorded_at: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

    @property
    def symptom_terms(self) -> list[str]:
        """The retrieval index for this episode (the hook for hybrid search).

        `scenario_id` is deliberately NOT indexed. It is the simulator's answer key:
        indexing it let a repeat of a scenario retrieve its own past solution by
        name, which no real incident could ever do.
        """
        return tokenize(f"{self.symptom} {self.root_cause} {self.service}")

    @property
    def is_cautionary(self) -> bool:
        """True when this episode is an example of what NOT to conclude."""
        return self.outcome != "resolved"

    def as_prompt_line(self) -> str:
        """How this episode appears to the Diagnostician."""
        verdict = "RESOLVED" if self.outcome == "resolved" else f"NOT RESOLVED ({self.outcome})"
        worked = ", ".join(self.effective_actions) if self.effective_actions else "nothing worked"
        return (
            f"[{verdict}] symptom: {self.symptom[:220]} | root cause: {self.root_cause[:160]} | "
            f"actions tried: {', '.join(self.actions) or 'none'} | what worked: {worked}"
            + (f" | lesson: {self.lesson}" if self.lesson else "")
        )


@dataclass
class Recall:
    """One retrieval hit, with its score and the terms that caused it."""

    episode: Episode
    score: float
    matched_terms: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode.episode_id,
            "scenario_id": self.episode.scenario_id,
            "service": self.episode.service,
            "outcome": self.episode.outcome,
            "root_cause": self.episode.root_cause,
            "effective_actions": self.episode.effective_actions,
            "lesson": self.episode.lesson,
            "score": round(self.score, 4),
            "matched_terms": self.matched_terms,
            "cautionary": self.episode.is_cautionary,
        }


class RunbookMemory:
    """A small, file-backed, thread-safe episodic store with BM25-ish retrieval.

    JSON on disk is intentional for a teaching project: a learner can open
    `data/runbook_memory.json` and read exactly what the agent remembers. Swapping
    this for Postgres + pgvector is an exercise, and the interface is the seam.
    """

    # BM25 parameters. k1 damps term-frequency saturation, b controls length
    # normalization; these are the standard defaults.
    _K1 = 1.5
    _B = 0.75

    def __init__(self, path: Path | str | None = None, max_entries: int = 200) -> None:
        self.path = Path(path) if path else None
        self.max_entries = max_entries
        self._episodes: list[Episode] = []
        self._lock = threading.RLock()
        if self.path is not None:
            self._load()

    # --- persistence --------------------------------------------------------

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("runbook memory at %s is unreadable (%s); starting empty", self.path, exc)
            return
        loaded, skipped = [], 0
        for entry in raw.get("episodes", []):
            if entry.get("schema_version") != SCHEMA_VERSION:
                skipped += 1
                continue
            try:
                loaded.append(Episode(**entry))
            except TypeError:
                skipped += 1
        self._episodes = loaded
        if skipped:
            logger.info("runbook memory: skipped %d episode(s) written under an older schema", skipped)

    def _flush(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"schema_version": SCHEMA_VERSION, "episodes": [asdict(e) for e in self._episodes]}
            # Write-then-rename so a crash mid-write cannot corrupt the store.
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:  # pragma: no cover - disk failure
            logger.warning("could not persist runbook memory to %s: %s", self.path, exc)

    # --- write --------------------------------------------------------------

    def record(self, episode: Episode) -> Episode:
        """Store one episode, evicting the oldest if the cap is reached."""
        with self._lock:
            self._episodes.append(episode)
            if len(self._episodes) > self.max_entries:
                self._episodes = self._episodes[-self.max_entries :]
            self._flush()
        return episode

    def clear(self) -> None:
        with self._lock:
            self._episodes = []
            self._flush()

    # --- read ---------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._episodes)

    def all(self) -> list[Episode]:
        with self._lock:
            return list(self._episodes)

    def recall(
        self,
        query: str,
        *,
        top_k: int = 3,
        exclude_incident: str | None = None,
        min_score: float = 0.05,
    ) -> list[Recall]:
        """Retrieve the most similar past episodes for a symptom description.

        Scores with BM25 over episode symptom terms. `min_score` suppresses the
        "retrieve something no matter what" failure mode: irrelevant prior art is
        worse than none, because it anchors the model on the wrong cause.
        """
        with self._lock:
            episodes = [e for e in self._episodes if e.episode_id != exclude_incident]
        if not episodes:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        docs = [e.symptom_terms for e in episodes]
        n_docs = len(docs)
        avg_len = sum(len(d) for d in docs) / n_docs
        # Document frequency per query term.
        df: dict[str, int] = {}
        for term in set(query_terms):
            df[term] = sum(1 for d in docs if term in d)

        scored: list[Recall] = []
        for episode, doc in zip(episodes, docs):
            score = 0.0
            matched: list[str] = []
            doc_len = len(doc) or 1
            for term in set(query_terms):
                tf = doc.count(term)
                if tf == 0:
                    continue
                # BM25 idf, floored at 0 so a term present in every doc cannot
                # contribute a negative score.
                idf = max(0.0, math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5)))
                denom = tf + self._K1 * (1 - self._B + self._B * doc_len / avg_len)
                score += idf * (tf * (self._K1 + 1)) / denom
                matched.append(term)
            if score > 0:
                scored.append(Recall(episode=episode, score=score, matched_terms=sorted(matched)))

        if not scored:
            return []
        # Normalize to 0..1 against the best hit so `min_score` is interpretable.
        best = max(r.score for r in scored)
        for r in scored:
            r.score = r.score / best if best else 0.0
        scored.sort(key=lambda r: (-r.score, r.episode.recorded_at))
        return [r for r in scored if r.score >= min_score][:top_k]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            episodes = list(self._episodes)
        resolved = sum(1 for e in episodes if e.outcome == "resolved")
        return {
            "episodes": len(episodes),
            "resolved": resolved,
            "cautionary": len(episodes) - resolved,
            "scenarios_seen": sorted({e.scenario_id for e in episodes}),
            "path": str(self.path) if self.path else None,
            "schema_version": SCHEMA_VERSION,
        }


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_memory: RunbookMemory | None = None
_memory_lock = threading.Lock()


def get_memory() -> RunbookMemory:
    """The process-wide runbook memory, built from settings on first use."""
    global _memory
    with _memory_lock:
        if _memory is None:
            from ..config import get_settings

            settings = get_settings()
            _memory = RunbookMemory(
                path=settings.memory_path if settings.memory_enabled else None,
                max_entries=settings.memory_max_entries,
            )
        return _memory


def reset_memory_cache() -> None:
    """Test hook: drop the memoized store."""
    global _memory
    with _memory_lock:
        _memory = None


def episode_from_state(state, postmortem: dict[str, Any] | None) -> Episode:
    """Distill a finished incident into one storable episode.

    Called only on a terminal incident, so `outcome` is the truth about the run
    rather than the agent's hope for it.
    """
    from ..models.state import ActionStatus, IncidentStatus

    leading = ""
    if state.diagnostics:
        leading = state.diagnostics[-1].get("leading_cause", "")
    evidence: list[str] = []
    for diag in state.diagnostics:
        evidence.extend(diag.get("evidence", []) or [])

    executed = [a for a in state.actions if a.status == ActionStatus.EXECUTED]
    effective = [a.action_type for a in executed if a.result and a.result.get("progress")]

    outcome = {
        IncidentStatus.RESOLVED: "resolved",
        IncidentStatus.ESCALATED: "escalated",
        IncidentStatus.CANCELLED: "cancelled",
    }.get(state.status, "failed")

    return Episode(
        episode_id=state.incident_id,
        scenario_id=state.scenario_id,
        service=state.service,
        symptom=" | ".join(filter(None, [state.triage_summary, *evidence[:4]]))[:600],
        root_cause=(postmortem or {}).get("root_cause") or leading or "unknown",
        actions=[a.action_type for a in executed],
        effective_actions=effective,
        outcome=outcome,
        halt_reason=state.halt_reason.value if state.halt_reason else None,
        iterations=state.iterations,
        lesson=(postmortem or {}).get("runbook_lesson", "") or "",
    )

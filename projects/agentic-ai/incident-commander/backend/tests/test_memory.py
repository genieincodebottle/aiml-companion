"""Episodic runbook memory: retrieval quality, poisoning resistance, and the loop.

Memory is the feature most likely to be added carelessly. Two failure modes matter
more than "does it store things":

1. **Retrieving something no matter what.** Irrelevant prior art is worse than none,
   because it anchors the model on a superficially similar cause. `min_score` and
   the relevance tests exist for that.

2. **Poisoning.** An agent that writes its conclusions into a store it later trusts
   can talk itself into a wrong belief permanently. Failed runs are therefore
   recorded AS failures and retrieved as cautionary examples.
"""

from __future__ import annotations

import asyncio
import json

from src.loop import IncidentManager
from src.memory.runbook import Episode, RunbookMemory, episode_from_state, get_memory, tokenize
from src.models.schemas import SCHEMA_VERSION
from src.models.state import EventType, IncidentStatus

from ._fake_llm import FakeLLMClient
from .conftest import drive


def _episode(eid: str, symptom: str, **kw) -> Episode:
    defaults = dict(
        scenario_id=kw.pop("scenario_id", "memory_leak"),
        service=kw.pop("service", "svc"),
        root_cause=kw.pop("root_cause", "heap exhaustion"),
        actions=kw.pop("actions", ["restart_service"]),
        effective_actions=kw.pop("effective_actions", ["restart_service"]),
        outcome=kw.pop("outcome", "resolved"),
    )
    return Episode(episode_id=eid, symptom=symptom, **defaults, **kw)


# --- retrieval --------------------------------------------------------------


def test_recalls_the_relevant_episode_not_just_any_episode():
    memory = RunbookMemory(path=None)
    memory.record(_episode("e1", "OutOfMemoryError java heap space pod OOMKilled"))
    memory.record(_episode("e2", "HikariPool connection is not available request timed out", scenario_id="db_pool"))
    memory.record(_episode("e3", "cache hit ratio collapsed thundering herd on origin", scenario_id="cache"))

    hits = memory.recall("java.lang.OutOfMemoryError heap space OOMKilled", top_k=1)
    assert len(hits) == 1
    assert hits[0].episode.episode_id == "e1"
    assert hits[0].matched_terms, "retrieval must be explainable"


def test_irrelevant_query_retrieves_nothing():
    """Returning the least-bad match on an unrelated query is an anchoring hazard."""
    memory = RunbookMemory(path=None)
    memory.record(_episode("e1", "OutOfMemoryError java heap space"))
    assert memory.recall("kubernetes ingress certificate rotation schedule", top_k=3) == []


def test_empty_memory_returns_nothing_rather_than_failing():
    assert RunbookMemory(path=None).recall("anything at all") == []


def test_scores_are_normalised_and_ordered():
    memory = RunbookMemory(path=None)
    memory.record(_episode("e1", "OutOfMemoryError java heap space OOMKilled exit 137"))
    memory.record(_episode("e2", "OutOfMemoryError only"))
    hits = memory.recall("OutOfMemoryError java heap space OOMKilled", top_k=2)
    assert hits[0].score == 1.0
    assert hits[0].score >= hits[1].score


def test_tokenizer_keeps_diagnostic_terms_and_drops_noise():
    tokens = tokenize("The service error rate for the HikariPool timeout was elevated")
    assert "hikaripool" in tokens and "timeout" in tokens
    assert "the" not in tokens and "was" not in tokens
    # 'service' and 'error' are stopwords here: they appear in literally every
    # incident, so they carry no discriminating signal.
    assert "service" not in tokens


# --- poisoning resistance ---------------------------------------------------


def test_failed_runs_are_stored_as_cautionary_not_as_advice():
    memory = RunbookMemory(path=None)
    failed = _episode(
        "e1",
        "SSLHandshakeException PKIX path validation certificate expired",
        outcome="escalated",
        effective_actions=[],
        halt_reason="verifier_escalated",
    )
    memory.record(failed)
    hit = memory.recall("SSLHandshakeException certificate expired", top_k=1)[0]
    assert hit.episode.is_cautionary
    assert "NOT RESOLVED" in hit.episode.as_prompt_line()
    assert "nothing worked" in hit.episode.as_prompt_line()


def test_episodes_from_an_older_schema_are_ignored_on_read(tmp_path):
    """A contract change must not silently poison retrieval with misread entries."""
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "episodes": [
                    {**_episode("old", "stale entry").__dict__, "schema_version": SCHEMA_VERSION - 99},
                    {**_episode("new", "current entry").__dict__},
                ],
            }
        ),
        encoding="utf-8",
    )
    memory = RunbookMemory(path=path)
    assert [e.episode_id for e in memory.all()] == ["new"]


def test_corrupt_store_starts_empty_rather_than_crashing(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert len(RunbookMemory(path=path)) == 0


# --- persistence and bounds -------------------------------------------------


def test_memory_persists_and_reloads(tmp_path):
    path = tmp_path / "memory.json"
    RunbookMemory(path=path).record(_episode("e1", "OutOfMemoryError heap"))
    assert [e.episode_id for e in RunbookMemory(path=path).all()] == ["e1"]


def test_memory_is_bounded_and_evicts_oldest():
    memory = RunbookMemory(path=None, max_entries=3)
    for i in range(6):
        memory.record(_episode(f"e{i}", "OutOfMemoryError heap"))
    assert len(memory) == 3
    assert [e.episode_id for e in memory.all()] == ["e3", "e4", "e5"]


# --- the loop integration ---------------------------------------------------


def test_incidents_are_recorded_and_recalled_on_the_next_run():
    manager = IncidentManager(llm=FakeLLMClient())
    first = asyncio.run(drive(manager, "memory_leak", "approve"))
    assert len(get_memory()) >= 1

    second = asyncio.run(drive(manager, "memory_leak", "approve"))
    assert second.recalled_episodes, "a repeat of the same fault should recall prior art"
    assert second.recalled_episodes[0]["effective_actions"] == ["restart_service"]
    assert first.incident_id != second.incident_id


def test_memory_recall_is_streamed_to_the_console():
    manager = IncidentManager(llm=FakeLLMClient())
    asyncio.run(drive(manager, "memory_leak", "approve"))
    second = asyncio.run(drive(manager, "memory_leak", "approve"))
    assert any(e.type == EventType.MEMORY_RECALL for e in second.events)


def test_an_incident_never_recalls_itself():
    manager = IncidentManager(llm=FakeLLMClient())
    state = asyncio.run(drive(manager, "memory_leak", "approve"))
    assert all(r["episode_id"] != state.incident_id for r in state.recalled_episodes)


def test_memory_can_be_disabled(env):
    """Turning it off and diffing behaviour is a supported experiment, not a hack."""
    env(IC_MEMORY_ENABLED="false")
    manager = IncidentManager(llm=FakeLLMClient())
    asyncio.run(drive(manager, "memory_leak", "approve"))
    second = asyncio.run(drive(manager, "memory_leak", "approve"))
    assert second.recalled_episodes == []


def test_episode_distillation_captures_the_real_outcome():
    """The stored outcome must be the truth about the run, not the agent's hope."""
    manager = IncidentManager(llm=FakeLLMClient())
    state = asyncio.run(drive(manager, "memory_leak", "reject"))
    episode = episode_from_state(state, state.postmortem)
    assert episode.outcome == "escalated"
    assert episode.effective_actions == []
    assert episode.halt_reason == "human_rejected"
    assert state.status == IncidentStatus.ESCALATED

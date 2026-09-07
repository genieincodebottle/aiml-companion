"""Composition root: build every service once, at startup.

This is the only place the object graph is wired. Routes receive services and
never construct them, which is what lets a test swap a service for a fake
without touching a route.

WHY SINGLETONS, AND WHY IT IS A CORRECTNESS ISSUE
=================================================
  GraphClient      Creating a Neo4j driver opens a connection pool. One per
                   request exhausts the server's connection limit within a
                   minute of normal traffic, and the symptom is an intermittent
                   ServiceUnavailable that looks like a network fault.
  LLMClient        Holds the embedding cache. A fresh instance per request
                   throws away every cached vector, so the cache never warms.
  GuardrailEngine  Holds the rate-limiter state. Rebuilding it per request
                   resets the sliding window - so the rate limiter allows
                   unlimited requests while appearing, in config and in the
                   health check, to be enabled.

That last one is worth dwelling on: it is a guardrail that reports itself as
on and enforces nothing. Object lifetime is part of a control's correctness,
not a performance detail.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.answer import AnswerEngine
from src.config import Config, get_config
from src.graph.client import GraphClient
from src.guardrails import GuardrailEngine
from src.llm import LLMClient
from src.retrieval.strategies import Retriever
from src.services import GraphService, JobService, QAService, SecurityService


@dataclass
class Services:
    config: Config
    graph_client: GraphClient
    llm: LLMClient
    guard: GuardrailEngine
    qa: QAService
    graph: GraphService
    jobs: JobService
    security: SecurityService

    _entity_names: list[str] | None = field(default=None, repr=False)

    def entity_names(self) -> list[str]:
        """Every entity name and alias in the graph, for the output validator.

        Cached for the process lifetime: the set only changes on ingestion,
        while the validator runs on every answer. `invalidate()` is called by
        the ingest job - without that, the validator would flag newly-added,
        entirely real suppliers as hallucinations.
        """
        if self._entity_names is None:
            names = [
                row["name"] for row in
                self.graph_client.run("MATCH (e:Entity) RETURN e.name AS name")
                if row["name"]
            ]
            names.extend(
                row["name"] for row in self.graph_client.run(
                    "MATCH (e:Entity) WHERE e.aliases IS NOT NULL "
                    "UNWIND e.aliases AS alias RETURN DISTINCT alias AS name"
                ) if row["name"]
            )
            self._entity_names = names
        return self._entity_names

    def invalidate(self) -> None:
        self._entity_names = None


_services: Services | None = None
_lock = threading.Lock()


def build_services(config: Config | None = None) -> Services:
    config = config or get_config()

    graph_client = GraphClient(config)
    graph_client.verify()
    llm = LLMClient(config)
    guard = GuardrailEngine(config)
    retriever = Retriever(graph_client, llm, config)
    answers = AnswerEngine(llm, config, guard)

    services = Services(
        config=config,
        graph_client=graph_client,
        llm=llm,
        guard=guard,
        qa=None,      # type: ignore[arg-type]  - set below, needs `services`
        graph=GraphService(graph=graph_client, config=config),
        jobs=None,    # type: ignore[arg-type]
        security=SecurityService(config=config, guard=guard),
    )
    # QAService takes a callable, not a snapshot, so it always sees the current
    # entity set rather than one captured before the last ingestion.
    services.qa = QAService(
        config=config, retriever=retriever, answers=answers, guard=guard,
        llm=llm, entity_names=services.entity_names,
    )
    services.jobs = JobService(
        config=config, guard=guard, on_graph_changed=services.invalidate,
    )
    return services


def set_services(services: Services | None) -> None:
    global _services
    with _lock:
        _services = services


def get_services() -> Services:
    """FastAPI dependency. Routes depend on this, never on a constructor."""
    if _services is None:
        raise RuntimeError(
            "Services are not initialised. The API builds them in its lifespan "
            "handler; if you are seeing this, startup failed - check the logs."
        )
    return _services

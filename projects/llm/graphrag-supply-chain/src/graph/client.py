"""Neo4j connection handling.

Deliberately thin.  There is no ORM and no query builder here, because the
whole point of the project is that you can read the Cypher.  What this module
does add is the three things a raw driver does not give you and that every
production Neo4j service ends up writing anyway:

  1. a connectivity check with an error message that tells you what to do,
  2. one shared driver for the process (driver creation opens a connection
     pool - creating one per query is the classic Neo4j performance bug),
  3. results returned as plain dicts, so nothing downstream holds an open
     transaction open by accident.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from ..config import Config, get_config

log = logging.getLogger(__name__)


class GraphUnavailable(RuntimeError):
    """Neo4j is not reachable, with an actionable message attached."""


class GraphClient:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        settings = self.config.neo4j
        self.database = settings.database
        self._driver: Driver = GraphDatabase.driver(
            settings.uri, auth=(settings.user, settings.password)
        )

    # ------------------------------------------------------------------ core
    def verify(self) -> None:
        try:
            self._driver.verify_connectivity()
        except AuthError as exc:
            raise GraphUnavailable(
                "Neo4j rejected the credentials.\n"
                "NEO4J_USER / NEO4J_PASSWORD in .env must match your database. "
                "If you are using the bundled docker-compose.yml the password "
                "is 'graphrag123'. If you changed it in the Neo4j Browser after "
                "first login, put the NEW password in .env - Neo4j forces a "
                "password change on first manual login and that trips most "
                "people up once."
            ) from exc
        except ServiceUnavailable as exc:
            raise GraphUnavailable(
                f"Cannot reach Neo4j at {self.config.neo4j.uri}.\n"
                "  - Using Docker?   docker compose up -d   then wait ~20s for "
                "the healthcheck to pass (docker compose ps).\n"
                "  - Using Aura?     the URI must start with neo4j+s:// and the "
                "instance must be resumed - free instances auto-pause.\n"
                "  - Check the port: Bolt is 7687, the browser UI is 7474. "
                "Pointing the driver at 7474 gives exactly this error."
            ) from exc

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Execute and fully materialise.  Small result sets only - every query
        in this project is bounded by a LIMIT for that reason."""
        with self._driver.session(database=self.database) as session:
            # Parameters go in the `parameters` dict, never as **kwargs.
            # `session.run(cypher, **params)` looks tidier and breaks the moment
            # a query has a parameter named `query`, `parameters`, `timeout` or
            # `database` - it collides with the driver's own signature and
            # raises "got multiple values for argument 'query'", which points
            # at the driver rather than at the query that caused it. The
            # full-text searches in this project genuinely use $query.
            result = session.run(cypher, parameters=params)
            return [record.data() for record in result]

    def run_write(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Same as run(), but routed as a write transaction.  On a cluster the
        distinction decides which member serves the query; on single-instance
        Neo4j it is free.  Getting it right locally means the code does not
        need changing when it moves to a cluster."""
        with self._driver.session(database=self.database) as session:
            return session.execute_write(
                lambda tx: [r.data() for r in tx.run(cypher, parameters=params)]
            )

    def run_batch(self, cypher: str, rows: Iterable[dict[str, Any]],
                  batch_size: int = 500) -> int:
        """UNWIND-based bulk write.

        This is the single most important Neo4j performance idiom in the
        project.  Writing 2,000 nodes with 2,000 separate MERGE statements
        means 2,000 network round trips and 2,000 transactions.  Sending one
        MERGE that UNWINDs a list of 500 parameter maps is one round trip and
        one transaction, and is typically 50-100x faster.

        The caller's Cypher must start with ``UNWIND $rows AS row``.
        """
        rows = list(rows)
        written = 0
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            self.run_write(cypher, rows=chunk)
            written += len(chunk)
        return written

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "GraphClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ housekeeping
    def wipe(self) -> None:
        """Delete every node and relationship in the database.

        DETACH DELETE in batches rather than one statement: a single
        transaction holding a million deletes will exhaust the heap on the 1 GB
        container this project ships with.  At this corpus size one pass is
        enough, but the loop is what you would actually write."""
        while True:
            result = self.run_write(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS n"
            )
            if not result or result[0]["n"] == 0:
                break

    def counts(self) -> dict[str, int]:
        """Label and relationship-type census - used by the UI and by the
        ingestion smoke check."""
        labels = self.run(
            "MATCH (n) UNWIND labels(n) AS label "
            "RETURN label, count(*) AS n ORDER BY n DESC"
        )
        rels = self.run(
            "MATCH ()-[r]->() RETURN type(r) AS label, count(*) AS n ORDER BY n DESC"
        )
        out = {f"node:{row['label']}": row["n"] for row in labels}
        out.update({f"rel:{row['label']}": row["n"] for row in rels})
        return out

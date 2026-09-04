"""Architecture tests: the layer boundaries are enforced, not merely described.

    app/            UI. Renders. Decides nothing.
    api/routes_*    Routing. Validates, delegates to ONE service, maps results.
    src/services/   Orchestration and policy.
    src/            Capabilities.

A layering rule that lives only in a README is a rule that decays. These tests
fail the build when it does.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
API = ROOT / "api"
APP = ROOT / "app"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestBusinessLogicIsTransportAgnostic:
    def test_src_never_imports_fastapi(self):
        """The rule that makes the boundary real.

        If a service could raise HTTPException it would be callable only from a
        web request, and the CLI, the notebook and the tests would each need
        their own copy of the same orchestration.
        """
        offenders = [
            str(path.relative_to(ROOT))
            for path in SRC.rglob("*.py")
            if any(name.split(".")[0] in {"fastapi", "starlette", "uvicorn"}
                   for name in _imports(path))
        ]
        assert not offenders, f"business logic importing a web framework: {offenders}"

    def test_src_never_imports_streamlit(self):
        offenders = [
            str(path.relative_to(ROOT))
            for path in SRC.rglob("*.py")
            if any(name.split(".")[0] == "streamlit" for name in _imports(path))
        ]
        assert not offenders, f"business logic importing the UI framework: {offenders}"


class TestUIHoldsNoBusinessLogic:
    def test_ui_never_imports_src(self):
        """The UI must reach the system only over HTTP.

        Importing `src` from the UI would let it bypass every guardrail the API
        enforces - and a control a frontend can skip is a control anyone can
        skip with curl.
        """
        offenders = [
            str(path.relative_to(ROOT))
            for path in APP.rglob("*.py")
            if any(name.split(".")[0] in {"src", "api"} for name in _imports(path))
        ]
        assert not offenders, f"UI importing business logic directly: {offenders}"

    def test_ui_never_imports_a_database_or_model_driver(self):
        offenders = [
            str(path.relative_to(ROOT))
            for path in APP.rglob("*.py")
            if any(name.split(".")[0] in {"neo4j", "google"} for name in _imports(path))
        ]
        assert not offenders, f"UI talking to infrastructure directly: {offenders}"

    def test_ui_contains_no_cypher(self):
        """A MATCH clause in the UI means a query that skipped the read-only
        guard and the service layer."""
        pattern = re.compile(r"\b(MATCH|MERGE|UNWIND)\s*\(", re.IGNORECASE)
        offenders = [
            str(path.relative_to(ROOT))
            for path in APP.rglob("*.py")
            # The example query in the "write your own" box is user-facing text,
            # not executed by the UI, so it is allowed in the one file that
            # renders it.
            if pattern.search(path.read_text(encoding="utf-8"))
            and path.name != "streamlit_app.py"
        ]
        assert not offenders, f"Cypher in the UI layer: {offenders}"


class TestRoutingIsThin:
    ROUTE_FILES = sorted(API.glob("routes_*.py"))

    def test_route_files_exist(self):
        assert self.ROUTE_FILES, "no route modules found"

    @pytest.mark.parametrize("path", ROUTE_FILES, ids=lambda p: p.name)
    def test_routes_do_not_import_capabilities_directly(self, path):
        """Routes talk to services, not to the retriever, the graph client or
        the LLM. A route that constructs a Retriever is orchestration that has
        leaked into the transport layer."""
        banned = {
            "src.retrieval.strategies", "src.llm", "src.answer",
            "src.ingest.pipeline", "src.evaluate", "src.graph.client",
        }
        leaked = banned & _imports(path)
        assert not leaked, f"{path.name} imports capabilities directly: {leaked}"

    @pytest.mark.parametrize("path", ROUTE_FILES, ids=lambda p: p.name)
    def test_route_handlers_stay_short(self, path):
        """A long handler is orchestration in the wrong layer.

        The threshold is generous on purpose - this catches a handler that grew
        a pipeline inside it, not one with a few mapping lines.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"))
        long_handlers = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(isinstance(d, ast.Call) and getattr(
                getattr(d.func, "value", None), "id", "") == "router"
                for d in node.decorator_list)
            and (node.end_lineno or 0) - node.lineno > 25
        ]
        assert not long_handlers, (
            f"{path.name} has handlers over 25 lines: {long_handlers}. "
            "Move the orchestration into src/services/."
        )


class TestServicesAreComposable:
    def test_every_service_is_exported(self):
        from src import services
        for name in ("QAService", "GraphService", "JobService", "SecurityService"):
            assert hasattr(services, name), f"{name} is not exported"

    def test_cli_and_api_share_the_service_layer(self):
        """`run.py ask` must not bypass the guardrails the API enforces."""
        source = (ROOT / "run.py").read_text(encoding="utf-8")
        assert "services.qa.ask" in source, (
            "the CLI does not go through QAService, so it skips the guardrails, "
            "the budget cap and the audit log"
        )

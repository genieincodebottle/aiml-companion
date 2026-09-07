"""Architecture tests: the layer boundaries are enforced, not merely described.

    app/            UI. Renders. Decides nothing.
    api/routes_*    Transport. Validates, calls ONE service, maps the result.
    src/services/   Orchestration and policy.
    src/            Capabilities.

A layering rule that lives only in a README is a rule that decays. These fail
the build when it does.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
API = ROOT / "api"
APP = ROOT / "app"

VENDOR_SDKS = {"google", "openai", "anthropic"}


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
    def test_src_never_imports_a_web_framework(self):
        """The rule that makes the boundary real.

        A service that could raise HTTPException would be callable only from a
        web request, and the CLI, the notebook and the tests would each need
        their own copy of the same orchestration - which is how a guardrail
        ends up enforced on one path and not the others.
        """
        offenders = [
            str(p.relative_to(ROOT))
            for p in SRC.rglob("*.py")
            if any(
                n.split(".")[0] in {"fastapi", "starlette", "uvicorn"}
                for n in _imports(p)
            )
        ]
        assert not offenders, f"business logic importing a web framework: {offenders}"

    def test_src_never_imports_the_ui_framework(self):
        offenders = [
            str(p.relative_to(ROOT))
            for p in SRC.rglob("*.py")
            if any(n.split(".")[0] == "streamlit" for n in _imports(p))
        ]
        assert not offenders, f"business logic importing the UI framework: {offenders}"


class TestVendorSDKsAreConfinedToAdapters:
    def test_only_provider_adapters_import_a_vendor_sdk(self):
        """The seam that makes the self-preference experiment possible.

        If any other module imported a vendor client, swapping the judge's
        provider would stop being a config change - and the whole argument for
        having four independently-configured roles would quietly stop holding.
        """
        adapters = SRC / "providers"
        offenders = [
            str(p.relative_to(ROOT))
            for p in SRC.rglob("*.py")
            if p.parent != adapters
            and any(n.split(".")[0] in VENDOR_SDKS for n in _imports(p))
        ]
        assert not offenders, f"vendor SDK imported outside an adapter: {offenders}"


class TestUIHoldsNoBusinessLogic:
    def test_ui_never_imports_src_or_api(self):
        """The UI reaches the system only over HTTP.

        Importing `src` from the UI would let it bypass every guardrail the API
        enforces, and a control a frontend can skip is a control anyone can skip
        with curl.
        """
        offenders = [
            str(p.relative_to(ROOT))
            for p in APP.rglob("*.py")
            if any(n.split(".")[0] in {"src", "api"} for n in _imports(p))
        ]
        assert not offenders, f"UI importing business logic directly: {offenders}"

    def test_ui_never_imports_a_vendor_sdk(self):
        offenders = [
            str(p.relative_to(ROOT))
            for p in APP.rglob("*.py")
            if any(n.split(".")[0] in VENDOR_SDKS for n in _imports(p))
        ]
        assert not offenders, f"UI talking to a model vendor directly: {offenders}"


class TestRoutingIsThin:
    ROUTE_FILES = sorted(API.glob("routes_*.py"))

    def test_route_files_exist(self):
        assert self.ROUTE_FILES, "no route modules found"

    @pytest.mark.parametrize("path", ROUTE_FILES, ids=lambda p: p.name)
    def test_routes_do_not_import_capabilities_directly(self, path):
        """Routes call services, not the judge, the RART loop or a provider.
        A route that builds a ServingLoop is orchestration leaking into
        transport."""
        banned = {
            "src.judge", "src.rart", "src.serving", "src.monitoring",
            "src.evaluate", "src.generator", "src.providers", "src.meta_judge",
        }
        leaked = banned & _imports(path)
        assert not leaked, f"{path.name} imports capabilities directly: {leaked}"

    @pytest.mark.parametrize("path", ROUTE_FILES, ids=lambda p: p.name)
    def test_route_handlers_stay_short(self, path):
        """A long handler is orchestration in the wrong layer. The threshold is
        generous: this catches a pipeline grown inside a handler, not a few
        lines of mapping."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        long_handlers = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(d, ast.Call)
                and getattr(getattr(d.func, "value", None), "id", "") == "router"
                for d in node.decorator_list
            )
            and (node.end_lineno or 0) - node.lineno > 25
        ]
        assert not long_handlers, (
            f"{path.name} has handlers over 25 lines: {long_handlers}. "
            "Move the orchestration into src/services/."
        )


class TestServicesAreComposable:
    def test_every_service_is_exported(self):
        from src import services

        for name in (
            "BenchmarkService",
            "TuningService",
            "ServingService",
            "MonitoringService",
        ):
            assert hasattr(services, name), f"{name} is not exported"

    def test_cli_goes_through_the_service_layer(self):
        """`run.py` must not reach past the services, or the CLI and the API
        would enforce different budget caps and write different provenance."""
        source = (ROOT / "run.py").read_text(encoding="utf-8")
        assert "from src.services import" in source
        for banned in ("from src.serving import", "from src.rart import"):
            assert banned not in source, f"run.py bypasses the service layer: {banned}"

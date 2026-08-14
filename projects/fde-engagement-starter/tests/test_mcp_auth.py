"""RUBRIC: read and write live on separate scopes, and the separation holds.

These fail until you implement src/mcp_server/auth.py.

The property under test is not "there is an auth check". It is that a read-scoped
caller physically cannot execute a write tool body. That distinction is what a
security reviewer is actually asking about, and it is why the test uses a side
effect rather than checking a return value.
"""
from __future__ import annotations

import pytest

from src.mcp_server.auth import (
    PermissionDenied,
    Principal,
    Scope,
    require,
    resolve_principal,
)


def test_read_token_resolves_without_write_scope():
    principal = resolve_principal("dev-read-token")
    assert isinstance(principal, Principal)
    assert principal.can(Scope.READ)
    assert not principal.can(Scope.WRITE), (
        "the read principal must not hold write scope. This asymmetry is the "
        "entire control."
    )


def test_write_token_resolves_with_write_scope():
    principal = resolve_principal("dev-write-token")
    assert principal.can(Scope.WRITE)


def test_unknown_token_is_denied_not_defaulted():
    """Fail closed.

    Defaulting an unrecognised token to read scope feels harmless and is how
    anonymous callers end up with data access.
    """
    with pytest.raises(PermissionDenied):
        resolve_principal("not-a-real-token")


def test_missing_token_is_denied():
    with pytest.raises(PermissionDenied):
        resolve_principal(None)


def test_write_tool_body_never_executes_for_a_read_principal():
    """The load-bearing test in this file.

    A tool that raises after doing the work is not access control. The side effect
    below proves the body did not run at all.
    """
    executed: list[str] = []

    @require(Scope.WRITE)
    def reschedule(principal: Principal, shipment_id: str) -> str:
        executed.append(shipment_id)
        return "rescheduled"

    reader = resolve_principal("dev-read-token")
    with pytest.raises(PermissionDenied):
        reschedule(reader, "SHP-1")

    assert executed == [], "the tool body ran despite the missing scope"


def test_write_tool_works_for_a_write_principal():
    executed: list[str] = []

    @require(Scope.WRITE)
    def reschedule(principal: Principal, shipment_id: str) -> str:
        executed.append(shipment_id)
        return "rescheduled"

    writer = resolve_principal("dev-write-token")
    assert reschedule(writer, "SHP-1") == "rescheduled"
    assert executed == ["SHP-1"]


def test_decorator_preserves_tool_identity():
    """MCP clients introspect tool names and docstrings.

    A decorator that eats them turns your tool list into a wall of `wrapper`.
    """

    @require(Scope.READ)
    def list_shipments(principal: Principal) -> list[str]:
        """List shipments."""
        return []

    assert list_shipments.__name__ == "list_shipments"
    assert (list_shipments.__doc__ or "").strip() == "List shipments."

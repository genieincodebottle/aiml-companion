"""Scoped authorisation for the MCP server.

YOU IMPLEMENT THIS. See tests/test_mcp_auth.py for the contract.

Why this file exists as its own module rather than an `if` inside each tool:

The question a customer's security reviewer asks is not "does your agent work",
it is "what can it do if it is wrong". An agent that reads shipment data and an
agent that reschedules a customer's delivery are the same code with different
credentials, and the difference between them is the only thing standing between a
retrieval bug and a hundred rescheduled trucks.

So: read and write are separate scopes, the read path physically cannot perform a
write, and that property is enforced in one place you can point at during a
security review.

The contract the tests enforce:

1. A token maps to a set of scopes.
2. Every tool declares the scope it requires.
3. Invoking a tool with a token lacking that scope raises `PermissionDenied` and
   the tool body never executes. Not "returns an error" - never executes.
4. An unknown or missing token is denied, not defaulted to read.
5. Denials are auditable (see audit.py). A refusal nobody recorded did not happen.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Scope(str, Enum):
    """The two scopes this engagement needs.

    Resist adding more until a tool genuinely needs one. Scope sprawl is how
    permission models stop being reviewable.
    """

    READ = "shipments:read"
    WRITE = "shipments:write"


class PermissionDenied(Exception):
    """Raised when a token lacks the scope a tool requires.

    Carry enough detail to audit: which principal, which tool, which scope was
    missing. Do not include the token itself.
    """


@dataclass(frozen=True)
class Principal:
    """Who is calling.

    In production this comes from a verified JWT (see deploy/jwt_middleware.py).
    Here it comes from a token lookup, which is the same shape with less ceremony.
    """

    name: str
    scopes: frozenset[Scope]

    def can(self, scope: Scope) -> bool:
        return scope in self.scopes


def resolve_principal(token: str | None) -> Principal:
    """Map a bearer token to a principal.

    Args:
        token: the raw token, or None.

    Returns:
        A Principal with the scopes that token carries.

    Raises:
        PermissionDenied: if the token is missing or unknown.

    Implementer notes:
        - Read `settings.mcp_read_token` and `settings.mcp_write_token`.
        - The write principal may hold both scopes. The read principal must NOT
          hold write. That asymmetry is the entire point.
        - Do not fall back to read scope on an unknown token. Fail closed.
        - Compare tokens with `hmac.compare_digest`, not `==`. It costs one import
          and it is the kind of detail a security reviewer notices.
    """
    raise NotImplementedError(
        "Implement resolve_principal. Run `pytest tests/test_mcp_auth.py -v`."
    )


def require(scope: Scope):
    """Decorator enforcing a scope on a tool function.

    Usage:
        @require(Scope.WRITE)
        def reschedule_shipment(principal: Principal, shipment_id: str) -> dict:
            ...

    The decorated function must receive the Principal as its first argument, so
    that authorisation is impossible to forget: a tool with no principal will not
    even have the right signature.

    Implementer notes:
        - Raise PermissionDenied BEFORE calling the wrapped function. The test
          asserts the body never ran, using a side effect.
        - Emit an audit record for the denial as well as for the success. Denials
          are the interesting half of an audit log.
        - Preserve the wrapped function's name and docstring (functools.wraps).
    """
    raise NotImplementedError(
        "Implement the require decorator. See tests/test_mcp_auth.py."
    )

"""API authentication and caller identity.

WHAT THIS IS
============
An optional shared-secret header, plus a stable caller identity used for rate
limiting and the audit log.

WHAT THIS IS NOT
================
An authentication system. There are no users, no sessions, no roles, and no
per-tenant isolation, and pretending otherwise would be worse than the honest
gap. A shared secret answers "is this caller allowed to use the API at all" and
nothing else - it cannot answer "which supplier data may this caller see",
which is the question a real deployment of this system has to answer.

The reason it is here at all: without it, a `run.py api` bound to anything but
localhost is an unauthenticated endpoint that spends money on a third-party
model API on behalf of whoever finds it. That is the actual risk in a project
like this, and a shared secret genuinely closes it.

WHAT PRODUCTION ADDS
====================
See docs/security.md. In short: OIDC or mTLS at the edge, per-user identity
carried through to the audit log, Neo4j RBAC with a read-only role for query
endpoints, and row-level filtering of retrieval results by the caller's
entitlements - which is the hard one, because a graph traversal naturally
crosses authorisation boundaries and the filtering has to happen inside the
Cypher rather than after it.
"""

from __future__ import annotations

import hashlib
import hmac
import os

from fastapi import Header, HTTPException, Request, status

API_KEY_HEADER = "X-API-Key"


def _expected_key() -> str | None:
    key = (os.getenv("API_KEY") or "").strip()
    return key or None


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> str:
    """Verify the shared secret if one is configured, and return a caller id.

    When `API_KEY` is unset the API is open, which is the right default for a
    learning project run on localhost and the wrong default for anything else.
    The startup banner says so explicitly rather than leaving it to be
    discovered.
    """
    expected = _expected_key()
    if expected is None:
        return _caller_id(request, None)

    # Constant-time comparison. A plain `==` on a secret leaks its length and,
    # in principle, its content through timing. The cost of doing it correctly
    # is one function call.
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid {API_KEY_HEADER} header.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
    return _caller_id(request, x_api_key)


def _caller_id(request: Request, api_key: str | None) -> str:
    """A stable, non-identifying handle for rate limiting and the audit log.

    The key is hashed rather than stored, because the audit log must never
    accumulate the credentials it was written to protect - a mistake that has
    caused real incidents. The client host is used when there is no key, which
    is weak (it is trivially spoofable behind a proxy) and is the correct
    strength for a localhost learning deployment.
    """
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:12]
    host = request.client.host if request.client else "unknown"
    return f"host:{host}"

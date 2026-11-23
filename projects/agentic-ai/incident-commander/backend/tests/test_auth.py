"""Auth: password hashing, tokens, and the fail-closed dependency behaviour.

No database is involved. Everything here is either pure (hashing, signing) or
exercises the "auth is off" path, which is the default configuration and the one
every other test in the suite depends on staying permissive.

The database-backed half (login, seeding, role enforcement) needs a live Postgres
and is covered by the manual flow in DEMO.md rather than by a test that would make
`pytest` require Docker. That trade is stated here rather than left implicit: it
means a regression in `authenticate()` would not be caught by CI.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.auth import (
    MIN_SECRET_BYTES,
    PLACEHOLDER_SECRET,
    ROLES,
    AuthConfigError,
    create_token,
    decode_token,
    hash_password,
    require_role,
    require_user,
    verify_password,
)
from src.config import get_settings

#: Both are >= MIN_SECRET_BYTES; a shorter one now raises rather than warning.
_GOOD_SECRET = "test-secret-abcdefghijklmnopqrstuvwxyz-0123456789"
_OTHER_SECRET = "other-secret-abcdefghijklmnopqrstuvwxyz-0123456789"


class _FakeUser:
    """Minimal stand-in for UserRow: create_token only reads these four fields."""

    def __init__(self, id=1, username="tester", role="operator"):
        self.id = id
        self.username = username
        self.role = role


# --- passwords --------------------------------------------------------------


def test_password_hash_is_salted_and_verifiable():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert hashed.startswith("$2b$")
    assert verify_password("correct horse battery staple", hashed)
    assert not verify_password("wrong", hashed)


def test_same_password_hashes_differently_each_time():
    """A per-password salt: two users with the same password must not collide.

    Without this, an attacker who cracks one hash cracks every account that shares
    that password, and equal hashes leak that two users chose the same one.
    """
    a, b = hash_password("same-password"), hash_password("same-password")
    assert a != b
    assert verify_password("same-password", a)
    assert verify_password("same-password", b)


def test_verify_password_survives_a_garbage_hash():
    """A corrupt row must fail the login, not 500 the endpoint."""
    for junk in ("", "not-a-hash", "$2b$broken", None):
        assert verify_password("anything", junk) is False


# --- tokens -----------------------------------------------------------------


def test_token_roundtrips_the_identity(env):
    env(IC_JWT_SECRET=_GOOD_SECRET)
    token, expires_in = create_token(_FakeUser(id=7, username="ada", role="admin"))
    claims = decode_token(token)
    assert claims["sub"] == "7"
    assert claims["username"] == "ada"
    assert claims["role"] == "admin"
    assert expires_in > 0


def test_token_signed_with_another_secret_is_rejected(env):
    """The signature is the whole security property, so prove it is checked."""
    env(IC_JWT_SECRET=_GOOD_SECRET)
    token, _ = create_token(_FakeUser())
    env(IC_JWT_SECRET=_OTHER_SECRET)
    assert decode_token(token) is None


def test_tampered_token_is_rejected(env):
    env(IC_JWT_SECRET=_GOOD_SECRET)
    token, _ = create_token(_FakeUser(role="viewer"))
    head, payload, sig = token.split(".")
    assert decode_token(f"{head}.{payload}.{sig[:-4]}xxxx") is None


def test_expired_token_is_rejected(env):
    """TTL of zero minutes: issued and already expired."""
    env(IC_JWT_SECRET=_GOOD_SECRET, IC_JWT_TTL_MINUTES="0")
    token, _ = create_token(_FakeUser())
    assert decode_token(token) is None


def test_placeholder_secret_is_refused(env):
    """A shipped default signing secret is the same as having none at all.

    Anyone holding the public repo could mint a valid admin token, so signing must
    fail loudly rather than quietly work with the documented placeholder.
    """
    env(IC_JWT_SECRET=PLACEHOLDER_SECRET)
    with pytest.raises(AuthConfigError):
        create_token(_FakeUser())
    # Decoding degrades to "invalid" rather than raising, so a request 401s.
    assert decode_token("anything.at.all") is None


def test_short_secret_is_refused(env):
    """HS256 with a key shorter than the hash output is weaker than it appears.

    Caught because PyJWT warned about it; an easily-guessed 12-character secret is
    not much better than the placeholder, so it fails the same way.
    """
    env(IC_JWT_SECRET="short-key-12")
    assert len("short-key-12".encode("utf-8")) < MIN_SECRET_BYTES
    with pytest.raises(AuthConfigError):
        create_token(_FakeUser())


# --- configuration seam -----------------------------------------------------


def test_auth_stays_off_without_a_database(env):
    """`IC_AUTH_REQUIRED` alone must not enable auth.

    Auth needs somewhere to keep users. Turning it on with no user store would be a
    locked door with no key: every request 401s and nobody can ever log in.
    """
    env(IC_AUTH_REQUIRED="true", IC_DATABASE_URL="")
    assert get_settings().auth_required is True
    assert get_settings().auth_enabled is False


def test_auth_enables_only_with_both(env):
    env(IC_AUTH_REQUIRED="true", IC_DATABASE_URL="postgresql://u:p@localhost/db")
    assert get_settings().auth_enabled is True


# --- dependencies -----------------------------------------------------------


@pytest.mark.asyncio
async def test_require_user_allows_anonymous_when_auth_is_off(env):
    """The default configuration must stay open, or the console breaks for everyone."""
    env(IC_AUTH_REQUIRED="false", IC_DATABASE_URL="")
    user = await require_user(user=None)
    assert user["username"] == "anonymous"


@pytest.mark.asyncio
async def test_require_user_rejects_anonymous_when_auth_is_on(env, monkeypatch):
    """Anonymous is refused whenever auth is on - but the two ways it can be on
    deserve different answers.

    With a reachable user store, "authentication required" (401) is the truth. With
    the store unreachable, nobody CAN authenticate and `/api/auth/login` cannot
    help either, so a 401 sends the operator hunting for a bad password when the
    real problem is that Postgres is down. Both are refusals; only one of them is
    diagnosable.
    """
    env(IC_AUTH_REQUIRED="true", IC_DATABASE_URL="postgresql://u:p@localhost/db")

    monkeypatch.setattr("src.auth.db_ready", lambda: True)
    with pytest.raises(HTTPException) as exc:
        await require_user(user=None)
    assert exc.value.status_code == 401

    monkeypatch.setattr("src.auth.db_ready", lambda: False)
    with pytest.raises(HTTPException) as exc:
        await require_user(user=None)
    assert exc.value.status_code == 503
    assert "database is unreachable" in exc.value.detail


@pytest.mark.asyncio
async def test_role_check_is_fail_closed(env):
    """An unrecognized role matches no allowlist and is therefore denied.

    Same rule as the agent's own RBAC: unknown means no permissions, never a
    permissive default.
    """
    env(IC_AUTH_REQUIRED="true", IC_DATABASE_URL="postgresql://u:p@localhost/db")
    check = require_role("operator", "admin")

    assert (await check(user={"username": "a", "role": "admin"}))["role"] == "admin"

    for role in ("viewer", "captain", "", None):
        with pytest.raises(HTTPException) as exc:
            await check(user={"username": "a", "role": role})
        assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_role_check_is_a_noop_when_auth_is_off(env):
    env(IC_AUTH_REQUIRED="false", IC_DATABASE_URL="")
    check = require_role("admin")
    assert await check(user={"username": "anonymous", "role": "viewer"}) is not None


def test_roles_are_ordered_least_to_most_privileged():
    """Guards against someone inserting a role in the middle and shifting meaning."""
    assert ROLES == ("viewer", "operator", "admin")

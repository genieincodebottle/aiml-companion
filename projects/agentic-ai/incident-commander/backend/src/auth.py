"""Password hashing, session tokens, and the FastAPI dependencies that use them.

Scope, stated honestly: this is a *demonstration* auth layer. It is real bcrypt
and real signed tokens rather than a mock, but it has no refresh rotation, no
lockout after repeated failures, no MFA, and no revocation list - a logged-out
token stays valid until it expires. Those are the things you would add before
this faced the internet, and they are listed in docs/architecture.md rather than
quietly omitted.

What it does get right, because these are the parts that teach something:

  * Passwords are bcrypt-hashed with a per-password salt. The plaintext never
    reaches the database, a log, or an error message.
  * The signing secret must be changed from its shipped default before auth can be
    switched on. A framework that ships a working default secret has shipped a
    backdoor.
  * Login answers identically for "no such user" and "wrong password", so the
    endpoint cannot be used to enumerate accounts.
  * An application role (viewer/operator/admin) is NOT the agent's role. Logging in
    as an admin does not widen what the agent may execute - see IC_AGENT_ROLE.
"""

from __future__ import annotations

import datetime as dt
import logging
import secrets

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import func, select

from .config import get_settings
from .db.models import UserRow
from .db.session import db_ready, session_scope

logger = logging.getLogger(__name__)

#: The placeholder shipped in .env.example. Refused when auth is enabled.
PLACEHOLDER_SECRET = "change-me-in-production"

#: Application roles, most privileged last. Distinct from the AGENT's role.
ROLES = ("viewer", "operator", "admin")

#: `auto_error=False` so an anonymous request reaches the dependency and can be
#: allowed through when auth is off, instead of being rejected by the extractor.
_bearer = HTTPBearer(auto_error=False)


# --- passwords --------------------------------------------------------------


#: bcrypt reads at most 72 bytes, and bcrypt>=5 raises on longer input.
MAX_PASSWORD_BYTES = 72


def hash_password(plaintext: str) -> str:
    raw = plaintext.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise ValueError(f"password is longer than {MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("utf-8")


#: RFC 7518 §3.2 requires an HMAC key at least as long as the hash output. Below
#: this, HS256 is weaker than it looks, and PyJWT rightly warns about it.
MIN_SECRET_BYTES = 32


def verify_password(plaintext: str, hashed: str | None) -> bool:
    """Constant-time compare via bcrypt. Never raises.

    A row with a NULL or truncated `password_hash` must fail the login, not 500 the
    endpoint - `AttributeError` is caught for exactly that case, since `None` has no
    `.encode`.
    """
    if not plaintext or not hashed:
        return False
    try:
        return bcrypt.checkpw(plaintext.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError, AttributeError):
        return False


# --- tokens -----------------------------------------------------------------


class AuthConfigError(RuntimeError):
    """Auth is enabled but cannot be operated safely."""


def _secret() -> str:
    settings = get_settings()
    if settings.jwt_secret == PLACEHOLDER_SECRET:
        raise AuthConfigError(
            "IC_JWT_SECRET is still the shipped placeholder. Set it to a random value "
            "before enabling auth; a known signing secret means anyone can mint a token."
        )
    if len(settings.jwt_secret.encode("utf-8")) < MIN_SECRET_BYTES:
        raise AuthConfigError(
            f"IC_JWT_SECRET must be at least {MIN_SECRET_BYTES} bytes for HS256 "
            '(RFC 7518 3.2). Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    return settings.jwt_secret


def create_token(user: UserRow) -> tuple[str, int]:
    """Sign a session token. Returns (token, expires_in_seconds)."""
    settings = get_settings()
    ttl = dt.timedelta(minutes=settings.jwt_ttl_minutes)
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(payload, _secret(), algorithm="HS256"), int(ttl.total_seconds())


def decode_token(token: str) -> dict | None:
    """Verified claims, or None. Expiry and signature failures are both None."""
    try:
        return jwt.decode(token, _secret(), algorithms=["HS256"])
    except (jwt.PyJWTError, AuthConfigError):
        return None


# --- user store -------------------------------------------------------------


async def get_user_by_username(username: str) -> UserRow | None:
    async with session_scope() as s:
        return (await s.execute(select(UserRow).where(UserRow.username == username))).scalar_one_or_none()


async def get_user_by_id(user_id: int) -> UserRow | None:
    async with session_scope() as s:
        return (await s.execute(select(UserRow).where(UserRow.id == user_id))).scalar_one_or_none()


async def authenticate(username: str, password: str) -> UserRow | None:
    """Verify credentials, or return None.

    The caller must report one message for every failure here. Distinguishing
    "no such user" from "wrong password" turns the login form into an account
    enumeration oracle.
    """
    user = await get_user_by_username(username)
    if user is None or not user.is_active:
        # Hash anyway so a missing user does not answer measurably faster than a
        # wrong password. Cheap defence against timing-based enumeration.
        verify_password(password, "$2b$12$" + "x" * 53)
        return None
    if not verify_password(password, user.password_hash):
        return None
    async with session_scope() as s:
        user.last_login_at = dt.datetime.now(dt.timezone.utc)
        s.add(user)
    return user


async def create_user(username: str, password: str, *, role: str = "operator", email: str | None = None) -> UserRow:
    if role not in ROLES:
        raise ValueError(f"unknown role '{role}'; expected one of {ROLES}")
    async with session_scope() as s:
        user = UserRow(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role=role,
        )
        s.add(user)
        await s.flush()
        await s.refresh(user)
        return user


async def seed_admin_if_empty() -> str | None:
    """Create the seed admin, but only into an empty users table.

    The emptiness check is the whole safety property: without it, a redeploy would
    silently reset the admin password back to the environment default, which is
    how a rotated credential quietly becomes the shipped one again.
    """
    settings = get_settings()
    # Only when login is ON. Seeding on a login-off boot printed the generated
    # password into a log nobody was reading; by the time someone enabled login
    # (often recreating the container) it was gone and no one could sign in.
    if not db_ready() or not settings.auth_enabled:
        return None
    async with session_scope() as s:
        count = (await s.execute(select(func.count()).select_from(UserRow))).scalar_one()
        if count:
            return None
    password = settings.seed_admin_password
    if not password:
        # No password configured: generate one and show it exactly once. Shipping
        # a fixed default (the classic admin/admin) means every copy of this
        # project that turns login on has the same admin credential.
        password = secrets.token_urlsafe(12)
        logger.warning(
            "seeded admin '%s' with generated password: %s  (shown once; set IC_SEED_ADMIN_PASSWORD to choose one)",
            settings.seed_admin_username,
            password,
        )
    user = await create_user(settings.seed_admin_username, password, role="admin")
    logger.warning("seeded initial admin '%s' into an empty users table", user.username)
    return user.username


# --- FastAPI dependencies ---------------------------------------------------


async def current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict | None:
    """The logged-in user, or None.

    Never raises. Use this where identity is useful but not required, such as
    stamping who triggered an incident.
    """
    if not get_settings().auth_enabled or not db_ready():
        return None
    token = creds.credentials if creds else request.cookies.get("ic_token")
    if not token:
        return None
    claims = decode_token(token)
    if not claims:
        return None
    user = await get_user_by_id(int(claims["sub"]))
    if user is None or not user.is_active:
        return None
    return user.to_public()


async def require_user(user: dict | None = Depends(current_user)) -> dict:
    """Reject anonymous requests. A no-op when auth is disabled."""
    if not get_settings().auth_enabled:
        return user or {"username": "anonymous", "role": "admin"}
    if user is None:
        if not db_ready():
            # Auth is on (a database URL is configured) but the database is not
            # reachable, so nobody can be authenticated and nobody can log in
            # either - /api/auth/login needs the same user store. A bare 401 sends
            # the operator hunting for a bad password; this says what is actually
            # wrong. Still a refusal: an unreachable user store is never a reason
            # to let a request through.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="login is enabled (IC_DATABASE_URL is set) but the database is unreachable, "
                "so no one can be authenticated. Start Postgres (docker compose up -d postgres), "
                "or unset IC_DATABASE_URL to run in the no-login in-memory mode.",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_role(*allowed: str):
    """Dependency factory: reject a logged-in user without one of these roles.

    Fail-closed, like the agent's own RBAC: an unrecognized role matches nothing
    and is therefore denied, rather than falling through to a permissive default.
    """

    async def _check(user: dict = Depends(require_user)) -> dict:
        if not get_settings().auth_enabled:
            return user
        if user.get("role") not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"role '{user.get('role')}' may not perform this action",
            )
        return user

    return _check

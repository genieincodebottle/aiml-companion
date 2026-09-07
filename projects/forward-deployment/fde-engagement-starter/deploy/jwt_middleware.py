"""Okta-style JWT verification for the in-VPC deployment.

WORKING CODE. This is not a skeleton. Auth is the one part of an FDE engagement
where a plausible-looking placeholder is worse than nothing, because it passes
review by looking finished.

What this does
--------------
  - fetches the identity provider's JWKS and caches it, with a bounded refresh
    on unknown key ids so a provider key rotation does not take you down
  - verifies signature, `iss`, `aud`, `exp`, and `nbf`
  - enforces required scopes per route, and fails closed

Why scope enforcement lives HERE and not in the tool
----------------------------------------------------
Put the check in the tool and you have as many implementations of the security
model as you have tools. Every new tool is a new chance to forget, and the one
that forgets is the one that writes to the dispatch system. Worse, a tool that
checks its own permissions cannot be audited without reading the tool: there is
no single place a reviewer can look to answer "what can a read-only token do?"

At the middleware, the check happens once, before any handler runs, in code the
customer's security team can read in one sitting. The tool then does one job,
which is the job it is named after. When Northwind asks whether a read-only
token can trigger a write, the answer is a route table, not an audit of every
tool body.

Corollary, and it is the part people skip: the middleware must fail CLOSED. An
unknown route, a missing scope claim, an unparseable token, or a JWKS fetch that
did not succeed all mean deny. A middleware that lets a request through when it
cannot decide is not a control, it is a log line.

Offline testability
-------------------
Nothing here requires the network at import time, and `JWTVerifier` accepts an
injected key so the whole class is unit-testable with no JWKS server:

    verifier = JWTVerifier(
        issuer="https://northwind.okta.com/oauth2/default",
        audience="api://northwind-dispatch-ai",
        public_key=my_test_public_key,      # skips JWKS entirely
    )
    claims = verifier.verify(token, required_scopes={"dispatch:read"})

Dependencies: stdlib + pyjwt.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

try:
    import jwt
    from jwt import PyJWKClient
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyjwt is required for deploy.jwt_middleware. "
        "Install it with: pip install 'pyjwt>=2.9.0'"
    ) from exc


DEFAULT_ALGORITHMS = ("RS256",)
DEFAULT_LEEWAY_SECONDS = 30  # tolerate modest clock skew, not an expired token
DEFAULT_JWKS_TTL_SECONDS = 3600
MIN_JWKS_REFRESH_INTERVAL = 60  # do not let a bad token DoS the IdP


class AuthError(Exception):
    """Base class. Carries an HTTP status so callers do not invent one."""

    status_code = 401

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class TokenInvalid(AuthError):
    """Signature, issuer, audience, or expiry check failed. 401."""

    status_code = 401


class ScopeDenied(AuthError):
    """Token is valid but lacks the scope this route requires. 403.

    The 401/403 split matters operationally: 401 means re-authenticate, 403
    means stop asking. Collapsing them into one status sends the customer's
    integration into a retry loop against a decision that will never change.
    """

    status_code = 403


# ---------------------------------------------------------------------------
# JWKS with cache
# ---------------------------------------------------------------------------


@dataclass
class JWKSCache:
    """Fetches and caches signing keys from the IdP's JWKS endpoint.

    Cache invalidation rule: a token whose `kid` is not in the cache triggers at
    most one refresh per MIN_JWKS_REFRESH_INTERVAL. Without that floor, a stream
    of tokens carrying a garbage `kid` becomes an amplified request flood
    against the customer's identity provider, which is a very bad first week.
    """

    jwks_uri: str
    ttl_seconds: int = DEFAULT_JWKS_TTL_SECONDS
    timeout_seconds: float = 5.0
    _keys: dict[str, Any] = field(default_factory=dict, repr=False)
    _fetched_at: float = 0.0
    _last_refresh_attempt: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_key(self, kid: str):
        with self._lock:
            fresh = (time.time() - self._fetched_at) < self.ttl_seconds
            if kid in self._keys and fresh:
                return self._keys[kid]
            if not fresh or kid not in self._keys:
                if (time.time() - self._last_refresh_attempt) >= MIN_JWKS_REFRESH_INTERVAL:
                    self._refresh_locked()
            key = self._keys.get(kid)
        if key is None:
            # Fail closed. An unknown kid after a refresh is either a rotation
            # we cannot see or a forged header, and both mean deny.
            raise TokenInvalid("token signed by an unknown key id")
        return key

    def _refresh_locked(self) -> None:
        self._last_refresh_attempt = time.time()
        try:
            req = urllib.request.Request(self.jwks_uri, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            # Keep any keys we already hold. A transient JWKS outage should not
            # log out every active session, but it must never mint trust either.
            if not self._keys:
                raise TokenInvalid(f"cannot reach JWKS endpoint: {exc}") from exc
            return
        keys = {}
        for entry in payload.get("keys", []):
            kid = entry.get("kid")
            if not kid:
                continue
            try:
                keys[kid] = jwt.PyJWK(entry).key
            except Exception:
                continue  # skip keys of algorithms we do not accept
        if keys:
            self._keys = keys
            self._fetched_at = time.time()


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class JWTVerifier:
    """Verifies Okta-style access tokens and enforces scopes.

    Args:
        issuer:     exact expected `iss`. Not a prefix match. A prefix match on
                    an issuer URL is how a neighbouring tenant becomes a valid
                    signer for your API.
        audience:   exact expected `aud`. Okta will happily issue a valid token
                    for a different audience in the same org, and it will verify
                    perfectly against the same JWKS. `aud` is what separates
                    "a real token" from "a token for this API".
        jwks_uri:   JWKS endpoint. Ignored when `public_key` is supplied.
        public_key: inject a key to bypass JWKS entirely. Used by tests, and by
                    air-gapped deployments where the key is provisioned by
                    configuration rather than fetched.
        algorithms: allowed signature algorithms. Never include "none", and
                    never widen this to include HS256 alongside RS256: the
                    classic forgery is an HS256 token signed with the public key
                    the verifier already trusts.
    """

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_uri: str | None = None,
        public_key: Any | None = None,
        algorithms: Iterable[str] = DEFAULT_ALGORITHMS,
        leeway_seconds: int = DEFAULT_LEEWAY_SECONDS,
        jwks_ttl_seconds: int = DEFAULT_JWKS_TTL_SECONDS,
    ) -> None:
        if not issuer or not audience:
            raise ValueError("issuer and audience are both required")
        if public_key is None and not jwks_uri:
            raise ValueError("supply either jwks_uri or public_key")
        self.issuer = issuer
        self.audience = audience
        self.algorithms = list(algorithms)
        if "none" in {a.lower() for a in self.algorithms}:
            raise ValueError("algorithm 'none' is never acceptable")
        self.leeway_seconds = leeway_seconds
        self._public_key = public_key
        self._jwks = JWKSCache(jwks_uri, ttl_seconds=jwks_ttl_seconds) if jwks_uri else None

    # -- key resolution ----------------------------------------------------

    def _key_for(self, token: str):
        if self._public_key is not None:
            return self._public_key
        try:
            header = jwt.get_unverified_header(token)
        except Exception as exc:
            raise TokenInvalid(f"malformed token header: {exc}") from exc
        kid = header.get("kid")
        if not kid:
            raise TokenInvalid("token header has no kid")
        return self._jwks.get_key(kid)

    # -- verification ------------------------------------------------------

    def verify(
        self,
        token: str,
        required_scopes: Iterable[str] | None = None,
        require_all: bool = True,
    ) -> dict:
        """Verify a token and enforce scopes. Returns the claims, or raises.

        Args:
            required_scopes: scopes this route needs. Empty or None means the
                route is authenticated but unscoped - be deliberate about that,
                since it is how a write endpoint quietly becomes reachable.
            require_all: True demands every listed scope (AND), False accepts
                any one of them (OR). Default AND, because the safe default for
                a permission check is the strict one.

        Raises:
            TokenInvalid: 401. Signature, iss, aud, exp, or nbf failed.
            ScopeDenied:  403. Authenticated, but not permitted.
        """
        if not token or not isinstance(token, str):
            raise TokenInvalid("no bearer token presented")

        key = self._key_for(token)
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway_seconds,
                options={
                    "require": ["exp", "iss", "aud"],
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
            )
        except jwt.ExpiredSignatureError as exc:
            raise TokenInvalid("token has expired") from exc
        except jwt.InvalidAudienceError as exc:
            raise TokenInvalid("token audience does not match this API") from exc
        except jwt.InvalidIssuerError as exc:
            raise TokenInvalid("token issuer is not trusted") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise TokenInvalid(f"token is missing a required claim: {exc}") from exc
        except jwt.InvalidTokenError as exc:
            raise TokenInvalid(f"token is not valid: {exc}") from exc

        if required_scopes:
            self._enforce_scopes(claims, required_scopes, require_all)
        return claims

    @staticmethod
    def extract_scopes(claims: Mapping[str, Any]) -> set[str]:
        """Read scopes from a claim set.

        Okta emits `scp` as a list. Many other providers emit `scope` as a
        space-delimited string. Accept both, because the customer may switch IdP
        during the engagement and you do not want that to be a code change.
        """
        raw = claims.get("scp", claims.get("scope", []))
        if isinstance(raw, str):
            return {s for s in raw.split() if s}
        if isinstance(raw, (list, tuple, set)):
            return {str(s) for s in raw if s}
        return set()

    def _enforce_scopes(
        self, claims: Mapping[str, Any], required: Iterable[str], require_all: bool
    ) -> None:
        held = self.extract_scopes(claims)
        needed = set(required)
        ok = needed <= held if require_all else bool(needed & held)
        if not ok:
            missing = sorted(needed - held)
            # Name the missing scope. The person debugging this at 2am is on the
            # customer's side and does not have your source. Do not echo the
            # scopes the token DOES hold: that hands an attacker a map.
            raise ScopeDenied(
                "token lacks required scope(s): " + ", ".join(missing)
            )


# ---------------------------------------------------------------------------
# Route policy and framework glue
# ---------------------------------------------------------------------------

# The whole security model, readable in one screen. This table is the artifact
# you walk the customer's security reviewer through.
#
# The read/write split is the point: `dispatch:read` cannot reach any route that
# mutates their system, and no tool can override that by being written badly.
ROUTE_SCOPES: dict[tuple[str, str], set[str]] = {
    ("GET", "/healthz"): set(),                      # unauthenticated liveness
    ("GET", "/tickets"): {"dispatch:read"},
    ("GET", "/tickets/{id}"): {"dispatch:read"},
    ("POST", "/triage"): {"dispatch:read"},          # analysis only, no mutation
    ("POST", "/dispatch/reroute"): {"dispatch:write"},
    ("POST", "/dispatch/hold"): {"dispatch:write"},
    ("GET", "/audit"): {"audit:read"},
}

# Routes that need no token at all. Explicit allowlist, never a prefix rule:
# "/health" as a prefix also matches "/healthcheck-admin", and someone will
# eventually add that route.
PUBLIC_ROUTES: set[tuple[str, str]] = {("GET", "/healthz")}


def scopes_for_route(method: str, path: str) -> set[str]:
    """Return required scopes, failing CLOSED on an unknown route.

    An unrecognised route raises rather than returning an empty set. Returning
    empty would mean "no scopes required", so a typo in the table, or a new
    endpoint nobody registered, would ship as a public endpoint.
    """
    key = (method.upper(), path)
    if key in ROUTE_SCOPES:
        return ROUTE_SCOPES[key]
    raise ScopeDenied(f"no scope policy registered for {method.upper()} {path}", 403)


def bearer_token(headers: Mapping[str, str]) -> str:
    """Pull the bearer token out of an Authorization header."""
    raw = headers.get("authorization") or headers.get("Authorization") or ""
    parts = raw.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise TokenInvalid("Authorization header must be 'Bearer <token>'")
    return parts[1].strip()


def authorize_request(
    verifier: JWTVerifier,
    method: str,
    path: str,
    headers: Mapping[str, str],
) -> dict:
    """One-call entry point: resolve policy, verify token, enforce scope.

    Framework-agnostic on purpose. Wrap it in FastAPI middleware, a decorator,
    or an ASGI app; the decision logic stays in one testable function.

    Returns the verified claims for the caller to log. Every allowed request
    should produce an audit record containing the subject, the route, and the
    scopes that permitted it. See src/mcp_server/audit.py.
    """
    key = (method.upper(), path)
    if key in PUBLIC_ROUTES:
        return {}
    required = scopes_for_route(method, path)
    claims = verifier.verify(bearer_token(headers), required_scopes=required)
    return claims


def build_verifier_from_env(getenv: Callable[[str, str], str] | None = None) -> JWTVerifier:
    """Construct a verifier from environment configuration.

    OIDC_ISSUER and OIDC_AUDIENCE are required and there is no default for
    either. A default issuer is a default trust anchor, and a misconfigured
    deployment must refuse to start rather than start with the wrong one.
    """
    import os

    get = getenv or (lambda k, d="": os.environ.get(k, d))
    issuer = get("OIDC_ISSUER", "")
    audience = get("OIDC_AUDIENCE", "")
    if not issuer or not audience:
        raise ValueError("OIDC_ISSUER and OIDC_AUDIENCE must both be set")
    jwks_uri = get("OIDC_JWKS_URI", "") or issuer.rstrip("/") + "/v1/keys"
    return JWTVerifier(issuer=issuer, audience=audience, jwks_uri=jwks_uri)


__all__ = [
    "AuthError",
    "TokenInvalid",
    "ScopeDenied",
    "JWKSCache",
    "JWTVerifier",
    "ROUTE_SCOPES",
    "PUBLIC_ROUTES",
    "scopes_for_route",
    "bearer_token",
    "authorize_request",
    "build_verifier_from_env",
]

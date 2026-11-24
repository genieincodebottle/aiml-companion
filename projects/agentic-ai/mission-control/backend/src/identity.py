"""Per-agent workload identity - short-lived, scoped credentials.

Zero-trust between agents: an agent does not carry ambient authority. To call a tool
it must present a token minted for its specific card, scoped to that card's allowed
tools, with a short TTL. The tool gateway verifies the token's scope before running.

This mirrors AgentCore Identity / SPIFFE-style workload identity: the credential is
the agent's *identity plus its permissions*, issued by the control plane, never
self-asserted. Here it is an HMAC-signed token so it is verifiable and tamper-evident
without any external service.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from .models.schemas import AgentCard

_DEFAULT_TTL_S = 300


@dataclass(frozen=True)
class AgentToken:
    agent: str
    scopes: tuple[str, ...]
    expires_at: float
    signature: str

    def to_header(self) -> str:
        body = json.dumps({"agent": self.agent, "scopes": list(self.scopes), "exp": self.expires_at})
        return f"{body}.{self.signature}"


class IdentityBroker:
    """Mints and verifies scoped, short-lived agent tokens.

    The signing key comes from settings (MC_SIGNING_KEY, or a random per-process key).
    In production it is a KMS-managed key.
    """

    def __init__(self, key: bytes) -> None:
        self._key = key

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()

    def mint(self, card: AgentCard, ttl_s: int = _DEFAULT_TTL_S, now: float | None = None) -> AgentToken:
        exp = (now or time.time()) + ttl_s
        payload = json.dumps({"agent": card.name, "scopes": card.allowed_tools, "exp": exp})
        return AgentToken(
            agent=card.name,
            scopes=tuple(card.allowed_tools),
            expires_at=exp,
            signature=self._sign(payload),
        )

    def verify(self, token: AgentToken, tool: str, now: float | None = None) -> tuple[bool, str]:
        """Return (ok, reason). Checks signature, expiry, and scope for `tool`."""
        now = now or time.time()
        payload = json.dumps({"agent": token.agent, "scopes": list(token.scopes), "exp": token.expires_at})
        if not hmac.compare_digest(self._sign(payload), token.signature):
            return False, "invalid signature (tampered token)"
        if now >= token.expires_at:
            return False, "token expired"
        if tool not in token.scopes:
            return False, f"tool '{tool}' not in token scope"
        return True, "ok"

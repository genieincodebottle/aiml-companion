"""The AI Gateway - every model call in the platform flows through here.

Responsibilities (in order along the request path):
  redaction -> routing -> semantic cache -> provider call (with fallback) ->
  usage/cost accounting -> audit log.

No agent talks to a model provider directly. That single chokepoint is what makes
budgets, caching, PII masking, model routing, and a complete audit trail *platform
guarantees* rather than per-agent good intentions.
"""

from .core import Gateway, GatewayError

__all__ = ["Gateway", "GatewayError"]

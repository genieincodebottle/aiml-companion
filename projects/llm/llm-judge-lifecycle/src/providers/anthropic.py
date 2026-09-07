"""Anthropic adapter.

Included as the third vendor mainly so the self-preference experiment has a
genuinely independent judge to run against. Two shape differences from the
others are worth knowing, because they are the kind of thing that silently
breaks a naive port.

1. ``system`` is a top-level parameter, not a message with role "system".
   Passing it as a message is accepted and quietly ignored, so the judge runs
   with no instructions at all and simply looks worse than it is.

2. There is no ``response_schema`` parameter. Structured output is obtained by
   defining a tool and forcing the model to call it. Asking for JSON in the
   prompt instead works most of the time, which is the problem: it fails on the
   awkward inputs, which are exactly the ones the benchmark is made of.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import RoleConfig
from . import Completion, TruncatedCompletion
from ._common import price_for, with_retry

_TOOL_NAME = "emit_response"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, role_config: RoleConfig) -> None:
        import anthropic  # lazy: only this adapter needs it

        self._cfg = role_config
        self.model = role_config.model
        self._client = anthropic.Anthropic(api_key=role_config.api_key)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        cap = max_output_tokens or self._cfg.max_output_tokens
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": cap,
            "temperature": (
                self._cfg.temperature if temperature is None else temperature
            ),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system  # top-level, see module docstring

        if json_schema is not None:
            kwargs["tools"] = [
                {
                    "name": _TOOL_NAME,
                    "description": "Return the structured response.",
                    "input_schema": json_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": _TOOL_NAME}

        resp = with_retry(
            lambda: self._client.messages.create(**kwargs),
            what=f"anthropic {self._cfg.role}",
        )

        if resp.stop_reason == "max_tokens":
            raise TruncatedCompletion(
                f"role {self._cfg.role!r} hit its output cap of {cap} tokens, so "
                "the response is a fragment.\n"
                f"Fix: raise {self._cfg.role}.max_output_tokens in "
                "configs/base.yaml."
            )

        text = _extract_text(resp, structured=json_schema is not None)
        usage = getattr(resp, "usage", None)
        return Completion(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            finish_reason="stop",
            model=self.model,
            provider=self.name,
            raw=resp,
        )

    def estimated_usd(self, input_tokens: int, output_tokens: int) -> float:
        pin, pout = price_for(self.model)
        return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def _extract_text(resp: Any, *, structured: bool) -> str:
    """Normalise to a string so callers never branch on provider.

    In structured mode the payload arrives as a tool-use block whose ``input``
    is already a parsed dict, so it is re-serialised here. That looks wasteful
    and is the right call anyway: every caller in this project then receives a
    JSON string regardless of vendor, and ``src/judge.py`` has exactly one
    parsing path to get right instead of three.
    """
    for part in resp.content or []:
        if structured and getattr(part, "type", "") == "tool_use":
            return json.dumps(part.input, ensure_ascii=False)
    chunks = [
        part.text
        for part in (resp.content or [])
        if getattr(part, "type", "") == "text"
    ]
    return "\n".join(chunks).strip()


def build(role_config: RoleConfig) -> AnthropicProvider:
    return AnthropicProvider(role_config)

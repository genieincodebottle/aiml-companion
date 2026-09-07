"""OpenAI-compatible adapter.

One adapter, four deployment stories. Anything speaking the chat-completions
shape works here by changing ``providers.openai_compatible.base_url`` in
``configs/base.yaml``:

    OpenAI          https://api.openai.com/v1
    vLLM            http://localhost:8000/v1
    Ollama          http://localhost:11434/v1
    Together / LM Studio / most self-hosted gateways

That last group is the reason this adapter earns its place in a teaching repo.
A learner who wants to run the full lifecycle against a real model, at zero
marginal cost, points this at a local server and everything else in the project
is unchanged. It also gives the cheapest honest answer to the self-preference
question: put the judge on a local open-weight model and the generator on
Gemini, and the judge can no longer be rewarding its own house style.
"""

from __future__ import annotations

import json
from typing import Any

from ..config import RoleConfig
from . import Completion, TruncatedCompletion
from ._common import price_for, with_retry


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, role_config: RoleConfig) -> None:
        from openai import OpenAI  # lazy: only this adapter needs it

        self._cfg = role_config
        self.model = role_config.model
        self._client = OpenAI(
            api_key=role_config.api_key or "not-needed-for-local-servers",
            base_url=role_config.base_url or "https://api.openai.com/v1",
        )

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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                self._cfg.temperature if temperature is None else temperature
            ),
            "max_tokens": cap,
        }

        if json_schema is not None:
            # Strict structured outputs where the server supports them. Many
            # local servers implement only `json_object`, which constrains the
            # syntax but not the shape, so the caller still has to validate the
            # parsed object rather than trusting the field set. That validation
            # lives in src/judge.py and is not optional for that reason.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        resp = with_retry(
            lambda: self._client.chat.completions.create(**kwargs),
            what=f"openai_compatible {self._cfg.role}",
        )

        choice = resp.choices[0]
        finish = choice.finish_reason or "stop"
        if finish == "length":
            raise TruncatedCompletion(
                f"role {self._cfg.role!r} hit its output cap of {cap} tokens, so "
                "the response is a fragment.\n"
                f"Fix: raise {self._cfg.role}.max_output_tokens in "
                "configs/base.yaml."
            )

        usage = getattr(resp, "usage", None)
        return Completion(
            text=(choice.message.content or "").strip(),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            finish_reason="stop" if finish == "stop" else finish,
            model=self.model,
            provider=self.name,
            raw=resp,
        )

    def estimated_usd(self, input_tokens: int, output_tokens: int) -> float:
        # A local server costs nothing per token, so a non-zero estimate here
        # would be actively misleading when sizing a retry budget.
        if _is_local(self._cfg.base_url):
            return 0.0
        pin, pout = price_for(self.model)
        return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def _is_local(base_url: str | None) -> bool:
    return bool(base_url) and any(
        host in base_url for host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    )


def build(role_config: RoleConfig) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(role_config)


# Re-exported for the contract test, which checks that every adapter agrees on
# what a parsed JSON response looks like.
def parse_json(text: str) -> Any:
    return json.loads(text)

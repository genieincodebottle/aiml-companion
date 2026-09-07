"""Gemini adapter (``google-genai``). The default provider.

Two things in here are not boilerplate and are worth reading before you copy
this file into your own project: the thinking budget, and the truncation check.
They are the same bug seen from two sides, and together they are the most
expensive lesson in this repository.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import RoleConfig
from . import Completion, TruncatedCompletion
from ._common import price_for, with_retry

# Roles whose output is a schema-constrained JSON object. These get thinking
# switched off; see _thinking_budget_for below.
_STRUCTURED_ROLES = {"judge", "meta_judge"}

log = logging.getLogger(__name__)


class GeminiProvider:
    name = "gemini"

    def __init__(self, role_config: RoleConfig) -> None:
        # Imported lazily. Only this adapter needs the SDK, so someone running
        # the project on `stub` or on a local OpenAI-compatible server never has
        # to install it.
        from google import genai
        from google.genai import types

        self._types = types
        self._cfg = role_config
        self.model = role_config.model
        self._client = genai.Client(api_key=role_config.api_key)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Completion:
        types = self._types
        cap = max_output_tokens or self._cfg.max_output_tokens

        kwargs: dict[str, Any] = {
            "temperature": (
                self._cfg.temperature if temperature is None else temperature
            ),
            "max_output_tokens": cap,
            "system_instruction": system,
        }

        if json_schema is not None:
            # A schema, not "reply in JSON please". Schema-constrained decoding
            # gives a parser that never fails; the polite request gives one that
            # fails on roughly one call in ten. Across a tuning run that is the
            # difference between a pipeline and a coin flip.
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = json_schema

        budget = self._thinking_budget_for(json_schema)
        if budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)

        resp = self._generate(kwargs, prompt)
        cap_used = kwargs["max_output_tokens"]

        finish = _finish_reason(resp)
        if finish == "MAX_TOKENS":
            # Raise, do not return. A truncated judge verdict is the worst
            # failure mode this system has, because it does not look like a
            # failure: the JSON prefix may still parse into {"label": "FAIL"}
            # with a reason cut off after four words, and that fragment is then
            # appended to the generator's revision prompt as its instruction.
            # The pipeline stays green and the revision loop is being steered by
            # half a sentence.
            raise TruncatedCompletion(
                f"role {self._cfg.role!r} hit its output cap of {cap_used} tokens, so "
                "the response is a fragment.\n"
                f"Fix: raise {self._cfg.role}.max_output_tokens in "
                "configs/base.yaml.\n"
                "If this role is a thinking model, note that reasoning tokens "
                "come out of this same budget - raising the cap may just buy "
                "more deliberation. See _thinking_budget_for in this file."
            )

        meta = getattr(resp, "usage_metadata", None)
        # Thinking tokens are reported SEPARATELY from candidates_token_count
        # and are billed as output. Counting only the visible response
        # understates cost by whatever the reasoning:output ratio happens to be,
        # and that ratio is not small: measured on this project's generator
        # prompt, gemini-3.5-flash spent 1,746 thinking tokens to produce a
        # 40-token sentence. A cost report built on candidates_token_count alone
        # would have been out by a factor of forty-four, in the direction that
        # makes a retry budget look affordable.
        thoughts = getattr(meta, "thoughts_token_count", 0) or 0
        return Completion(
            text=(resp.text or "").strip(),
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=(getattr(meta, "candidates_token_count", 0) or 0) + thoughts,
            finish_reason="stop" if finish in ("STOP", "") else "length",
            model=self.model,
            provider=self.name,
            raw=resp,
        )

    def _generate(self, kwargs: dict[str, Any], prompt: str) -> Any:
        """Send the request, and survive a model that refuses to stop thinking.

        Some Gemini models - the Pro tier at time of writing - reject
        ``thinking_budget: 0`` outright with
        ``400 INVALID_ARGUMENT: Budget 0 is invalid. This model only works in
        thinking mode.``

        That turns a sensible per-role default into a hard failure the moment
        somebody points the judge at a Pro model, which is a reasonable thing to
        want and gives no hint in the error that the *thinking* setting is the
        problem rather than the schema or the key. So the disable is treated as
        a preference rather than a requirement: on that specific rejection, drop
        the thinking config and send the request again.

        The warning is not decorative. Thinking is left on for a structured
        verdict, and reasoning tokens then share the output budget with the JSON
        (see `_thinking_budget_for`), so the caller needs to know their
        truncation risk just went up.
        """
        types = self._types

        def send(config_kwargs: dict[str, Any]) -> Any:
            cfg = types.GenerateContentConfig(**config_kwargs)
            return with_retry(
                lambda: self._client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                ),
                what=f"gemini {self._cfg.role}",
            )

        try:
            return send(kwargs)
        except Exception as exc:  # noqa: BLE001 - vendor raises many types
            if not _rejects_zero_thinking(exc) or "thinking_config" not in kwargs:
                raise
            log.warning(
                "model %r for role %r refuses thinking_budget=0, so it is being "
                "retried with thinking ON. Verdicts from this role are now less "
                "reproducible and share their output budget with reasoning "
                "tokens - raise %s.max_output_tokens if you see truncation.",
                self.model, self._cfg.role, self._cfg.role,
            )
            return send({k: v for k, v in kwargs.items() if k != "thinking_config"})

    def _thinking_budget_for(self, json_schema: dict[str, Any] | None) -> int | None:
        """Thinking off for structured verdicts, on for everything else.

        This is the single most consequential decision in the file.

        On a thinking model the reasoning tokens are drawn from the SAME
        ``max_output_tokens`` budget as the visible response. A judge verdict is
        a small JSON object - a label and a sentence - but the model will
        happily spend the entire budget deliberating and emit a truncated
        fragment. Raising the ceiling does not fix it. The failure is identical
        at 1,024 and at 8,192, because the extra room is spent thinking, not
        writing.

        It is also the wrong tool for the job. Applying a written rubric to a
        short piece of text is closer to transcription than to deliberation:
        the criteria are fixed, the vocabulary is closed, and the schema has
        already decided the shape of the answer. Deliberation buys nothing here
        and costs reproducibility - and a judge that is not reproducible cannot
        be monitored for drift, because you can no longer separate drift from
        the model's own variance.

        The reflector is the opposite case and keeps its thinking. Proposing a
        better rubric from a set of the judge's mistakes is a genuine reasoning
        task, and it runs a few dozen times per tuning run rather than once per
        served request, so the tokens are affordable.
        """
        # An explicit per-role setting wins. It is the honest way to express
        # this, because the right budget is a property of the TASK rather than
        # of whether the call happens to carry a schema.
        if self._cfg.thinking_budget is not None:
            return self._cfg.thinking_budget
        if self._cfg.role in _STRUCTURED_ROLES or json_schema is not None:
            return 0
        return None  # vendor default

    def estimated_usd(self, input_tokens: int, output_tokens: int) -> float:
        pin, pout = price_for(self.model)
        return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def _rejects_zero_thinking(exc: Exception) -> bool:
    """Is this the specific "I will not stop thinking" 400?

    Matched on the message rather than the status code, because a 400 has many
    causes and retrying the wrong one without the thinking config just produces
    the same failure more slowly.
    """
    text = str(exc).lower()
    return "budget 0 is invalid" in text or "only works in thinking mode" in text


def _finish_reason(resp: Any) -> str:
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return ""
    reason = getattr(candidates[0], "finish_reason", None)
    return getattr(reason, "name", str(reason or ""))


def build(role_config: RoleConfig) -> GeminiProvider:
    return GeminiProvider(role_config)

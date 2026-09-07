"""Two clients behind one call.

LiveClient   talks to any OpenAI-compatible endpoint, which covers llama-server,
             vLLM, SGLang, Ollama and the hosted providers. Needs a running
             model, so it needs a GPU or a Mac with enough unified memory.

OfflineClient needs nothing. It is the default, so every script in this repo
             runs on a laptop.

OfflineClient is not a mock that returns a fixed string. It emits genuine
Glimmer-shaped output, channel headers and ATEM blocks included, composed from
the actual question and the actual tool catalogue. That matters because the
thing this lab teaches is the *format*, and a mock that returned plain prose
would teach nothing. The parsers in channels.py and atem.py cannot tell the two
clients apart, which is the property worth having.

What OfflineClient cannot give you is the model's judgement. It will not
surprise you, and it will not be wrong in the interesting ways a real model is
wrong. Point it at a real server once you have one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from . import atem
from .config import BASE_URL, MODEL_NAME, SAMPLING, API_KEY
from .channels import STOP_TOKENS


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Response:
    """A raw generation, before channels.parse touches it."""

    raw: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    mode: str = "offline"


def system_prompt(reasoning_strength: str = "high", tools_text: str = "") -> str:
    """Build the system prompt, with one documented trap.

    The model card says to set effort with a `Reasoning strength: <value>`
    line in the system prompt. On a served endpoint that is not the whole
    story. Glimmer's chat template appends its own reasoning directive *after*
    whatever you wrote, and its default is high, so a system prompt asking for
    low can be silently overridden by the template that renders it.

    If low latency matters, verify what you actually got rather than trusting
    the system prompt. Ask the server to echo the rendered prompt, or set the
    template's own parameter, which llama-server exposes as
    `{"chat_template_kwargs": {"reasoning_strength": "low"}}`.
    """
    parts = [f"Reasoning strength: {reasoning_strength}"]
    if tools_text:
        parts.append("You have these tools.\n" + tools_text)
    return "\n\n".join(parts)


class LiveClient:
    """OpenAI-compatible chat completions, with the Glimmer-specific settings.

    Two request fields are not optional for this model.

    `stop` lists only the true turn terminators. Adding `<|eom|>` here is the
    single most common way to break an agentic run, because it stops the model
    the moment it asks for a tool.

    Special tokens must survive decoding, or the channel headers this whole lab
    parses are deleted before you see them. vLLM's reasoning parser forces
    that; llama-server needs to be told.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        model: str = MODEL_NAME,
        api_key: str = API_KEY,
        timeout: float = 600.0,
    ):
        import httpx

        self.model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def complete(self, messages: list[Message], **overrides: Any) -> Response:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stop": list(STOP_TOKENS),
            # Without this the <|start|> and <|message|> markers never reach us
            # and every message collapses into one undifferentiated string.
            "skip_special_tokens": False,
            **SAMPLING,
            **overrides,
        }
        reply = self._client.post("/chat/completions", json=payload)
        reply.raise_for_status()
        body = reply.json()
        usage = body.get("usage", {})
        return Response(
            raw=body["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            mode="live",
        )

    def close(self) -> None:
        self._client.close()


class OfflineClient:
    """Deterministic Glimmer-shaped output, no model required.

    The policy is simple and stated in one place so it is easy to argue with.
    A question mentioning arithmetic gets a `calculate` call. One mentioning
    memory or cache gets `kv_cache_gib`. Anything else gets one `search_docs`
    call, and after any tool result it writes a final answer.

    Reasoning strength changes the length of the `to=self` message, the way it
    does on the real model, so scripts/02_reasoning_strength.py shows a real
    difference rather than a hardcoded one.
    """

    _EFFORT_LINES = {
        "low": 1,
        "medium": 2,
        "high": 4,
        "xhigh": 6,
    }

    def __init__(self, model: str = MODEL_NAME):
        self.model = model
        self.calls: list[list[Message]] = field(default_factory=list)  # type: ignore[assignment]
        self.calls = []

    def complete(self, messages: list[Message], **overrides: Any) -> Response:
        self.calls.append(list(messages))
        strength = self._strength(messages)
        question = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        already_used = any(m.role == "tool" for m in messages)

        reasoning = self._reasoning(question, strength, already_used)
        out = [f"<|start|>assistant to=self<|message|>{reasoning}<|eom|>"]

        if already_used:
            out.append(
                f"<|start|>assistant to=user<|message|>{self._answer(messages)}<|eot|>"
            )
        else:
            name, arguments = self._choose_tool(question)
            block = atem.render_tool_call(name, arguments)
            out.append(f"<|start|>assistant to={name}<|message|>{block}<|eom|>")

        raw = "".join(out)
        return Response(
            raw=raw,
            prompt_tokens=sum(len(m.content) for m in messages) // 4,
            completion_tokens=len(raw) // 4,
            mode="offline",
        )

    # -- the small policy that stands in for a model --------------------------

    @staticmethod
    def _strength(messages: list[Message]) -> str:
        for message in messages:
            if message.role == "system":
                found = re.search(
                    r"Reasoning strength:\s*(low|medium|high|xhigh)", message.content
                )
                if found:
                    return found.group(1)
        return "high"

    def _reasoning(self, question: str, strength: str, has_tool_result: bool) -> str:
        steps = [
            f"The user asks about {question.strip()[:60] or 'something unstated'}.",
            "I should check the tool catalogue before answering from memory.",
            "The documentation index is local, so a lookup is cheap and cannot be stale.",
            "I will keep the answer to what the tool actually returned.",
            "If the result is thin I can call a second tool rather than guess.",
            "Nothing here needs a plan longer than one step.",
        ]
        if has_tool_result:
            steps = ["I have the tool result. I will state it plainly and stop."]
        return " ".join(steps[: self._EFFORT_LINES.get(strength, 4)])

    @staticmethod
    def _choose_tool(question: str) -> tuple[str, dict[str, Any]]:
        lowered = question.lower()
        expression = re.search(r"[\d]+\s*[-+*/]\s*[\d\s+*/().-]+", question)
        if expression and any(w in lowered for w in ("calculate", "what is", "compute", "+", "*")):
            return "calculate", {"expression": expression.group(0).strip()}
        if any(w in lowered for w in ("memory", "kv", "cache", "vram")):
            context = re.search(r"(\d[\d,_]{3,})", question)
            return "kv_cache_gib", {
                "context": (context.group(1).replace(",", "").replace("_", "") if context else "131072")
            }
        return "search_docs", {"query": question.strip()[:80] or "muse glimmer", "top_k": "3"}

    @staticmethod
    def _answer(messages: list[Message]) -> str:
        result = next((m.content for m in reversed(messages) if m.role == "tool"), "")
        return f"Based on the tool result: {result}"

    def close(self) -> None:
        pass


def build_client(mode: str | None = None):
    """Pick a client from GLIMMER_MODE, defaulting to offline.

    The import of httpx lives inside LiveClient so that offline mode has no
    third-party import at all on the path a learner runs first.
    """
    from .config import MODE

    chosen = (mode or MODE).lower()
    if chosen == "live":
        return LiveClient()
    return OfflineClient()

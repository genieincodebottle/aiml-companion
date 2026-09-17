"""A minimal Gemini client for JSON output, safe to call from several threads.

Two details are not boilerplate. Thinking is switched off where the model allows
it, because thinking tokens share the output budget and a truncated JSON layout
is useless. And a response cut off at the token cap raises instead of returning,
because a half-finished layout would silently score zero and look like a bad idea
rather than a broken call.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

# Published per-1M-token prices, used only for a rough estimate on screen.
# Vendors change prices; check the current pricing page before relying on it.
PRICES = {"gemini-3.5-flash": (0.30, 2.50)}


def find_api_key(start: Path | None = None) -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    # Look for a .env in this project folder, then in each parent folder, so a
    # key at the repository root also works.
    here = (start or Path(__file__)).resolve()
    for folder in [here, *here.parents]:
        env = folder / ".env"
        if env.is_file():
            for line in _read_text_any_encoding(env).splitlines():
                key, _, value = line.partition("=")
                key = key.strip().removeprefix("export ").strip()
                if key in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and value.strip():
                    return value.strip().strip('"').strip("'")
    return None


def _read_text_any_encoding(path: Path) -> str:
    """Windows PowerShell's `echo ... > .env` writes UTF-16, and Notepad may add a
    byte-order mark. Both would hide the key from a plain UTF-8 read."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


class GeminiClient:
    def __init__(self, api_key: str | None = None) -> None:
        from google import genai
        from google.genai import types

        key = api_key or find_api_key()
        if not key:
            raise RuntimeError(
                "No GEMINI_API_KEY found.\n"
                "Create a file named .env in the dream-rsi-explorer folder containing one line:\n"
                "GEMINI_API_KEY=your-key-here"
            )
        self._types = types
        self._client = genai.Client(api_key=key)
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.failures = 0

    def usage_line(self, model: str) -> str:
        line = f"Gemini usage: {self.calls} calls, {self.input_tokens:,} input tokens, {self.output_tokens:,} output tokens"
        if model in PRICES:
            cost = self.input_tokens / 1e6 * PRICES[model][0] + self.output_tokens / 1e6 * PRICES[model][1]
            line += f", roughly ${cost:.2f} at list price"
        if self.failures:
            line += f", {self.failures} failed calls"
        return line

    def generate_json(
        self, *, model: str, prompt: str, schema: dict, max_output_tokens: int, temperature: float
    ) -> dict:
        types = self._types
        base = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        try:
            try:
                resp = self._send(model, prompt, {**base, "thinking_config": types.ThinkingConfig(thinking_budget=0)})
            except Exception as exc:  # Pro-tier models reject a zero thinking budget
                if "thinking" not in str(exc).lower():
                    raise
                resp = self._send(model, prompt, base)
        except Exception:
            with self._lock:
                self.failures += 1
            raise

        meta = getattr(resp, "usage_metadata", None)
        with self._lock:
            self.calls += 1
            self.input_tokens += getattr(meta, "prompt_token_count", 0) or 0
            # Thinking tokens are billed as output but reported in their own field.
            self.output_tokens += (getattr(meta, "candidates_token_count", 0) or 0) + (
                getattr(meta, "thoughts_token_count", 0) or 0
            )

        finish = str(getattr(resp.candidates[0], "finish_reason", "")) if resp.candidates else ""
        if "MAX_TOKENS" in finish:
            raise RuntimeError(f"{model} hit max_output_tokens={max_output_tokens}, response is a fragment")
        return json.loads(resp.text)

    def _send(self, model: str, prompt: str, config: dict):
        cfg = self._types.GenerateContentConfig(**config)
        for attempt in range(5):
            try:
                return self._client.models.generate_content(model=model, contents=prompt, config=cfg)
            except Exception as exc:
                transient = any(s in str(exc) for s in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                if not transient or attempt == 4:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("unreachable")

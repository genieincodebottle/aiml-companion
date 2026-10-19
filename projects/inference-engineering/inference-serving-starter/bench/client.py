"""Streaming OpenAI-compatible client. GIVEN.

Works against engine_sim and against a real `vllm serve` (the gpu/ path). It does not
interpret anything: it hands you the raw timestamps and you turn them into TTFT, ITL and
E2E in bench/records.py.

Two time bases:
  * chunk_wall  - wall-clock seconds (time.monotonic) when each content chunk arrived.
  * chunk_sim   - engine_sim only: the engine's own timestamp for each ENGINE STEP the stream
                  covered (the server may batch several steps into one HTTP chunk, so this list
                  can be longer than chunk_wall). None against a real server. Prefer it when
                  present: at time_scale 10 a single decode step is under a millisecond of wall
                  time and the wall clock cannot resolve it.
                  A real vLLM sends one chunk per scheduler step per request, which is the same
                  unit, so ITL means the same thing on both.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


@dataclass
class RawResult:
    rid: int
    prompt_tokens: int
    max_tokens: int
    send_sim: float                       # client clock (sim seconds) when the request was sent
    wall_start: float = 0.0
    wall_end: float = 0.0
    ok: bool = False
    error: Optional[str] = None
    chunk_wall: list = field(default_factory=list)
    chunk_sim: Optional[list] = None      # engine time per engine step (engine_sim only)
    chunk_tokens: Optional[list] = None   # tokens per engine step (engine_sim only)
    sim_arrival: Optional[float] = None   # x_sim_arrival (engine_sim only)
    completion_tokens: Optional[int] = None   # from the final usage chunk


async def fetch_info(client: httpx.AsyncClient, base_url: str) -> dict:
    """engine_sim's /sim/info, or {} against a real server (which has no such route)."""
    try:
        r = await client.get(base_url + "/sim/info", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


async def stream_completion(client: httpx.AsyncClient, base_url: str, model: str, prompt_ids: list,
                            max_tokens: int, rid: int, send_sim: float) -> RawResult:
    res = RawResult(rid=rid, prompt_tokens=len(prompt_ids), max_tokens=max_tokens, send_sim=send_sim)
    body = {
        "model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
        "ignore_eos": True,   # vLLM extension: always generate exactly max_tokens
    }
    res.wall_start = time.monotonic()
    try:
        async with client.stream("POST", base_url + "/v1/completions", json=body, timeout=None) as r:
            if r.status_code != 200:
                res.error = f"HTTP {r.status_code}"
                res.wall_end = time.monotonic()
                return res
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if "error" in obj:
                    res.error = str(obj["error"])
                    break
                usage = obj.get("usage")
                if usage:
                    res.completion_tokens = usage.get("completion_tokens")
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    res.chunk_wall.append(time.monotonic())
                    steps = obj.get("x_steps")
                    if steps is None and "x_sim_t" in obj:
                        steps = [[obj["x_sim_t"], obj.get("x_n_tokens", 1)]]
                    if steps is not None:
                        if res.chunk_sim is None:
                            res.chunk_sim, res.chunk_tokens = [], []
                        for t, n in steps:      # one entry per engine step, however the server batched them
                            res.chunk_sim.append(t)
                            res.chunk_tokens.append(n)
                    if "x_sim_arrival" in obj:
                        res.sim_arrival = obj["x_sim_arrival"]
        res.ok = res.error is None and len(res.chunk_wall) > 0
    except Exception as e:   # network failure is a data point, not a crash
        res.error = f"{type(e).__name__}: {e}"
    res.wall_end = time.monotonic()
    return res

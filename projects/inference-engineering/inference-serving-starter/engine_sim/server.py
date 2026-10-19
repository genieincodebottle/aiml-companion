"""OpenAI-compatible HTTP front end for engine_sim.

Endpoints
    POST /v1/completions        prompt = string OR a list of token ids, stream true/false
    POST /v1/chat/completions   messages -> one token per whitespace-separated word
    GET  /v1/models
    GET  /metrics               vLLM-style Prometheus text
    GET  /health
    POST /reset_prefix_cache    same path a real vLLM server uses
    GET  /sim/info              engine_sim only: config and time_scale. A real vLLM returns 404.

TIME. The server owns a virtual clock = wall seconds * time_scale, so a run that takes
10 minutes of "engine time" finishes in 10 minutes / time_scale of wall time. Every
streamed chunk carries the engine's own timestamps, because the wall clock cannot resolve a
20 ms decode step when it is being sped up:

    x_sim_t        engine time of the last step in this chunk
    x_n_tokens     tokens in this chunk
    x_steps        [[t, n], ...] one entry per engine step the chunk covers
    x_sim_arrival  (first chunk only) engine time at which the server received the request

To keep the HTTP layer from becoming the bottleneck at high token rates, the server batches the
steps of one request into a chunk about every FLUSH_SIM_S of engine time. The first chunk and
the last chunk are never delayed. Your benchmark must use the x_ fields when they are present
and fall back to the wall clock when they are not (a real vLLM sends none of them).

The model behaves as if it emitted exactly `max_tokens` tokens (like `ignore_eos`).
"""
from __future__ import annotations

import asyncio
import itertools
import json
import sys
import time
import zlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .config import EngineConfig
from .core import Engine, Request as SimRequest

FLUSH_SIM_S = 0.5


def tokenize_text(text: str) -> list:
    """The simulator's tokenizer: one token per whitespace-separated word."""
    return [zlib.crc32(w.encode("utf-8")) for w in text.split()]


def _raise_timer_resolution() -> None:
    """Windows sleeps in 15.6 ms steps unless asked otherwise. At time_scale 10 that is 0.16 s of
    engine time between a request finishing and the next being seen, which would bias a
    closed-loop benchmark. Ask for 1 ms. No effect on other platforms."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass


def create_app(cfg: EngineConfig) -> FastAPI:
    _raise_timer_resolution()
    eng = Engine(cfg)
    waiters: dict = {}      # rid -> asyncio.Queue of lists of TokenEvent
    buffers: dict = {}      # rid -> (events not yet sent, engine time of last flush)
    counter = itertools.count(1)
    st = {"t0": time.monotonic(), "max_lag": 0.0}
    model_name = cfg.model_spec.name

    def sim_now() -> float:
        return (time.monotonic() - st["t0"]) * cfg.time_scale

    def dispatch(events: list) -> None:
        for e in events:
            q = waiters.get(e.rid)
            if q is None:
                buffers.pop(e.rid, None)
                continue
            pending, last_flush = buffers.get(e.rid, ([], -1e18))
            pending.append(e)
            if e.first or e.done or e.error or e.t - last_flush >= FLUSH_SIM_S:
                q.put_nowait(pending)
                buffers[e.rid] = ([], e.t)
            else:
                buffers[e.rid] = (pending, last_flush)

    async def driver() -> None:
        while True:
            target = sim_now()
            dispatch(eng.run_until(target))
            if eng.busy:
                st["max_lag"] = max(st["max_lag"], target - eng.now)
            await asyncio.sleep(0.0005 if eng.busy else 0.002)

    @asynccontextmanager
    async def lifespan(_app):
        st["t0"] = time.monotonic()
        task = asyncio.create_task(driver())
        yield
        task.cancel()

    app = FastAPI(title="engine_sim", lifespan=lifespan)

    def prompt_ids_from(prompt) -> list:
        if isinstance(prompt, str):
            return tokenize_text(prompt)
        if isinstance(prompt, list) and all(isinstance(x, int) for x in prompt):
            return list(prompt)
        raise HTTPException(400, "prompt must be a string or a list of token ids")

    def submit(ids: list, max_tokens: int):
        if max_tokens < 1:
            raise HTTPException(400, "max_tokens must be >= 1")
        if not ids:
            raise HTTPException(400, "empty prompt")
        rid = f"cmpl-{next(counter)}"
        q: asyncio.Queue = asyncio.Queue()
        waiters[rid] = q
        eng.submit(SimRequest(rid, sim_now(), ids, max_tokens))
        return rid, q

    def chunk_obj(chat: bool, rid: str, batch: list, final_reason):
        last = batch[-1]
        n = sum(e.n for e in batch)
        text = " tok" * n
        base = {"id": rid, "created": int(time.time()), "model": model_name,
                "x_sim_t": last.t, "x_n_tokens": n, "x_steps": [[e.t, e.n] for e in batch]}
        if batch[0].first:
            base["x_sim_arrival"] = batch[0].arrival
            base["x_cached_tokens"] = batch[0].cached_tokens
        if chat:
            base["object"] = "chat.completion.chunk"
            base["choices"] = [{"index": 0, "delta": {"content": text}, "finish_reason": final_reason}]
        else:
            base["object"] = "text_completion"
            base["choices"] = [{"index": 0, "text": text, "logprobs": None, "finish_reason": final_reason}]
        return base

    async def handle(body: dict, ids: list, chat: bool):
        max_tokens = int(body.get("max_tokens") or 16)
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        rid, q = submit(ids, max_tokens)

        if not stream:
            total = 0
            try:
                while True:
                    batch = await q.get()
                    if batch[-1].error:
                        raise HTTPException(400, batch[-1].error)
                    total += sum(e.n for e in batch)
                    if batch[-1].done:
                        break
            finally:
                waiters.pop(rid, None)
                buffers.pop(rid, None)
            last = batch[-1]
            usage = {"prompt_tokens": last.prompt_tokens, "completion_tokens": total,
                     "total_tokens": last.prompt_tokens + total}
            if chat:
                ch = [{"index": 0, "message": {"role": "assistant", "content": " tok" * total},
                       "finish_reason": "length"}]
                obj = "chat.completion"
            else:
                ch = [{"index": 0, "text": " tok" * total, "finish_reason": "length"}]
                obj = "text_completion"
            return JSONResponse({"id": rid, "object": obj, "created": int(time.time()), "model": model_name,
                                 "choices": ch, "usage": usage, "x_sim_t": last.t, "x_sim_arrival": last.arrival})

        async def gen():
            total = 0
            try:
                while True:
                    batch = await q.get()
                    last = batch[-1]
                    if last.error:
                        yield "data: " + json.dumps({"error": {"message": last.error}}) + "\n\n"
                        break
                    total += sum(e.n for e in batch)
                    obj = chunk_obj(chat, rid, batch, "length" if last.done else None)
                    yield "data: " + json.dumps(obj) + "\n\n"
                    if last.done:
                        if include_usage:
                            u = {"id": rid, "object": obj["object"], "model": model_name, "choices": [],
                                 "usage": {"prompt_tokens": last.prompt_tokens, "completion_tokens": total,
                                           "total_tokens": last.prompt_tokens + total}}
                            yield "data: " + json.dumps(u) + "\n\n"
                        break
                yield "data: [DONE]\n\n"
            finally:
                waiters.pop(rid, None)
                buffers.pop(rid, None)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        return await handle(body, prompt_ids_from(body.get("prompt")), chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        msgs = body.get("messages") or []
        ids: list = []
        for m in msgs:
            ids.extend(tokenize_text(f"{m.get('role', '')} {m.get('content', '')}"))
        return await handle(body, ids, chat=True)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(eng.render_metrics(), media_type="text/plain; version=0.0.4")

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache():
        eng.reset_prefix_cache()
        return {"status": "ok"}

    @app.get("/sim/info")
    async def info():
        return {"engine_sim": True, "time_scale": cfg.time_scale, "sim_now": sim_now(),
                "engine_now": eng.now, "max_lag_s": st["max_lag"], "config": cfg.describe()}

    return app

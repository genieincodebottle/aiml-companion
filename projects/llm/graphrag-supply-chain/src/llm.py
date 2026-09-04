"""The only place in this project that talks to a model provider.

Three jobs, deliberately kept in one small module:

  embed_documents / embed_query   ->  vectors for the Neo4j vector index
  extract_json                    ->  structured output for graph extraction
  generate                        ->  the grounded answer

Why one module?  Because provider APIs churn.  When Google renames a model or
changes ``embed_content``, exactly one file changes and the retrieval code, the
ingestion code and the Streamlit app are untouched.  That is the whole point of
a boundary.

Why not LangChain here?  This project's teaching goal is the *graph*, and a
framework would hide the two things a learner most needs to see: the exact JSON
schema handed to the extractor, and the exact vector handed to Cypher.  For a
production system spanning many providers and chains, the abstraction pays for
itself; for a teaching pipeline this size it costs more than it returns.  That
trade-off is discussed in the README.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

from google import genai
from google.genai import types

from .config import Config, get_config

log = logging.getLogger(__name__)

# Published per-1M-token rates for the Flash tier, used only to give the UI an
# order-of-magnitude number.  Treat it as an estimate, never as a bill.
_USD_PER_1M_INPUT = 0.30
_USD_PER_1M_OUTPUT = 2.50


class Usage:
    """Running token/cost tally for one process.  Surfaced in the Streamlit UI
    so a learner can see what each retrieval strategy actually costs."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.embed_calls = 0
        self.embed_cache_hits = 0

    def add(self, resp: Any) -> None:
        self.calls += 1
        meta = getattr(resp, "usage_metadata", None)
        if meta is None:
            return
        self.input_tokens += getattr(meta, "prompt_token_count", 0) or 0
        self.output_tokens += getattr(meta, "candidates_token_count", 0) or 0

    @property
    def estimated_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * _USD_PER_1M_INPUT
            + self.output_tokens / 1_000_000 * _USD_PER_1M_OUTPUT
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "embed_calls": self.embed_calls,
            "embed_cache_hits": self.embed_cache_hits,
            "estimated_usd": round(self.estimated_usd, 6),
        }


class LLMClient:
    """Thin wrapper over google-genai with retries, an embedding cache and
    usage accounting."""

    def __init__(self, config: Config | None = None,
                 cache_dir: Path | None = None) -> None:
        self.config = config or get_config()
        self._client = genai.Client(api_key=self.config.google_api_key)
        self.usage = Usage()
        self.cache_dir = cache_dir or (self.config.root / "artifacts" / "embed_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ chat
    def generate(self, prompt: str, *, system: str | None = None,
                 temperature: float | None = None) -> str:
        """Answering. Unlike extraction, thinking stays ON here.

        Answering a multi-hop question is a genuine reasoning task: the model
        has to read a derived path, connect it to quoted evidence, and decide
        what is stated versus inferred. Turning thinking off measurably hurts
        that. But thinking tokens come out of the SAME output budget, so the
        budget has to accommodate both.
        """
        cfg = types.GenerateContentConfig(
            temperature=self.config.llm["temperature"] if temperature is None else temperature,
            max_output_tokens=self.config.llm["max_output_tokens"],
            system_instruction=system,
        )
        resp = self._call(prompt, cfg)

        # The same truncation check as extract_json, and it belongs here for the
        # same reason.
        #
        # This was found the expensive way. The answering path originally had no
        # such check, and on the flagship multi-hop question the model spent
        # 1,899 of its 2,048 tokens thinking and 145 writing. The answer stopped
        # after the first of four products. Nothing errored: retrieval was
        # correct, the graph facts were in context, and the truncated answer read
        # as a complete one that had simply found less.
        #
        # It corrupted the BENCHMARK, not just one response - term coverage was
        # measuring how much of the answer fit in the budget rather than how much
        # of the evidence the system found. Guarding one call site and not the
        # other was the actual mistake.
        if _finish_reason(resp) == "MAX_TOKENS":
            raise RuntimeError(
                "The model hit its output token limit while answering, so the "
                "response is cut off mid-sentence.\n"
                f"Current limit: {cfg.max_output_tokens} tokens, shared between "
                "reasoning and the visible answer.\n"
                "Fix: raise `llm.max_output_tokens` in configs/base.yaml.\n"
                "This is raised rather than returned quietly because a truncated "
                "answer reads exactly like a complete one that found less, which "
                "silently corrupts any measurement taken over it."
            )
        return (resp.text or "").strip()

    def extract_json(self, prompt: str, schema: dict[str, Any], *,
                     system: str | None = None,
                     max_output_tokens: int | None = None) -> Any:
        """Structured output.  We hand the model a JSON schema rather than
        asking it to "reply in JSON" - the difference is a parser that never
        fails versus one that fails on roughly one document in ten.

        ``max_output_tokens`` defaults to the *extraction* budget, not the
        answering one, and the two are deliberately different.  An answer is a
        few paragraphs; an extraction from a full document is dozens of objects
        each carrying a verbatim evidence quote, and it is routinely five times
        longer than the answer the same model would write about the same text.

        This was not a theoretical concern.  Built with a shared 2048-token
        budget, this pipeline extracted **nothing at all** from **every**
        document, and reported it as a JSON parse error - see the truncation
        check below for why that diagnosis was so misleading.
        """
        cfg = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens or self.config.llm["extraction_max_output_tokens"],
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            # Thinking OFF for extraction, and this is the single most
            # consequential line in the file.
            #
            # On a thinking model, reasoning tokens are drawn from the SAME
            # max_output_tokens budget as the visible response.  Measured on
            # this corpus: a supplier profile that needs ~500 tokens of JSON
            # burned the entire 8,192-token budget on reasoning and returned a
            # truncated fragment.  Raising the budget does not fix it, it just
            # buys more reasoning; the pipeline failed identically at 2,048 and
            # at 8,192.
            #
            # It is also the wrong tool for the job.  Extraction is a
            # transcription task against a closed vocabulary with the schema
            # already constraining the shape - there is nothing to deliberate
            # about, and deliberation makes the output less reproducible, not
            # more accurate.  Answering is different and keeps its thinking; see
            # `generate` below.
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.config.llm["extraction_thinking_budget"]
            ),
        )
        resp = self._call(prompt, cfg)

        # Check truncation BEFORE trying to parse.  A response cut off at the
        # token limit is still valid JSON *prefix*, so json.loads fails with
        # "Unterminated string" and every instinct says the model formatted its
        # output wrongly.  It did not: the output was correct and the budget was
        # too small.  Diagnosing that from the parse error alone costs an hour;
        # reading finish_reason costs nothing, so read finish_reason.
        if _finish_reason(resp) == "MAX_TOKENS":
            raise RuntimeError(
                "The model hit its output token limit mid-JSON, so the "
                "extraction for this document is incomplete.\n"
                f"Current limit: {cfg.max_output_tokens} tokens.\n"
                "Fix: raise `llm.extraction_max_output_tokens` in "
                "configs/base.yaml, or split the document. Do NOT simply "
                "ignore the failure - a truncated extraction silently drops "
                "every entity after the cut, and the graph looks merely sparse "
                "rather than broken."
            )

        text = (resp.text or "").strip()
        if not text:
            log.warning("extractor returned empty output; treating as no findings")
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Should be unreachable now that truncation is handled above, but a
            # crash mid-ingest is worse than a logged skip.
            log.error("extractor emitted unparseable JSON: %.300s", text)
            return None

    def _call(self, prompt: str, cfg: types.GenerateContentConfig,
              attempts: int = 4) -> Any:
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._client.models.generate_content(
                    model=self.config.llm["model"], contents=prompt, config=cfg
                )
                self.usage.add(resp)
                return resp
            except Exception as exc:  # noqa: BLE001 - provider raises many types
                last = exc
                if not _is_retryable(exc) or attempt == attempts - 1:
                    raise
                delay = 2 ** attempt
                log.warning("LLM call failed (%s); retrying in %ss",
                            type(exc).__name__, delay)
                time.sleep(delay)
        raise RuntimeError("unreachable retry loop exit") from last

    # ------------------------------------------------------------- embeddings
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, task="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        """Note the different task type.  Gemini embeds a *question* into a
        different region of the space than a *passage*; using
        RETRIEVAL_DOCUMENT for both is a common and quietly expensive bug -
        recall drops and nothing errors."""
        return self._embed([text], task="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: Sequence[str], *, task: str) -> list[list[float]]:
        dims = self.config.embedding["dimensions"]
        model = self.config.embedding["model"]
        out: list[list[float] | None] = [None] * len(texts)
        pending: list[int] = []

        for i, text in enumerate(texts):
            hit = self._cache_get(text, task, model, dims)
            if hit is not None:
                out[i] = hit
                self.usage.embed_cache_hits += 1
            else:
                pending.append(i)

        batch = self.config.embedding["batch_size"]
        for start in range(0, len(pending), batch):
            idxs = pending[start:start + batch]
            resp = self._client.models.embed_content(
                model=model,
                contents=[texts[i] for i in idxs],
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=dims
                ),
            )
            self.usage.embed_calls += 1
            for i, emb in zip(idxs, resp.embeddings):
                vec = [float(v) for v in emb.values]
                out[i] = vec
                self._cache_put(texts[i], task, model, dims, vec)

        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            raise RuntimeError(
                f"embedding provider returned no vector for {len(missing)} of "
                f"{len(texts)} inputs (indices {missing[:5]}...). Aborting "
                "rather than writing a partially-embedded graph."
            )
        return [v for v in out if v is not None]

    # An embedding is a pure function of (text, task, model, dimensions), so it
    # is cacheable forever and re-running ingestion after a code change costs
    # nothing.  The key includes model + dims on purpose: change either and you
    # MUST get fresh vectors, not stale ones from the previous model.
    def _cache_key(self, text: str, task: str, model: str, dims: int) -> Path:
        digest = hashlib.sha256(
            f"{model}|{dims}|{task}|{text}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.pkl"

    def _cache_get(self, text: str, task: str, model: str,
                   dims: int) -> list[float] | None:
        path = self._cache_key(text, task, model, dims)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:  # a corrupt cache entry is not worth crashing over
            path.unlink(missing_ok=True)
            return None

    def _cache_put(self, text: str, task: str, model: str, dims: int,
                   vector: list[float]) -> None:
        with open(self._cache_key(text, task, model, dims), "wb") as fh:
            pickle.dump(vector, fh)


def _is_retryable(exc: Exception) -> bool:
    """Retry transient provider failures only.

    A 400 for a bad model name should fail immediately and loudly - retrying it
    four times just makes the learner wait 15 seconds for the same wrong answer.

    The transport-level entries were added after a real failure: a 118-call
    evaluation run died 40 calls in with `httpx.RemoteProtocolError: Server
    disconnected without sending a response`. That is textbook transient - the
    connection dropped, nothing was wrong with the request - but it matched none
    of the HTTP status markers, so it propagated and threw away every completed
    call in the run.

    The lesson generalises: a retry policy written against status codes misses
    every failure that happens BELOW the status code, and those are the ones a
    long batch job hits.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in (
            # HTTP-level
            "429", "resource_exhausted", "503", "unavailable", "500",
            "internal", "deadline", "timeout",
            # Transport-level: the connection failed, the request never landed
            "remoteprotocolerror", "server disconnected", "connectionerror",
            "connecterror", "connection reset", "connection aborted",
            "readerror", "writeerror", "protocolerror", "incomplete read",
            "ssl", "eof occurred",
        )
    )


def _finish_reason(resp: Any) -> str:
    """Why the model stopped.  'STOP' is normal completion; 'MAX_TOKENS' means
    the output was cut off and anything parsed from it is a fragment."""
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return ""
    reason = getattr(candidates[0], "finish_reason", None)
    return getattr(reason, "name", str(reason or ""))

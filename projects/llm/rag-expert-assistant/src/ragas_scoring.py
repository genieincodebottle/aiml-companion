# ============================================================
# RAGAS SCORING (ragas 0.4 collections API)
# One scorer shared by src/evaluate.py and src/ab_comparison.py
# ============================================================
#
# Why this file exists. Both scripts used the legacy API: metric instances from
# `ragas.metrics` passed to `ragas.evaluate()` with LangChain wrappers. In ragas 0.4 that
# path is deprecated (every import warns) and it is removed in ragas 1.0. The supported
# path is the collections metrics, `ragas.metrics.collections`, each scored with
# `await metric.ascore(...)` by an async judge.
#
# Gemini is wired through Google's own SDK (google-genai, `genai.Client`), wrapped by
# instructor for structured judge output (needs the `instructor[google-genai]` extra). The
# same client serves the embeddings that AnswerRelevancy needs. One GOOGLE_API_KEY, no OpenAI.
#
# Names line up with the legacy columns so results stay comparable in shape:
#   faithfulness       Faithfulness                    (question, answer, contexts)
#   answer_relevancy   AnswerRelevancy                 (question, answer) + embeddings
#   context_precision  ContextPrecisionWithReference   (question, reference, contexts)
#   context_recall     ContextRecall                   (question, contexts, reference)
#
# An LLM judge is not deterministic, so run it more than once before trusting a number.
# ============================================================

import asyncio
import math
import os

from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def ensure_ragas_imports() -> None:
    """Compatibility for the latest ragas with the latest langchain-community.

    ragas 0.4.3 (latest) does `from langchain_community.chat_models.vertexai import
    ChatVertexAI` at import time, and langchain-community 0.4.2 (latest) removed that
    module, so `import ragas` crashes on a fresh install of current versions. ragas only
    puts the class in a list of legacy wrapper types (MULTIPLE_COMPLETION_SUPPORTED) for
    isinstance checks; the collections API used here never touches it. So when, and only
    when, the module is missing, register a placeholder class under that name. Pinning
    langchain-community back to 0.4.1 would also work, but leaves a stale library.

    Self-retiring: once ragas stops importing the module, or langchain-community brings it
    back, the import succeeds and nothing is registered.
    """
    import importlib
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        placeholder = types.ModuleType(name)

        class ChatVertexAI:  # never instantiated: a type ragas can list, nothing more
            """Placeholder for the removed langchain_community ChatVertexAI."""

        placeholder.ChatVertexAI = ChatVertexAI
        sys.modules[name] = placeholder


def build_metrics() -> dict:
    """The four collections metrics, judged by Gemini through google-genai."""
    ensure_ragas_imports()
    from google import genai
    from ragas.embeddings import GoogleEmbeddings
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    # The judge must be ASYNC: collections metrics score with `await metric.ascore(...)`,
    # and a sync judge raises "Cannot use agenerate() with a synchronous client" on every
    # call. ragas 0.4.3's llm_factory wraps a google-genai client with
    # `instructor.from_genai(client)` and never passes use_async=True, so through the
    # factory a Gemini judge is always sync. Build the async instructor client here and
    # hand it to ragas's own InstructorLLM, which takes the async branch for "google".
    import instructor
    from ragas.llms.base import InstructorLLM

    llm = InstructorLLM(
        client=instructor.from_genai(client, use_async=True),
        model=JUDGE_MODEL,
        provider="google",
    )
    assert llm.is_async, "the ragas judge must be async for ascore()"
    # ragas defaults the judge to temperature=0.01 and top_p=0.1. For Gemini 3.5 models
    # Google marks temperature/top_p/top_k as deprecated and ignored, and future models
    # reject them with HTTP 400, so they are not sent at all. ragas builds the Gemini
    # generation_config from model_args (_map_google_params), so removing the keys here
    # leaves only max_output_tokens.
    for key in ("temperature", "top_p", "top_k"):
        llm.model_args.pop(key, None)
    embeddings = GoogleEmbeddings(client=client, model=EMBEDDING_MODEL)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def _with_retry(make_call, attempts: int, delay: float):
    """Retry a judge call on any failure, with exponential backoff (delay, 2x delay, ...).

    A single dropped connection ("connection forcibly closed", a truncated response) used
    to cost the data point: the underlying client tried once. Retrying keeps transient
    network faults out of the scores; a call that fails every attempt is still NaN."""
    for attempt in range(attempts):
        try:
            return await make_call()
        except Exception:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(delay * (2 ** attempt))


async def _score_one(metrics: dict, question: str, answer: str, contexts: list, reference: str,
                     attempts: int = 3, delay: float = 2.0) -> dict:
    calls = {
        "faithfulness": lambda: metrics["faithfulness"].ascore(
            user_input=question, response=answer, retrieved_contexts=contexts),
        "answer_relevancy": lambda: metrics["answer_relevancy"].ascore(
            user_input=question, response=answer),
        "context_precision": lambda: metrics["context_precision"].ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts),
        "context_recall": lambda: metrics["context_recall"].ascore(
            user_input=question, retrieved_contexts=contexts, reference=reference),
    }
    results = await asyncio.gather(
        *(_with_retry(make, attempts, delay) for make in calls.values()), return_exceptions=True)
    row = {}
    for name, result in zip(calls, results):
        value = None if isinstance(result, BaseException) else getattr(result, "value", None)
        # A failed judge call is NaN, never 0: a zero would be scored as a real verdict.
        row[name] = float(value) if isinstance(value, (int, float)) else float("nan")
        if isinstance(result, BaseException):
            row.setdefault("_errors", []).append(f"{name}: {type(result).__name__}: {result}"[:200])
    return row


def score_rows(questions: list, answers: list, contexts: list, references: list,
               concurrency: int = 4, metrics: dict = None, attempts: int = 3,
               retry_delay: float = 2.0) -> list:
    """Score each (question, answer, contexts, reference) row on all four metrics.

    Returns one dict per row: {metric: float, ...} plus `_errors` when a judge call failed
    on every attempt. `concurrency` caps rows in flight, so a run does not burst past the
    API rate limit; each judge call is retried up to `attempts` times on failure.
    """
    if not (len(questions) == len(answers) == len(contexts) == len(references)):
        raise ValueError("questions, answers, contexts and references must be the same length")
    metrics = metrics or build_metrics()

    async def run():
        gate = asyncio.Semaphore(concurrency)

        async def one(args):
            async with gate:
                return await _score_one(metrics, *args, attempts=attempts, delay=retry_delay)

        return await asyncio.gather(*(one(a) for a in zip(questions, answers, contexts, references)))

    return asyncio.run(run())


def mean_scores(rows: list) -> dict:
    """Mean of each metric over the rows that were scored. NaN rows are dropped and
    counted in `_failed`, so a failed judge call can never pull a mean down silently."""
    out = {}
    failed = 0
    for metric in METRICS:
        values = [r[metric] for r in rows if not math.isnan(r.get(metric, float("nan")))]
        failed += len(rows) - len(values)
        out[metric] = sum(values) / len(values) if values else float("nan")
    out["_failed"] = failed
    return out

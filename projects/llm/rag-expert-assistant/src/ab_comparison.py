# ============================================================
# A/B COMPARISON: Naive vs Optimized RAG
# Measure the impact of each optimization -- for real.
# ============================================================
#
# This file used to be a stub that invented its own results. evaluate_rag drew
# scores from random.uniform around a base of 0.65 for "Naive RAG" and 0.88 for
# "Optimized RAG", so the optimized config won by roughly 0.23 every single time
# -- not because it retrieves better, but because 0.88 is a bigger number than
# 0.65. The conclusion was hardcoded and the output was formatted exactly like a
# real result table, which is how it ended up quoted as fact.
#
# It now builds both pipelines against the real corpus, runs the real questions,
# and scores the real answers with RAGAS. It costs API calls and takes a few
# minutes. That is what a measurement costs.
# ============================================================

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---- Define RAG Configurations ----
@dataclass
class RAGConfig:
    name: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int          # candidates pulled from dense search
    rerank_top_n: int         # survivors after reranking
    use_reranking: bool
    grounded_prompt: bool


# Configuration A: Naive (the common starting point)
naive_config = RAGConfig(
    name="Naive RAG",
    chunk_size=1000,
    chunk_overlap=0,
    retriever_k=3,
    rerank_top_n=3,
    use_reranking=False,
    grounded_prompt=False,
)

# Configuration B: Optimized (production-grade)
optimized_config = RAGConfig(
    name="Optimized RAG",
    chunk_size=512,
    chunk_overlap=50,
    retriever_k=20,           # retrieve broadly, then rerank
    rerank_top_n=5,
    use_reranking=True,
    grounded_prompt=True,
)

# NOTE: there is deliberately no use_hybrid_search flag any more. The old config
# carried one, nothing in the repo ever read it, and no BM25 retriever exists
# here. A switch wired to nothing is a claim you cannot cash.


# A generic prompt, of the kind people reach for before hallucination has bitten
# them. No grounding constraint, no citations, no confidence rating.
NAIVE_SYSTEM_PROMPT = """You are a helpful assistant. Use the context below to
answer the question.

Context:
{context}
"""


def _pipeline_for(config: RAGConfig, docs: list, rebuild: bool = True):
    """Build a complete retriever + chain for one configuration.

    Each config gets its OWN Chroma collection. Sharing one would mix
    1000-token and 512-token chunks in a single index, and then neither arm of
    the A/B is the config it claims to be.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.rag_pipeline import (
        chunk_documents, build_vectorstore, build_retriever,
        format_docs_with_sources, SYSTEM_PROMPT,
    )

    slug = config.name.lower().replace(" ", "_")
    persist_dir = "./chroma_db_ab/" + slug
    if rebuild and Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

    chunks = chunk_documents(docs, config.chunk_size, config.chunk_overlap)
    vectorstore = build_vectorstore(chunks, persist_dir=persist_dir)
    retriever = build_retriever(
        vectorstore,
        use_reranking=config.use_reranking,
        top_k=config.retriever_k,
        top_n=config.rerank_top_n,
    )

    system = SYSTEM_PROMPT if config.grounded_prompt else NAIVE_SYSTEM_PROMPT
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), ("human", "{question}")
    ])
    chain = (
        {"context": retriever | format_docs_with_sources,
         "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return retriever, chain


# ---- Evaluation Framework ----
def evaluate_rag(config: RAGConfig, test_questions: list, ground_truth: list,
                 docs: list = None) -> dict:
    """Run test_questions through a real pipeline built from config, and score
    the answers with RAGAS. Returns the mean of each metric.

    Every number this returns came out of a model. None are seeded, hardcoded,
    or drawn from a distribution.
    """
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy, context_precision, context_recall,
    )
    from langchain_google_genai import (
        ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings,
    )

    from src.rag_pipeline import load_documents

    if docs is None:
        docs = load_documents()

    retriever, chain = _pipeline_for(config, docs)

    answers, contexts = [], []
    for i, question in enumerate(test_questions, 1):
        print("  [%s] Q%d/%d: %s..." % (
            config.name, i, len(test_questions), question[:48]))
        retrieved = retriever.invoke(question)
        contexts.append([d.page_content for d in retrieved])
        answers.append(chain.invoke(question).text)

    dataset = Dataset.from_dict({
        "question": list(test_questions),
        "answer": answers,
        "contexts": contexts,
        "ground_truth": list(ground_truth),
    })

    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy,
                 context_precision, context_recall],
        llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite"),
        embeddings=GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"),
    )

    df = result.to_pandas()
    scores = {}
    for metric in ("faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"):
        # RAGAS emits NaN when a judge call fails. Dropping those beats
        # silently scoring a failed judgement as zero.
        if metric in df.columns:
            scores[metric] = float(df[metric].dropna().mean())
        else:
            scores[metric] = float("nan")

    scores["_avg_contexts"] = sum(len(c) for c in contexts) / len(contexts)
    return scores


# ---- Comparison Report ----
def run_ab_comparison(test_questions: list, ground_truth: list,
                      out_path: str = None):
    """Compare naive vs optimized RAG on the same corpus and questions."""
    from src.rag_pipeline import load_documents

    docs = load_documents()

    print("Running Naive RAG evaluation...")
    t0 = time.time()
    naive_scores = evaluate_rag(naive_config, test_questions, ground_truth, docs)
    naive_time = time.time() - t0

    print("Running Optimized RAG evaluation...")
    t0 = time.time()
    opt_scores = evaluate_rag(optimized_config, test_questions, ground_truth, docs)
    opt_time = time.time() - t0

    metrics = ["faithfulness", "answer_relevancy",
               "context_precision", "context_recall"]

    print("\n" + "=" * 62)
    print("%-25s %10s %10s %10s" % ("Metric", "Naive", "Optimized", "Delta"))
    print("=" * 62)
    for metric in metrics:
        naive, opt = naive_scores[metric], opt_scores[metric]
        delta = opt - naive
        sign = "+" if delta > 0 else ""
        status = "BETTER" if delta > 0 else "WORSE" if delta < 0 else "SAME"
        print("  %-23s %9.3f %10.3f %s%9.3f  %s" % (
            metric, naive, opt, sign, delta, status))
    print("-" * 62)
    print("  %-23s %9.1f %10.1f" % (
        "Chunks seen by LLM",
        naive_scores["_avg_contexts"], opt_scores["_avg_contexts"]))
    print("  %-23s %9.1f %10.1f" % ("Wall clock (s)", naive_time, opt_time))
    print("=" * 62)

    improved = sum(1 for m in metrics if opt_scores[m] > naive_scores[m])
    print("\nResult: Optimized RAG improved %d/%d metrics" % (
        improved, len(metrics)))

    if out_path:
        payload = {
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "judge_model": "gemini-3.5-flash-lite",
            "n_questions": len(test_questions),
            "configs": {"naive": asdict(naive_config),
                        "optimized": asdict(optimized_config)},
            "naive": naive_scores,
            "optimized": opt_scores,
            "wall_clock_s": {"naive": naive_time, "optimized": opt_time},
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("\nWrote " + out_path)

    return naive_scores, opt_scores


# ---- Sample Test Data ----
TEST_QUESTIONS = [
    "What is the refund policy for enterprise customers?",
    "How do I reset my API key?",
    "What are the rate limits for the Pro plan?",
    "Does the platform support SSO with Okta?",
    "How do I upgrade from Basic to Pro?",
    "What data retention policies apply to EU users?",
    "Can I export my data in CSV format?",
    "What happens when I exceed the rate limit?",
    "How do I add team members to my organization?",
    "What security certifications does the platform have?",
]

GROUND_TRUTH = [
    "Enterprise customers get full refund within 30 days, prorated after.",
    "Go to Settings > API Keys > Regenerate. Old key is invalidated.",
    "Pro plan: 10,000 req/min (burst 15,000), 1M daily limit.",
    "Yes, supports SSO via SAML 2.0 including Okta, Azure AD, OneLogin.",
    "Go to Billing > Plans > Select Pro. Prorated upgrade, no downtime.",
    "EU user data retained per GDPR: deleted 30 days after account closure.",
    "Yes, Settings > Data > Export supports CSV, JSON, and Parquet formats.",
    "Requests return 429 status. Auto-retry after cooldown. No data loss.",
    "Organization Admin > Team > Invite. Supports email and SSO provisioning.",
    "SOC 2 Type II, ISO 27001, GDPR compliant, HIPAA BAA available.",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure naive vs optimized RAG.")
    ap.add_argument("--out", default="artifacts/results/ab_comparison.json",
                    help="where to write the measured scores as JSON")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N questions (cheaper smoke run)")
    args = ap.parse_args()

    qs, gt = TEST_QUESTIONS, GROUND_TRUTH
    if args.limit:
        qs, gt = qs[:args.limit], gt[:args.limit]
    run_ab_comparison(qs, gt, out_path=args.out)

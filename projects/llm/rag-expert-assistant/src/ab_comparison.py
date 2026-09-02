# ============================================================
# A/B COMPARISON: Naive vs Optimized RAG
# Measure the impact of each optimization
# ============================================================

import hashlib
import random
import time
from dataclasses import dataclass


# ---- Define RAG Configurations ----
@dataclass
class RAGConfig:
    name: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    use_reranking: bool
    use_hybrid_search: bool


# Configuration A: Naive (common starting point)
naive_config = RAGConfig(
    name="Naive RAG",
    chunk_size=1000,
    chunk_overlap=0,
    retriever_k=3,
    use_reranking=False,
    use_hybrid_search=False,
)

# Configuration B: Optimized (production-grade)
optimized_config = RAGConfig(
    name="Optimized RAG",
    chunk_size=512,
    chunk_overlap=50,
    retriever_k=20,  # Retrieve more, then rerank
    use_reranking=True,
    use_hybrid_search=True,
)


# ---- Evaluation Framework ----
def evaluate_rag(config: RAGConfig, test_questions: list, ground_truth: list,
                 allow_mock: bool = False) -> dict:
    """Evaluate a RAG configuration using RAGAS-style metrics. **SKELETON.**

    This function does not measure anything yet, and that used to be far too
    easy to miss. It drew scores from `random.uniform` around a base of 0.65
    for "Naive RAG" and 0.88 for "Optimized RAG" -- so the optimized config won
    by roughly 0.23 every time, not because it retrieves better but because the
    number 0.88 is larger than the number 0.65. The conclusion was hardcoded
    and the output was formatted exactly like a real result table.

    That is the failure mode this whole project is supposed to teach against.
    A/B numbers you did not measure are worse than no numbers: they end up in
    the README, then in a slide, then in someone's decision.

    So the mock is now opt-in. Pass `allow_mock=True` to see the report shape
    while you wire up the real thing; without it this raises. To implement:

      1. Build the pipeline from `config`:
           chunk_documents(docs, config.chunk_size, config.chunk_overlap)
           build_retriever(vs, config.use_reranking, top_k=config.retriever_k)
           if config.use_hybrid_search: add BM25 + reciprocal rank fusion
      2. Run each question through `build_rag_chain(...)`
      3. Score the real answers with RAGAS `evaluate()` (see src/evaluate.py)
      4. Append those scores instead of the placeholders below
    """
    if not allow_mock:
        raise NotImplementedError(
            "evaluate_rag is a skeleton: it returns invented numbers, not "
            "measurements. Wire it to the real pipeline (see the docstring), "
            "or pass allow_mock=True if you only want to see the report "
            "layout. Do not publish anything it prints."
        )
    results = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    for question, truth in zip(test_questions, ground_truth):
        # ================================================
        # TODO: Replace with your actual RAG pipeline call
        # ================================================
        # 1. Build pipeline with config settings:
        #    - splitter = RecursiveCharacterTextSplitter(
        #          chunk_size=config.chunk_size,
        #          chunk_overlap=config.chunk_overlap)
        #    - retriever.search_kwargs["k"] = config.retriever_k
        #    - if config.use_reranking: add CohereRerank
        #    - if config.use_hybrid_search: add BM25 + RRF
        #
        # 2. Run query through pipeline
        # 3. Score with RAGAS evaluate()
        # 4. Append scores to results lists
        #
        # PLACEHOLDER SCORES -- not a measurement. See the docstring.
        #
        # Seeded from a stable hash, not the builtin hash(): Python randomises
        # str hashing per process (PYTHONHASHSEED), so the original version was
        # not even reproducible between runs -- two runs of the same "A/B test"
        # disagreed.
        seed = int(hashlib.sha256(
            f"{question}{config.name}".encode("utf-8")).hexdigest()[:8], 16)
        random.seed(seed)
        base = 0.65 if config.name == "Naive RAG" else 0.88
        for metric in results:
            score = min(1.0, max(0.0, base + random.uniform(-0.1, 0.1)))
            results[metric].append(score)

    # Average across all questions
    return {k: sum(v) / len(v) if v else 0 for k, v in results.items()}


# ---- Comparison Report ----
def run_ab_comparison(test_questions: list, ground_truth: list,
                      allow_mock: bool = False):
    """Compare naive vs optimized RAG configurations.

    `allow_mock=True` prints the report LAYOUT using placeholder scores. It is
    not an experiment, and the banner says so on every run.
    """
    if allow_mock:
        print("!" * 62)
        print("!! PLACEHOLDER RUN -- the numbers below are invented, not")
        print("!! measured. evaluate_rag() is still a skeleton. Nothing here")
        print("!! may be quoted, screenshotted, or put in a README.")
        print("!" * 62)

    print("Running Naive RAG evaluation...")
    t0 = time.time()
    naive_scores = evaluate_rag(naive_config, test_questions, ground_truth,
                                allow_mock=allow_mock)
    naive_time = time.time() - t0

    print("Running Optimized RAG evaluation...")
    t0 = time.time()
    opt_scores = evaluate_rag(optimized_config, test_questions, ground_truth,
                              allow_mock=allow_mock)
    opt_time = time.time() - t0

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Metric':<25} {'Naive':>10} {'Optimized':>10} {'Delta':>10}")
    print("=" * 60)

    for metric in naive_scores:
        naive = naive_scores[metric]
        opt = opt_scores[metric]
        delta = opt - naive
        sign = "+" if delta > 0 else ""
        status = "BETTER" if delta > 0 else "WORSE"
        print(f"  {metric:<23} {naive:>9.3f} {opt:>10.3f} {sign}{delta:>9.3f}  {status}")

    print("-" * 60)
    print(f"  {'Eval time (s)':<23} {naive_time:>9.2f} {opt_time:>10.2f}")
    print("=" * 60)

    # Summary
    improvements = sum(1 for m in naive_scores if opt_scores[m] > naive_scores[m])
    print(f"\nResult: Optimized RAG improved {improvements}/{len(naive_scores)} metrics")

    if allow_mock:
        print("Status: NOT MEASURED -- placeholder scores, see the banner above")
    elif all(opt_scores[m] >= 0.85 for m in opt_scores):
        print("Status: PRODUCTION READY (all metrics >= 0.85)")
    else:
        failing = [m for m in opt_scores if opt_scores[m] < 0.85]
        print(f"Status: NEEDS WORK on: {', '.join(failing)}")

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
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--allow-mock", action="store_true",
        help="print the report layout using PLACEHOLDER scores. Not a "
             "measurement; the output is labelled as such.")
    run_ab_comparison(TEST_QUESTIONS, GROUND_TRUTH,
                      allow_mock=ap.parse_args().allow_mock)
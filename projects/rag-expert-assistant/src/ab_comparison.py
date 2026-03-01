# ============================================================
# A/B COMPARISON: Naive vs Optimized RAG
# Measure the impact of each optimization
# ============================================================

import time
import random
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
def evaluate_rag(config: RAGConfig, test_questions: list, ground_truth: list) -> dict:
    """
    Evaluate a RAG configuration using RAGAS-style metrics.
    Replace the mock scores with your actual RAGAS evaluation.
    """
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
        # Mock scores for demonstration:
        random.seed(hash(question) + hash(config.name))
        base = 0.65 if config.name == "Naive RAG" else 0.88
        for metric in results:
            score = min(1.0, max(0.0, base + random.uniform(-0.1, 0.1)))
            results[metric].append(score)

    # Average across all questions
    return {k: sum(v) / len(v) if v else 0 for k, v in results.items()}


# ---- Comparison Report ----
def run_ab_comparison(test_questions: list, ground_truth: list):
    """Compare naive vs optimized RAG configurations."""
    print("Running Naive RAG evaluation...")
    t0 = time.time()
    naive_scores = evaluate_rag(naive_config, test_questions, ground_truth)
    naive_time = time.time() - t0

    print("Running Optimized RAG evaluation...")
    t0 = time.time()
    opt_scores = evaluate_rag(optimized_config, test_questions, ground_truth)
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
        status = "\u2713" if delta > 0 else "\u2717"
        print(f"  {metric:<23} {naive:>9.3f} {opt:>10.3f} {sign}{delta:>9.3f}  {status}")

    print("-" * 60)
    print(f"  {'Eval time (s)':<23} {naive_time:>9.2f} {opt_time:>10.2f}")
    print("=" * 60)

    # Summary
    improvements = sum(1 for m in naive_scores if opt_scores[m] > naive_scores[m])
    print(f"\nResult: Optimized RAG improved {improvements}/{len(naive_scores)} metrics")

    if all(opt_scores[m] >= 0.85 for m in opt_scores):
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
    run_ab_comparison(TEST_QUESTIONS, GROUND_TRUTH)
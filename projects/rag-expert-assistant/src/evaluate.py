# ============================================================
# RAG EVALUATION WITH RAGAS
# Measure retrieval + generation quality independently
# ============================================================
# pip install ragas langchain-google-genai datasets
# ============================================================

import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

load_dotenv()

# ---- Evaluation Dataset ----
# In production: build 50-100 examples from real user queries
EVAL_DATA = {
    "question": [
        "What is the refund policy for enterprise customers?",
        "How do I reset my API key?",
        "What are the rate limits for the Pro plan?",
        "Does the platform support SSO with Okta?",
    ],
    "answer": [
        "Enterprise customers can request a full refund within 30 days [Source 1]. "
        "After 30 days, prorated refunds are available [Source 2]. Confidence: HIGH",
        "Navigate to Settings > API Keys > Regenerate. Your old key will be "
        "invalidated immediately [Source 1]. Confidence: HIGH",
        "Pro plan allows 10,000 requests/minute with burst capacity to 15,000 "
        "[Source 1]. Daily limit is 1M requests [Source 2]. Confidence: HIGH",
        "Yes, the platform supports SSO via SAML 2.0, including Okta, Azure AD, "
        "and OneLogin [Source 1]. Setup requires admin access. Confidence: MEDIUM",
    ],
    "contexts": [
        [
            "Enterprise refund policy: Full refund within 30 days of purchase.",
            "After 30 days, prorated refunds based on remaining subscription period.",
        ],
        [
            "To reset your API key: Go to Settings > API Keys > Regenerate. "
            "Warning: your previous key will be immediately invalidated.",
        ],
        [
            "Pro plan rate limits: 10,000 req/min, burst to 15,000.",
            "Daily limit: 1,000,000 requests. Overage charged at $0.001/req.",
        ],
        [
            "SSO Support: SAML 2.0 compatible. Tested with Okta, Azure AD, OneLogin.",
            "SSO setup requires Organization Admin role. Contact support for SCIM.",
        ],
    ],
    "ground_truth": [
        "Enterprise customers get full refund within 30 days, prorated after.",
        "Go to Settings > API Keys > Regenerate to reset. Old key is invalidated.",
        "Pro plan: 10,000 req/min (burst 15,000), 1M daily limit.",
        "Yes, supports SSO via SAML 2.0 including Okta, Azure AD, OneLogin.",
    ],
}


def run_evaluation(eval_data: dict = None) -> dict:
    """Run RAGAS evaluation on the provided dataset."""
    if eval_data is None:
        eval_data = EVAL_DATA

    dataset = Dataset.from_dict(eval_data)

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # Display results
    print("=" * 60)
    print("RAG EVALUATION RESULTS (RAGAS)")
    print("=" * 60)

    for metric, score in results.items():
        if isinstance(score, (int, float)):
            status = "PASS" if score >= 0.85 else "NEEDS WORK" if score >= 0.70 else "FAILING"
            bar = "\u2588" * int(score * 20) + "\u2591" * (20 - int(score * 20))
            print(f"  {metric:<25} {bar} {score:.3f}  [{status}]")

    print("\n" + "-" * 60)
    print("INTERPRETATION GUIDE:")
    print("-" * 60)
    print("  Faithfulness < 0.85  -> Tighten grounding in system prompt")
    print("  Answer Rel.  < 0.80  -> Add query expansion / clarification")
    print("  Context Prec < 0.70  -> Improve chunking or add reranking")
    print("  Context Rec  < 0.70  -> Increase top-k, try hybrid search")

    # Per-question breakdown
    print("\n" + "=" * 60)
    print("PER-QUESTION ANALYSIS")
    print("=" * 60)

    df = results.to_pandas()
    for i, row in df.iterrows():
        print(f"\nQ{i+1}: {eval_data['question'][i][:50]}...")
        print(f"  Faith: {row.get('faithfulness', 0):.2f}  "
              f"Relevancy: {row.get('answer_relevancy', 0):.2f}  "
              f"Precision: {row.get('context_precision', 0):.2f}  "
              f"Recall: {row.get('context_recall', 0):.2f}")
        for metric_name in ['faithfulness', 'answer_relevancy',
                            'context_precision', 'context_recall']:
            val = row.get(metric_name, 0)
            if val < 0.80:
                print(f"  \u26a0 {metric_name} is low ({val:.2f}) - investigate!")

    return results


if __name__ == "__main__":
    run_evaluation()
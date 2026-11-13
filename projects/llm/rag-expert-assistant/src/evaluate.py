# ============================================================
# RAG EVALUATION WITH RAGAS
# Measure retrieval + generation quality independently
# ============================================================
# Scoring uses the ragas 0.4 collections API (src/ragas_scoring.py). The legacy
# `ragas.metrics` instances + `ragas.evaluate()` path is deprecated in 0.4 and
# removed in ragas 1.0.
# ============================================================

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.ragas_scoring import JUDGE_MODEL, METRICS, mean_scores, score_rows

load_dotenv()

# The judge and embeddings are Gemini (single GOOGLE_API_KEY); see src/ragas_scoring.py.

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
    """Score the dataset once with RAGAS. Returns {metric: mean, "_failed": n, "rows": [...]}."""
    if eval_data is None:
        eval_data = EVAL_DATA

    rows = score_rows(
        eval_data["question"], eval_data["answer"],
        eval_data["contexts"], eval_data["ground_truth"],
    )
    means = mean_scores(rows)

    print("=" * 60)
    print("RAG EVALUATION RESULTS (RAGAS, judge %s)" % JUDGE_MODEL)
    print("=" * 60)
    for metric in METRICS:
        score = means[metric]
        if score != score:  # NaN: every judge call for this metric failed
            print(f"  {metric:<25} {'-' * 20} n/a    [JUDGE FAILED]")
            continue
        status = "PASS" if score >= 0.85 else "NEEDS WORK" if score >= 0.70 else "FAILING"
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        print(f"  {metric:<25} {bar} {score:.3f}  [{status}]")
    if means["_failed"]:
        print(f"\n  {means['_failed']} judge call(s) failed and were left out of the means.")

    print("\n" + "-" * 60)
    print("INTERPRETATION GUIDE:")
    print("-" * 60)
    print("  Faithfulness < 0.85  -> Tighten grounding in system prompt")
    print("  Answer Rel.  < 0.80  -> Add query expansion / clarification")
    print("  Context Prec < 0.70  -> Improve chunking or add reranking")
    print("  Context Rec  < 0.70  -> Increase top-k, or add a second retriever")

    print("\n" + "=" * 60)
    print("PER-QUESTION ANALYSIS")
    print("=" * 60)
    for i, row in enumerate(rows):
        print(f"\nQ{i+1}: {eval_data['question'][i][:50]}...")
        print("  " + "  ".join(f"{m}: {row[m]:.2f}" for m in METRICS))
        for m in METRICS:
            if row[m] == row[m] and row[m] < 0.80:
                print(f"  WARNING: {m} is low ({row[m]:.2f}) - investigate!")
        for err in row.get("_errors", []):
            print(f"  JUDGE ERROR: {err}")

    return {**means, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score the fixed evaluation set with RAGAS.")
    ap.add_argument("--runs", type=int, default=1,
                    help="score the same data N times; an LLM judge varies between runs")
    ap.add_argument("--out", default=None,
                    help="write every run's means and per-question scores as JSON")
    args = ap.parse_args()

    runs = []
    for n in range(1, args.runs + 1):
        if args.runs > 1:
            print("\n##### RUN %d of %d #####" % (n, args.runs))
        runs.append(run_evaluation())

    if args.runs > 1:
        print("\n" + "=" * 60)
        print("ACROSS %d RUNS (same data, same judge)" % args.runs)
        print("=" * 60)
        for m in METRICS:
            vals = ["%.3f" % r[m] for r in runs]
            print(f"  {m:<25} " + "  ".join(vals))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "judge_model": JUDGE_MODEL,
            "n_questions": len(EVAL_DATA["question"]),
            "runs": runs,
        }, indent=2), encoding="utf-8")
        print("\nWrote " + args.out)
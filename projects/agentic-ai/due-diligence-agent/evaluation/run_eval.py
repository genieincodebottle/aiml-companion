"""
Evaluation framework for the Due Diligence Agent.

Measures:
1. Coverage - breadth of research areas covered
2. Source diversity - number of unique sources cited
3. Factual consistency - cross-agent agreement
4. Actionability - quality of recommendations
5. Latency - end-to-end pipeline execution time

Usage:
    python -m evaluation.run_eval
    python main.py --stage evaluate
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Test companies with expected characteristics
EVAL_COMPANIES = [
    {
        "name": "Tesla",
        "expected_industry": "automotive",
        "expected_risks": ["regulatory", "reputational", "operational"],
        "is_public": True,
        "description": "Well-known public company with extensive financial data",
    },
    {
        "name": "Stripe",
        "expected_industry": "fintech",
        "expected_risks": ["regulatory", "competitive"],
        "is_public": False,
        "description": "Large private company - tests handling of limited financial data",
    },
    {
        "name": "Anthropic",
        "expected_industry": "artificial intelligence",
        "expected_risks": ["regulatory", "competitive", "technology"],
        "is_public": False,
        "description": "AI company - tests coverage of emerging industry risks",
    },
]


def evaluate_coverage(result: dict) -> dict:
    """Score the breadth of research areas covered."""
    areas = {
        "financial": len(result.get("financial_findings", [])) > 0,
        "news": len(result.get("news_findings", [])) > 0,
        "competitive": len(result.get("competitive_findings", [])) > 0,
        "risk": len(result.get("risk_findings", [])) > 0,
        "fact_check": len(result.get("fact_check_results", [])) > 0,
    }

    covered = sum(areas.values())
    total = len(areas)

    return {
        "metric": "coverage",
        "score": covered / total,
        "covered": covered,
        "total": total,
        "details": areas,
    }


def evaluate_source_diversity(result: dict) -> dict:
    """Score the number of unique sources cited."""
    all_sources = set()

    for key in ["financial_findings", "news_findings", "competitive_findings", "risk_findings"]:
        for finding in result.get(key, []):
            for source in finding.get("sources", []):
                if source:
                    all_sources.add(source)

    count = len(all_sources)
    # Score: 0-5 sources = low, 5-15 = moderate, 15+ = high
    score = min(count / 15, 1.0)

    return {
        "metric": "source_diversity",
        "score": round(score, 3),
        "unique_sources": count,
    }


def evaluate_factual_consistency(result: dict) -> dict:
    """Score cross-agent agreement based on fact-check results."""
    fc_results = result.get("fact_check_results", [{}])
    fc = fc_results[0] if fc_results else {}

    total_checked = fc.get("total_checked", 0)
    verified = fc.get("verified", 0)
    contradicted = fc.get("contradicted", 0)

    if total_checked == 0:
        return {"metric": "factual_consistency", "score": 0.5, "note": "No claims checked"}

    consistency = verified / total_checked if total_checked > 0 else 0
    contradiction_penalty = contradicted / total_checked if total_checked > 0 else 0

    score = max(0, consistency - contradiction_penalty * 0.5)

    return {
        "metric": "factual_consistency",
        "score": round(score, 3),
        "verified": verified,
        "contradicted": contradicted,
        "total_checked": total_checked,
    }


def evaluate_actionability(result: dict) -> dict:
    """Score the quality of actionable recommendations."""
    report = result.get("final_report", "")

    has_verdict = bool(result.get("executive_summary"))
    has_risk_rating = result.get("overall_risk_rating", "unknown") != "unknown"
    has_recommendations = "next steps" in report.lower() or "action items" in report.lower() or "recommend" in report.lower()
    has_uncertainties = "uncertain" in report.lower() or "data gap" in report.lower() or "insufficient" in report.lower()

    checks = [has_verdict, has_risk_rating, has_recommendations, has_uncertainties]
    score = sum(checks) / len(checks)

    return {
        "metric": "actionability",
        "score": round(score, 3),
        "details": {
            "has_verdict": has_verdict,
            "has_risk_rating": has_risk_rating,
            "has_recommendations": has_recommendations,
            "acknowledges_uncertainties": has_uncertainties,
        },
    }


def evaluate_single(company_info: dict) -> dict:
    """Run full evaluation on a single company."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.agents.graph import run_pipeline

    name = company_info["name"]
    logger.info("Evaluating: %s", name)

    start = time.time()
    result = run_pipeline(company_name=name, depth="quick")
    duration = time.time() - start

    evals = {
        "company": name,
        "status": result.get("status", "unknown"),
        "duration_seconds": round(duration, 2),
        "risk_rating": result.get("overall_risk_rating", "unknown"),
        "confidence": result.get("overall_confidence", 0),
        "errors": result.get("errors", []),
        "metrics": {
            "coverage": evaluate_coverage(result),
            "source_diversity": evaluate_source_diversity(result),
            "factual_consistency": evaluate_factual_consistency(result),
            "actionability": evaluate_actionability(result),
        },
    }

    # Composite score (equal weights)
    metric_scores = [m["score"] for m in evals["metrics"].values()]
    evals["composite_score"] = round(sum(metric_scores) / len(metric_scores), 3) if metric_scores else 0

    return evals


def run_evaluation(companies: list[dict] | None = None):
    """Run the full evaluation suite."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    companies = companies or EVAL_COMPANIES
    results = []

    print("\n" + "=" * 70)
    print("  Due Diligence Agent - Evaluation Suite")
    print("=" * 70)

    for company in companies:
        try:
            eval_result = evaluate_single(company)
            results.append(eval_result)

            print(f"\n  {company['name']}: score={eval_result['composite_score']:.2f}, "
                  f"risk={eval_result['risk_rating']}, duration={eval_result['duration_seconds']:.1f}s")

            for name, metric in eval_result["metrics"].items():
                print(f"    {name}: {metric['score']:.2f}")

        except Exception as e:
            logger.error("Evaluation failed for %s: %s", company["name"], e)
            results.append({"company": company["name"], "error": str(e), "composite_score": 0})

    # Summary
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)

    scores = [r["composite_score"] for r in results if "composite_score" in r]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  Average composite score: {avg:.2f}")
        print(f"  Companies evaluated: {len(results)}")
        print(f"  Failures: {sum(1 for r in results if 'error' in r)}")

    # Save results
    output_path = PROJECT_ROOT / "artifacts" / "reports" / "evaluation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_evaluation()

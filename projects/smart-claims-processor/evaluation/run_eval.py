"""
Batch Evaluation Runner

Processes all sample claims and computes aggregate quality metrics.
Use this to regression-test pipeline quality after changes.

Usage:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --claims-dir data/sample_claims
    python evaluation/run_eval.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.graph import process_claim
from src.models.state import ClaimInput


def run_batch_evaluation(claims_dir: str = "data/sample_claims") -> dict:
    """Run all claims and collect evaluation metrics."""
    claims_path = Path(claims_dir)
    claims = list(claims_path.glob("*.json"))

    if not claims:
        print(f"No claims found in {claims_dir}")
        return {}

    results = []
    for claim_file in claims:
        if claim_file.stem.startswith("_"):
            continue  # Skip files starting with underscore
        print(f"Evaluating: {claim_file.name}")
        try:
            with open(claim_file) as f:
                data = json.load(f)
            # Remove test-only fields
            data.pop("_test_note", None)
            claim = ClaimInput(**data)
            final_state = process_claim(claim)
            results.append({
                "claim_file": claim_file.name,
                "claim_id": data["claim_id"],
                "decision": str(final_state.get("final_decision", "unknown")),
                "amount_usd": final_state.get("final_amount_usd", 0),
                "eval_score": final_state.get("evaluation_output").overall_score if final_state.get("evaluation_output") else None,
                "eval_passed": final_state.get("evaluation_passed"),
                "hitl_required": final_state.get("hitl_required", False),
                "fraud_score": final_state.get("fraud_output").fraud_score if final_state.get("fraud_output") else None,
                "agent_calls": final_state.get("agent_call_count", 0),
                "cost_usd": final_state.get("total_cost_usd", 0),
                "errors": final_state.get("error_log", []),
            })
        except Exception as e:
            results.append({
                "claim_file": claim_file.name,
                "error": str(e),
            })

    # Aggregate stats
    successful = [r for r in results if "error" not in r]
    eval_scores = [r["eval_score"] for r in successful if r["eval_score"] is not None]
    total_cost = sum(r.get("cost_usd", 0) for r in successful)

    summary = {
        "total_claims": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "hitl_triggered": sum(1 for r in successful if r.get("hitl_required")),
        "avg_eval_score": sum(eval_scores) / len(eval_scores) if eval_scores else 0,
        "all_evals_passed": all(r.get("eval_passed", False) for r in successful),
        "total_cost_usd": round(total_cost, 4),
        "avg_agent_calls": sum(r.get("agent_calls", 0) for r in successful) / len(successful) if successful else 0,
    }

    print("\n" + "=" * 50)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 50)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("\nPer-claim results:")
    for r in results:
        status = "ERROR" if "error" in r else ("PASS" if r.get("eval_passed") else "REVIEW")
        print(f"  [{status}] {r['claim_file']}: {r.get('decision', 'N/A')} | ${r.get('amount_usd', 0):,.0f} | eval={r.get('eval_score', 'N/A')}")

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims-dir", default="data/sample_claims")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    run_batch_evaluation(args.claims_dir)

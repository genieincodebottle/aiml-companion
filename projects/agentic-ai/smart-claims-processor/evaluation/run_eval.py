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
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.graph import process_claim
from src.config import set_country_override
from src.models.state import ClaimInput


def _country_for(claim_file: Path) -> str:
    """Sample files are written for one country (amounts in USD or INR,
    country-specific claim types). Scoring an India claim under the US
    profile compares rupees to dollar baselines."""
    return "india" if "india" in claim_file.stem.lower() else "us"


def _load_claims(claim_file: Path) -> list[tuple[str, dict]]:
    """A sample file is either one claim, or a dict of named claims
    (test_all_paths_*.json) whose keys starting with "_" are documentation."""
    with open(claim_file, encoding="utf-8") as f:
        data = json.load(f)
    if "incident_type" in data:
        items = [(claim_file.name, data)]
    else:
        items = [
            (f"{claim_file.name}:{name}", body)
            for name, body in data.items()
            if not name.startswith("_") and isinstance(body, dict)
        ]
    claims = []
    for label, body in items:
        claim = {k: v for k, v in body.items() if not k.startswith("_")}
        claim.setdefault("claim_id", f"EVAL-{uuid.uuid4().hex[:8].upper()}")
        for key, default in _CLAIM_DEFAULTS.items():
            claim.setdefault(key, default)
        claims.append((label, claim))
    return claims


_CLAIM_DEFAULTS = {
    "claimant_name": "Eval Claimant",
    "claimant_email": "eval@example.com",
    "claimant_phone": "000-000-0000",
    "claimant_dob": "1990-01-01",
    "incident_location": "",
    "police_report_number": None,
    "vehicle_year": None,
    "vehicle_make": None,
    "vehicle_model": None,
    "documents": [],
    "is_appeal": False,
    "original_claim_id": None,
}


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value or "unknown")


def run_batch_evaluation(claims_dir: str = "data/sample_claims", country: str | None = None) -> dict:
    """Run all claims and collect evaluation metrics.

    process_claim() returns {"paused": bool, "state": {...}}. A paused claim
    is waiting for a human reviewer: it has no final decision yet and is
    reported as `paused_for_review`, not as a pass or a failure.
    """
    claims_path = Path(claims_dir)
    claim_files = sorted(claims_path.glob("*.json"))

    if not claim_files:
        print(f"No claims found in {claims_dir}")
        return {}

    results = []
    for claim_file in claim_files:
        if claim_file.stem.startswith("_"):
            continue  # Skip files starting with underscore
        try:
            claims = _load_claims(claim_file)
        except Exception as e:
            results.append({"claim_file": claim_file.name, "error": f"unreadable: {e}"})
            continue
        set_country_override(country or _country_for(claim_file))
        for label, data in claims:
            print(f"Evaluating: {label}")
            try:
                claim = ClaimInput(**data)
                result = process_claim(claim)
                final_state = result["state"]
                paused = bool(result.get("paused"))
                evaluation = final_state.get("evaluation_output")
                fraud = final_state.get("fraud_output")
                results.append({
                    "claim_file": label,
                    "claim_id": data["claim_id"],
                    "paused": paused,
                    "decision": "paused_for_review" if paused else _enum_value(final_state.get("final_decision")),
                    "amount_usd": final_state.get("final_amount_usd") or 0,
                    "eval_score": evaluation.overall_score if evaluation else None,
                    "eval_passed": final_state.get("evaluation_passed"),
                    "hitl_required": paused or bool(final_state.get("hitl_required", False)),
                    "fraud_score": fraud.fraud_score if fraud else None,
                    "agent_calls": final_state.get("agent_call_count", 0),
                    "cost_usd": final_state.get("total_cost_usd", 0),
                    "path": [e.get("agent") for e in final_state.get("pipeline_trace", []) if isinstance(e, dict)],
                    "errors": final_state.get("error_log", []),
                })
            except Exception as e:
                results.append({
                    "claim_file": label,
                    "error": str(e),
                })

    set_country_override(None)

    # Aggregate stats
    successful = [r for r in results if "error" not in r]
    # Only claims the judge actually scored. eval_passed is None when the
    # sampler skipped the claim -- that is not a pass and not a failure.
    evaluated = [r for r in successful if r.get("eval_passed") is not None]
    eval_scores = [r["eval_score"] for r in successful if r["eval_score"] is not None]
    total_cost = sum(r.get("cost_usd", 0) for r in successful)

    summary = {
        "total_claims": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "hitl_triggered": sum(1 for r in successful if r.get("hitl_required")),
        "paused_for_review": sum(1 for r in successful if r.get("paused")),
        # Denominators must match. `avg_eval_score` was already computed over
        # the evaluated subset while `all_evals_passed` ran over every claim,
        # counting sampled-out ones as passes because they carried
        # evaluation_passed=True. At the configured sample rate of 0.10, that
        # headline was ~90% claims nobody scored.
        "evaluated": len(evaluated),
        "skipped_by_sampling": len(successful) - len(evaluated),
        "avg_eval_score": sum(eval_scores) / len(eval_scores) if eval_scores else 0,
        "eval_pass_rate": (
            sum(1 for r in evaluated if r["eval_passed"]) / len(evaluated)
            if evaluated else None
        ),
        "all_evaluated_claims_passed": (
            all(r["eval_passed"] for r in evaluated) if evaluated else None
        ),
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
        if "error" in r:
            status = "ERROR"
        elif r.get("paused"):
            status = "PAUSED"
        elif r.get("eval_passed") is None:
            status = "UNJUDGED"
        else:
            status = "PASS" if r["eval_passed"] else "FAIL"
        print(f"  [{status}] {r['claim_file']}: {r.get('decision', 'N/A')} | ${r.get('amount_usd', 0):,.0f} | eval={r.get('eval_score', 'N/A')}")

    return {"summary": summary, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims-dir", default="data/sample_claims")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--country", choices=["us", "india"], default=None,
                        help="Force one country profile (default: from the file name)")
    args = parser.parse_args()
    run_batch_evaluation(args.claims_dir, country=args.country)

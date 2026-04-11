"""
Smart Claims Processor - CLI Entry Point

Usage:
    python main.py --claim data/sample_claims/auto_accident.json
    python main.py --claim-id CLM-2024-001234 --policy POL-AUTO-789456 \
                   --type auto_collision --amount 8500 \
                   --description "Rear-ended at intersection"
    python main.py --batch data/sample_claims/
    python main.py --demo            # Run all sample claims
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.agents.graph import process_claim
from src.models.state import ClaimInput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_claim_from_file(path: str) -> ClaimInput:
    with open(path) as f:
        data = json.load(f)
    return ClaimInput(**data)


def load_claim_from_args(args) -> ClaimInput:
    import uuid
    return ClaimInput(
        claim_id=args.claim_id or f"CLM-{uuid.uuid4().hex[:8].upper()}",
        policy_number=args.policy,
        claimant_name="[Provided via CLI]",
        claimant_email="[Provided via CLI]",
        claimant_phone="[Provided via CLI]",
        claimant_dob="[Provided via CLI]",
        incident_date=args.date or "2024-11-15",
        incident_type=args.type,
        incident_description=args.description,
        incident_location=args.location or "Unknown",
        police_report_number=args.police_report,
        estimated_amount=float(args.amount),
        vehicle_year=args.vehicle_year,
        vehicle_make=args.vehicle_make,
        vehicle_model=args.vehicle_model,
        documents=args.documents.split(",") if args.documents else [],
        is_appeal=False,
        original_claim_id=None,
    )


def print_result(final_state: dict, verbose: bool = False):
    """Print formatted pipeline results."""
    claim_id = final_state.get("claim", {}).get("claim_id", "UNKNOWN")
    decision = final_state.get("final_decision")
    decision_str = decision.value if hasattr(decision, "value") else str(decision or "unknown")
    amount = final_state.get("final_amount_usd") or 0
    fraud = final_state.get("fraud_output")
    evaluation = final_state.get("evaluation_output")
    hitl = final_state.get("hitl_required", False)
    comm = final_state.get("communication_output")

    print("\n" + "=" * 60)
    print(f"  SMART CLAIMS PROCESSOR - RESULT")
    print("=" * 60)
    print(f"  Claim ID:    {claim_id}")
    print(f"  Decision:    {decision_str.upper().replace('_', ' ')}")
    print(f"  Settlement:  ${amount:,.2f}")
    if fraud:
        print(f"  Fraud Risk:  {fraud.fraud_risk_level.value.upper()} (score: {fraud.fraud_score:.2f})")
    if evaluation:
        print(f"  Eval Score:  {evaluation.overall_score:.2f}/1.0 ({'PASS' if evaluation.passed else 'FAIL'})")
    print(f"  HITL:        {'YES - ticket: ' + (final_state.get('hitl_ticket_id') or 'pending') if hitl else 'No'}")
    print(f"  Agents Used: {final_state.get('agent_call_count') or 0}")
    print(f"  Cost:        ${(final_state.get('total_cost_usd') or 0):.4f}")
    print(f"  Tokens:      {(final_state.get('total_tokens_used') or 0):,}")

    if comm:
        print("\n" + "-" * 60)
        print("  CLAIMANT NOTIFICATION:")
        print("-" * 60)
        print(f"  Subject: {comm.subject}")
        print()
        for line in comm.message.split("\n"):
            print(f"  {line}")

    if verbose:
        print("\n" + "-" * 60)
        print("  PIPELINE TRACE:")
        print("-" * 60)
        for entry in final_state.get("pipeline_trace", []):
            print(f"  {entry}")

    errors = final_state.get("error_log", [])
    if errors:
        print("\n  ERRORS:")
        for err in errors:
            print(f"  ! {err}")

    print("=" * 60 + "\n")


def run_demo():
    """Run all sample claims to demonstrate the full pipeline."""
    sample_dir = Path("data/sample_claims")
    if not sample_dir.exists():
        print("Sample claims directory not found. Run from project root.")
        return

    claims = list(sample_dir.glob("*.json"))
    if not claims:
        print("No sample claims found in data/sample_claims/")
        return

    print(f"\nRunning {len(claims)} sample claims...\n")
    for claim_file in claims:
        try:
            claim = load_claim_from_file(str(claim_file))
            print(f"Processing: {claim_file.name}")
            result = process_claim(claim)
            print_result(result)
        except Exception as e:
            print(f"Error processing {claim_file.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Smart Claims Processor")
    parser.add_argument("--claim", help="Path to JSON claim file")
    parser.add_argument("--demo", action="store_true", help="Run all sample claims")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show pipeline trace")

    # Direct claim args
    parser.add_argument("--claim-id")
    parser.add_argument("--policy")
    parser.add_argument("--type", default="auto_collision")
    parser.add_argument("--amount", type=float)
    parser.add_argument("--description", default="")
    parser.add_argument("--date")
    parser.add_argument("--location")
    parser.add_argument("--police-report")
    parser.add_argument("--vehicle-year", type=int)
    parser.add_argument("--vehicle-make")
    parser.add_argument("--vehicle-model")
    parser.add_argument("--documents", help="Comma-separated document names")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.claim:
        claim = load_claim_from_file(args.claim)
        result = process_claim(claim)
        print_result(result, verbose=args.verbose)
    elif args.policy and args.amount:
        claim = load_claim_from_args(args)
        result = process_claim(claim)
        print_result(result, verbose=args.verbose)
    else:
        parser.print_help()
        print("\nQuick start: python main.py --demo")


if __name__ == "__main__":
    main()

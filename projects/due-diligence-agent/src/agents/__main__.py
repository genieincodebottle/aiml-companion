"""
CLI entry point for the due diligence pipeline.

Usage:
    python -m src.agents "Tesla"
    python -m src.agents "Stripe" --depth deep
    python -m src.agents "OpenAI" --depth quick --output reports/openai.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.graph import run_pipeline


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Due Diligence Analyst",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.agents "Tesla"
  python -m src.agents "Stripe" --depth deep
  python -m src.agents "OpenAI" --query "Focus on AI safety risks and competition"
  python -m src.agents "Databricks" --output reports/databricks.md --verbose
        """,
    )
    parser.add_argument("company", help="Company or entity name to analyze")
    parser.add_argument("--query", "-q", default="", help="Specific research focus/questions")
    parser.add_argument("--depth", "-d", choices=["quick", "standard", "deep"], default="standard", help="Analysis depth")
    parser.add_argument("--output", "-o", default=None, help="Save report to file (markdown)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")

    print(f"\n{'='*60}")
    print(f"  Multi-Agent Due Diligence Analysis")
    print(f"  Company: {args.company}")
    print(f"  Depth: {args.depth}")
    if args.query:
        print(f"  Focus: {args.query}")
    print(f"{'='*60}\n")

    result = run_pipeline(
        company_name=args.company,
        query=args.query,
        depth=args.depth,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Status: {result.get('status', 'unknown')}")
    print(f"  Risk Rating: {result.get('overall_risk_rating', 'unknown')}")
    print(f"  Confidence: {result.get('overall_confidence', 0):.0%}")
    print(f"  Duration: {result.get('_execution_time_seconds', 0):.1f}s")

    token_summary = result.get("_token_summary", {})
    print(f"  Tokens Used: {token_summary.get('total_tokens', 0):,}")
    print(f"  Estimated Cost: ${token_summary.get('estimated_cost_usd', 0):.4f}")

    errors = result.get("errors", [])
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    - {e}")

    print(f"{'='*60}\n")

    # Print executive summary
    exec_summary = result.get("executive_summary", "")
    if exec_summary:
        print(f"VERDICT: {exec_summary}\n")

    # Save report
    report = result.get("final_report", "")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report saved to: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()

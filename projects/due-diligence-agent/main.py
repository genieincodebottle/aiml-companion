"""
Due Diligence Agent - CLI Entry Point

Usage:
    python main.py                                    # Interactive prompt
    python main.py --company "Tesla"                  # Direct analysis
    python main.py --company "Stripe" --depth deep    # Deep analysis
    python main.py --stage ui                         # Launch Streamlit UI
    python main.py --stage evaluate                   # Run evaluation suite

Stages:
    analyze   - Run due diligence analysis (default)
    ui        - Launch Streamlit web interface
    evaluate  - Run evaluation framework
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file if present (so users can just copy .env.example to .env)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv not installed, rely on shell environment variables


def _check_api_key():
    """Validate API key is set before running the pipeline."""
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        print("\nError: No API key found.")
        print("")
        print("Option 1 - Create a .env file:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your Google API key (free at https://aistudio.google.com/apikey)")
        print("")
        print("Option 2 - Set environment variable:")
        print("  Linux/Mac:  export GOOGLE_API_KEY=your_key_here")
        print("  Windows:    set GOOGLE_API_KEY=your_key_here")
        print("  PowerShell: $env:GOOGLE_API_KEY='your_key_here'")
        sys.exit(1)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def run_analyze(args):
    """Run the due diligence pipeline."""
    _check_api_key()
    from src.agents.graph import run_pipeline

    company = args.company
    if not company:
        company = input("Enter company name: ").strip()
        if not company:
            print("No company provided. Exiting.")
            return

    print(f"\nAnalyzing: {company} (depth: {args.depth})")
    print("=" * 60)

    result = run_pipeline(
        company_name=company,
        query=args.query or "",
        depth=args.depth,
    )

    # Print summary
    print(f"\nStatus: {result.get('status', 'unknown')}")
    print(f"Risk Rating: {result.get('overall_risk_rating', 'unknown')}")
    print(f"Confidence: {result.get('overall_confidence', 0):.0%}")
    print(f"Duration: {result.get('_execution_time_seconds', 0):.1f}s")

    # Save report
    report = result.get("final_report", "")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {output_path}")
    else:
        print("\n" + report)


def run_ui(args):
    """Launch Streamlit UI."""
    import subprocess
    app_path = PROJECT_ROOT / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


def run_evaluate(args):
    """Run the evaluation framework."""
    _check_api_key()
    from evaluation.run_eval import run_evaluation
    run_evaluation()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Due Diligence Analyst",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", choices=["analyze", "ui", "evaluate"], default="analyze",
                        help="Pipeline stage to run")
    parser.add_argument("--company", "-c", default=None, help="Company name to analyze")
    parser.add_argument("--query", "-q", default="", help="Specific research focus")
    parser.add_argument("--depth", "-d", choices=["quick", "standard", "deep"], default="standard",
                        help="Analysis depth")
    parser.add_argument("--output", "-o", default=None, help="Output file path for report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    stages = {
        "analyze": run_analyze,
        "ui": run_ui,
        "evaluate": run_evaluate,
    }

    stages[args.stage](args)


if __name__ == "__main__":
    main()

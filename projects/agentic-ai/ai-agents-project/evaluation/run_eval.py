# ============================================
# Agent System Evaluation Framework
# ============================================
# uv pip install pandas google-genai

import json
import time

from evaluation.judge_prompt import COMBINED_PROMPT
from dotenv import load_dotenv

load_dotenv()

_client = None
MODEL = "gemini-3.6-flash"

# Cost
# ----
# There is no honest single "price per token" to hardcode here. On the free tier
# every call bills $0; on the paid tier input and output are priced differently
# and the published rates change. The old constant `PRICE_PER_TOKEN = 0` made the
# arithmetic run without complaint and produce $0.0000 for both arms and a "Cost
# multiplier: 0.0x" -- a number that cannot be right for a pipeline that makes
# three LLM calls where the baseline makes one.
#
# So: TOKENS are the measurement and are always reported. Dollars are a
# conversion, and only appear if you supply today's rates for your tier from
# https://ai.google.dev/pricing. Leave these as None and the report simply omits
# the currency column rather than inventing one.
PRICE_PER_1M_INPUT = None
PRICE_PER_1M_OUTPUT = None


def _get_client():
    """Lazy-init Gemini client. Only created when an API call is actually needed."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client()
    return _client

# === 1. Test Set (10 research questions) ===
TEST_QUESTIONS = [
    {"query": "What is retrieval-augmented generation?", "type": "factual", "difficulty": "easy"},
    {"query": "Compare transformer and mamba architectures", "type": "synthesis", "difficulty": "hard"},
    {"query": "Latest breakthroughs in protein folding AI", "type": "factual", "difficulty": "medium"},
    {"query": "How do AI agents differ from chatbots?", "type": "factual", "difficulty": "easy"},
    {"query": "Trade-offs between fine-tuning and RAG", "type": "synthesis", "difficulty": "hard"},
    {"query": "What is constitutional AI?", "type": "factual", "difficulty": "medium"},
    {"query": "How does RLHF work in language models?", "type": "factual", "difficulty": "medium"},
    {"query": "Compare LangGraph, CrewAI, and AutoGen frameworks", "type": "synthesis", "difficulty": "hard"},
    {"query": "What are mixture-of-experts models?", "type": "factual", "difficulty": "medium"},
    {"query": "Evaluate the impact of scaling laws on LLM development", "type": "synthesis", "difficulty": "hard"},
]


def _cost(input_tokens: int, output_tokens: int):
    """Dollar cost, or None when no price has been supplied. Never a guess."""
    if PRICE_PER_1M_INPUT is None or PRICE_PER_1M_OUTPUT is None:
        return None
    return (input_tokens * PRICE_PER_1M_INPUT
            + output_tokens * PRICE_PER_1M_OUTPUT) / 1_000_000


def _generate(prompt: str, max_tokens: int = 8000, json_output: bool = False):
    """Call Gemini and return (text, (input, output, total)) with real usage.

    On `max_tokens`
    ---------------
    These caps used to be 1000-2000, which was fine for a non-thinking model and
    is not fine for this one. Measured on gemini-3.6-flash, the single-agent
    prompt below spends 942 tokens on reasoning before it emits a word:

        max_output_tokens=2000 -> finish_reason MAX_TOKENS, report cut off
        max_output_tokens=8000 -> finish_reason STOP, report complete

    Reasoning tokens count against the same budget as visible output, so the old
    harness was scoring truncated reports. The multi-agent arm suffered worst,
    because each truncated stage fed the next, and the judge duly returned
    completeness 0.00. That is a measurement of the token cap, not of the
    architecture -- exactly the kind of number that ends up in a README as
    "multi-agent underperforms".

    Truncation is now an error rather than a silently lower score. A benchmark
    that quietly scores incomplete answers is measuring its own configuration.
    """
    config = {"max_output_tokens": max_tokens, "temperature": 0}
    if json_output:
        config["response_mime_type"] = "application/json"
    response = _get_client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )

    finish = getattr(response.candidates[0], "finish_reason", None)
    if finish is not None and "MAX_TOKENS" in str(finish):
        raise RuntimeError(
            f"Response hit max_output_tokens={max_tokens} and was truncated "
            f"(thinking tokens count against this budget). Raise the cap rather "
            f"than scoring a cut-off answer."
        )

    usage = response.usage_metadata
    return response.text, (
        usage.prompt_token_count or 0,
        (usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", 0) or 0),
        usage.total_token_count or 0,
    )


# === 2. Single-Agent Baseline ===
def single_agent_research(query: str) -> dict:
    """One LLM call to research and write about a topic."""
    start = time.time()
    text, (tin, tout, total) = _generate(
        f"Research this topic and write a brief report with citations: {query}",
        max_tokens=8000,
    )
    return {
        "report": text,
        "input_tokens": tin, "output_tokens": tout, "tokens": total,
        "calls": 1,
        "latency": time.time() - start,
        "cost": _cost(tin, tout),
    }


# === 3. Multi-Agent Research (Specialist Delegation) ===
def multi_agent_research(query: str) -> dict:
    """3-agent pipeline: researcher -> analyst -> writer."""
    start = time.time()
    tin = tout = total = 0

    # Agent 1: Researcher - gather information
    findings, u = _generate(
        f"You are a research specialist. Find key facts, data points, "
        f"and cite your sources. Return structured findings.\n\n"
        f"Research this topic thoroughly. List 5-8 key findings "
        f"with source references:\n{query}",
        max_tokens=8000,
    )
    tin, tout, total = tin + u[0], tout + u[1], total + u[2]

    # Agent 2: Analyst - extract claims and assess confidence
    analysis, u = _generate(
        f"You are a research analyst. Evaluate findings for accuracy "
        f"and rank claims by confidence (high/medium/low).\n\n"
        f"Analyze these research findings. Identify the strongest "
        f"claims and flag any that seem unsupported:\n{findings}",
        max_tokens=8000,
    )
    tin, tout, total = tin + u[0], tout + u[1], total + u[2]

    # Agent 3: Writer - produce structured report
    report, u = _generate(
        f"You are a technical writer. Write clear, well-structured "
        f"reports. Only include claims from the provided analysis.\n\n"
        f"Write a structured research report with these sections: "
        f"Introduction, Key Findings, Analysis, Conclusion.\n\n"
        f"Use ONLY these analyzed findings:\n{analysis}",
        max_tokens=8000,
    )
    tin, tout, total = tin + u[0], tout + u[1], total + u[2]

    return {
        "report": report,
        "input_tokens": tin, "output_tokens": tout, "tokens": total,
        "calls": 3,
        "latency": time.time() - start,
        "cost": _cost(tin, tout),
    }


# === 4. Evaluation Metrics (LLM-as-Judge) ===
def evaluate_report(query: str, report: str) -> dict:
    """Score a report on accuracy, completeness, and citation quality.

    The prompt comes from judge_prompt.COMBINED_PROMPT rather than a copy
    inlined here. There used to be two versions -- the module the docs and the
    tests pointed at, and the one this function actually sent -- so editing the
    rubric in the obvious place changed nothing about the scores.

    Read the numbers with the judge's limits in mind: it is the same model
    family that wrote both reports (self-preference is well documented for LLM
    judges), the scale is four integers wide so small differences round away,
    and one judge pass per report gives no read on the judge's own variance.
    """
    text, _ = _generate(
        COMBINED_PROMPT.format(query=query, report=report[:2000]),
        json_output=True,
    )
    try:
        scores = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge did not return JSON: {text[:200]!r}") from e
    missing = {"accuracy", "completeness", "citations"} - set(scores)
    if missing:
        raise ValueError(f"Judge omitted {sorted(missing)}: {scores}")
    return {k: float(scores[k]) for k in ("accuracy", "completeness", "citations")}


# === 5. Run Comparison ===
def run_comparison(num_questions: int = 5):
    """Run a head-to-head comparison on the test questions.

    Paired design: both arms answer the same questions, so the per-question
    difference is the unit of analysis and question difficulty cancels out.
    """
    import pandas as pd

    results = {"single": [], "multi": []}
    keys = ("tokens", "input_tokens", "output_tokens", "calls", "latency", "cost")

    for test in TEST_QUESTIONS[:num_questions]:
        print(f"Evaluating: {test['query'][:50]}...")

        single = single_agent_research(test["query"])
        single_scores = evaluate_report(test["query"], single["report"])
        results["single"].append({**single_scores, **{k: single[k] for k in keys}})

        multi = multi_agent_research(test["query"])
        multi_scores = evaluate_report(test["query"], multi["report"])
        results["multi"].append({**multi_scores, **{k: multi[k] for k in keys}})

    # === 6. Summary Report ===
    def print_summary(label: str, data: list) -> "pd.DataFrame":
        df = pd.DataFrame(data)
        print(f"\n=== {label} ===")
        for metric in ("accuracy", "completeness", "citations"):
            print(f"{metric.capitalize():13s} {df[metric].mean():.2f}/3  "
                  f"(SD {df[metric].std(ddof=1):.2f}, n={len(df)})")
        print(f"Avg tokens:   {df['tokens'].mean():.0f} "
              f"({df['input_tokens'].mean():.0f} in / "
              f"{df['output_tokens'].mean():.0f} out) over "
              f"{df['calls'].mean():.0f} LLM call(s)")
        print(f"Avg latency:  {df['latency'].mean():.1f}s")
        if df["cost"].notna().all():
            print(f"Avg cost:     ${df['cost'].mean():.6f}")
        else:
            print("Avg cost:     not priced (set PRICE_PER_1M_INPUT/OUTPUT)")
        return df

    df_single = print_summary("Single-Agent Baseline", results["single"])
    df_multi = print_summary("Multi-Agent System", results["multi"])

    # === 7. Head-to-Head Delta (paired) ===
    print("\n=== Improvement (Multi minus Single, paired by question) ===")
    n = len(df_single)
    decisive = {}
    for metric in ["accuracy", "completeness", "citations"]:
        diff = df_multi[metric] - df_single[metric]
        mean_d, sd_d = diff.mean(), diff.std(ddof=1)
        # Standard error of the paired mean. With n=5 and an integer 0-3 scale
        # this is usually wider than the effect itself -- which is the finding,
        # not something to hide. Reporting "+18%" off five samples with no
        # spread is how a coin flip turns into an architecture decision.
        se = sd_d / (n ** 0.5) if n > 1 and sd_d > 0 else 0.0
        decisive[metric] = se > 0 and abs(mean_d) > 2 * se
        label = "clear" if decisive[metric] else "within noise"
        print(f"{metric:14s}: {mean_d:+.2f} +/- {se:.2f} (SE, n={n})  [{label}]")

    tok_ratio = df_multi["tokens"].mean() / max(df_single["tokens"].mean(), 1)
    print(f"\nToken multiplier: {tok_ratio:.1f}x "
          f"({df_multi['calls'].mean():.0f} calls vs "
          f"{df_single['calls'].mean():.0f})")
    if df_single["cost"].notna().all() and df_multi["cost"].notna().all():
        ratio = df_multi["cost"].mean() / max(df_single["cost"].mean(), 1e-12)
        print(f"Cost multiplier:  {ratio:.1f}x")

    winners = [m for m, d in decisive.items()
               if d and (df_multi[m].mean() - df_single[m].mean()) > 0]
    if winners:
        print(f"\nVerdict: multi-agent is clearly better on {winners} at "
              f"{tok_ratio:.1f}x the tokens. Decide whether that trade is worth it.")
    else:
        print(f"\nVerdict: no difference distinguishable from noise at n={n}, and "
              f"multi-agent costs {tok_ratio:.1f}x the tokens. Either raise n until "
              f"the comparison can resolve the effect, or ship the cheaper arm. Do "
              f"NOT read the point estimates above as a result.")
    return results


if __name__ == "__main__":
    run_comparison(num_questions=5)
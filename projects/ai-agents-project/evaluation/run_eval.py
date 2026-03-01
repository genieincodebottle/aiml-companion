# ============================================
# Agent System Evaluation Framework
# ============================================
# uv pip install pandas google-genai

import json
import time
from dotenv import load_dotenv

load_dotenv()

_client = None
MODEL = "gemini-2.5-flash"
PRICE_PER_TOKEN = 0  # Gemini free tier - no cost


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


def _generate(prompt: str, max_tokens: int = 2000, json_output: bool = False):
    """Helper: call Gemini and return (text, token_count)."""
    config = {"max_output_tokens": max_tokens, "temperature": 0}
    if json_output:
        config["response_mime_type"] = "application/json"
    response = _get_client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )
    tokens = response.usage_metadata.total_token_count
    return response.text, tokens


# === 2. Single-Agent Baseline ===
def single_agent_research(query: str) -> dict:
    """One LLM call to research and write about a topic."""
    start = time.time()
    text, tokens = _generate(
        f"Research this topic and write a brief report with citations: {query}",
        max_tokens=2000,
    )
    return {
        "report": text,
        "tokens": tokens,
        "latency": time.time() - start,
        "cost": tokens * PRICE_PER_TOKEN,
    }


# === 3. Multi-Agent Research (Specialist Delegation) ===
def multi_agent_research(query: str) -> dict:
    """3-agent pipeline: researcher -> analyst -> writer."""
    start = time.time()
    total_tokens = 0

    # Agent 1: Researcher - gather information
    findings, t = _generate(
        f"You are a research specialist. Find key facts, data points, "
        f"and cite your sources. Return structured findings.\n\n"
        f"Research this topic thoroughly. List 5-8 key findings "
        f"with source references:\n{query}",
        max_tokens=1500,
    )
    total_tokens += t

    # Agent 2: Analyst - extract claims and assess confidence
    analysis, t = _generate(
        f"You are a research analyst. Evaluate findings for accuracy "
        f"and rank claims by confidence (high/medium/low).\n\n"
        f"Analyze these research findings. Identify the strongest "
        f"claims and flag any that seem unsupported:\n{findings}",
        max_tokens=1000,
    )
    total_tokens += t

    # Agent 3: Writer - produce structured report
    report, t = _generate(
        f"You are a technical writer. Write clear, well-structured "
        f"reports. Only include claims from the provided analysis.\n\n"
        f"Write a structured research report with these sections: "
        f"Introduction, Key Findings, Analysis, Conclusion.\n\n"
        f"Use ONLY these analyzed findings:\n{analysis}",
        max_tokens=2000,
    )
    total_tokens += t

    return {
        "report": report,
        "tokens": total_tokens,
        "latency": time.time() - start,
        "cost": total_tokens * PRICE_PER_TOKEN,
    }


# === 4. Evaluation Metrics (LLM-as-Judge) ===
def evaluate_report(query: str, report: str) -> dict:
    """Score a report on accuracy, completeness, and citation quality."""
    text, _ = _generate(
        f"Score this research report on a 0-3 scale for each dimension.\n\n"
        f"Query: {query}\nReport: {report[:2000]}\n\n"
        f"Score (0=missing, 1=poor, 2=adequate, 3=excellent):\n"
        f"- accuracy: (are claims factually correct?)\n"
        f"- completeness: (are key aspects covered?)\n"
        f"- citations: (are sources provided and real?)\n"
        f'Respond as JSON only: {{"accuracy": N, "completeness": N, "citations": N}}',
        json_output=True,
    )
    return json.loads(text)


# === 5. Run Comparison ===
def run_comparison(num_questions: int = 5):
    """Run head-to-head comparison on test questions."""
    import pandas as pd

    results = {"single": [], "multi": []}

    for test in TEST_QUESTIONS[:num_questions]:
        print(f"Evaluating: {test['query'][:50]}...")

        # Single-agent baseline
        single = single_agent_research(test["query"])
        single_scores = evaluate_report(test["query"], single["report"])
        results["single"].append({
            **single_scores, "tokens": single["tokens"],
            "latency": single["latency"], "cost": single["cost"]
        })

        # Multi-agent system
        multi = multi_agent_research(test["query"])
        multi_scores = evaluate_report(test["query"], multi["report"])
        results["multi"].append({
            **multi_scores, "tokens": multi["tokens"],
            "latency": multi["latency"], "cost": multi["cost"]
        })

    # === 6. Summary Report ===
    def print_summary(label: str, data: list[dict]):
        df = pd.DataFrame(data)
        print(f"\n=== {label} ===")
        print(f"Accuracy:     {df['accuracy'].mean():.1f}/3")
        print(f"Completeness: {df['completeness'].mean():.1f}/3")
        print(f"Citations:    {df['citations'].mean():.1f}/3")
        print(f"Avg tokens:   {df['tokens'].mean():.0f}")
        print(f"Avg latency:  {df['latency'].mean():.1f}s")
        print(f"Avg cost:     ${df['cost'].mean():.4f}")
        return df

    df_single = print_summary("Single-Agent Baseline", results["single"])
    df_multi = print_summary("Multi-Agent System", results["multi"])

    # === 7. Head-to-Head Delta ===
    print("\n=== Improvement (Multi vs Single) ===")
    for metric in ["accuracy", "completeness", "citations"]:
        delta = df_multi[metric].mean() - df_single[metric].mean()
        pct = (delta / max(df_single[metric].mean(), 0.01)) * 100
        print(f"{metric:14s}: {'+' if delta >= 0 else ''}{delta:.1f} ({pct:+.0f}%)")
    cost_ratio = df_multi["cost"].mean() / max(df_single["cost"].mean(), 0.0001)
    print(f"Cost multiplier: {cost_ratio:.1f}x")

    verdict = (
        "Multi-agent justified"
        if df_multi["accuracy"].mean() > df_single["accuracy"].mean() * 1.2
        else "Consider optimizing single-agent first"
    )
    print(f"\nVerdict: {verdict}")
    return results


if __name__ == "__main__":
    run_comparison(num_questions=5)
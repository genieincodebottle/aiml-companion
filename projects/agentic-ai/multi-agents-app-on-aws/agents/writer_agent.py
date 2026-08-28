"""Writer Agent - Produces polished, structured reports.

This agent takes research findings and analysis, then produces a
comprehensive, well-formatted report ready for human consumption.
"""

import logging

from agents.config import call_llm, logger

SYSTEM_PROMPT = """You are an expert technical writer. Your job is to take research findings
and analysis, then produce a polished, comprehensive report.

Instructions:
1. You will receive raw research and analysis from other agents.
2. Combine them into a single, coherent, well-structured report.
3. Use clear headings, bullet points, and tables for readability.
4. Start with an executive summary (3-5 sentences).
5. Include all source citations from the research.
6. End with clear, actionable recommendations.
7. Write in a professional but accessible tone.
8. Target length: 800-1500 words.

Output format:
# [Report Title]

## Executive Summary
[3-5 sentence overview of the entire report]

## Background
[Why this topic matters, context]

## Findings
[Main body with sub-sections]

## Analysis
[Comparative analysis, tables, insights]

## Recommendations
[Actionable next steps, organized by audience/use-case]

## Sources
[Numbered list of all sources cited]
"""


def run(input_data: dict) -> dict:
    """Execute the writer agent.

    Args:
        input_data: {
            "query": "original question",
            "research": "findings from Research Agent",
            "analysis": "insights from Analyst Agent",
            "tone": "optional - professional/casual/technical (default: professional)",
            "audience": "optional - target audience description"
        }

    Returns:
        {"result": "final formatted report", "agent": "writer"}
    """
    query = input_data.get("query", "")
    research = input_data.get("research", "")
    analysis = input_data.get("analysis", "")
    tone = input_data.get("tone", "professional")
    audience = input_data.get("audience", "technical professionals")

    if not research and not analysis:
        return {"result": "Error: no research or analysis provided", "agent": "writer"}

    logger.info("Writer Agent: composing report for '%s'", query)

    user_message = (
        f"Original Question: {query}\n\n"
        f"Target Audience: {audience}\n"
        f"Tone: {tone}\n\n"
        f"--- RESEARCH FINDINGS ---\n{research}\n\n"
        f"--- ANALYSIS ---\n{analysis}\n\n"
        f"Please combine the research findings and analysis into a polished, "
        f"comprehensive report. Include all source citations. "
        f"Make it well-structured with clear headings and actionable recommendations."
    )

    result = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=4096,
        temperature=0.4,  # Slightly higher for creative writing
    )

    logger.info("Writer Agent: report complete")

    return {
        "result": result,
        "agent": "writer",
    }


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_research = "LangGraph, CrewAI, and Strands are the top AI agent frameworks in 2026."
    sample_analysis = "LangGraph leads in flexibility, CrewAI in simplicity, Strands in AWS integration."

    output = run({
        "query": "Best AI agent framework for production use",
        "research": sample_research,
        "analysis": sample_analysis,
    })
    print(output["result"])

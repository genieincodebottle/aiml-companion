"""Analyst Agent - Analyzes research data and produces structured insights.

This agent takes research findings and produces comparative analysis,
identifies patterns, and generates data-driven insights.
"""

import logging

from agents.config import call_llm, logger

SYSTEM_PROMPT = """You are an expert data analyst. Your job is to analyze research findings
and produce structured, insightful analysis.

Instructions:
1. You will receive research findings from the Research Agent.
2. Identify key patterns, trends, and comparisons in the data.
3. Create structured comparisons (tables, rankings, pros/cons).
4. Quantify where possible - use numbers, percentages, rankings.
5. Highlight non-obvious insights that add value beyond the raw research.
6. Be objective - present both strengths and weaknesses.

Output format:
## Analysis Summary
[2-3 sentence executive summary of key insights]

## Comparative Analysis
| Criterion | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| ...       | ...      | ...      | ...      |

## Key Insights
1. **Insight title**: Explanation with supporting evidence
2. **Insight title**: Explanation with supporting evidence

## Trends
- Trend 1: description
- Trend 2: description

## Recommendations
- For [use case]: recommend [option] because [reason]
"""


def run(input_data: dict) -> dict:
    """Execute the analyst agent.

    Args:
        input_data: {
            "query": "original research question",
            "research": "findings from Research Agent",
            "focus_areas": "optional - specific aspects to analyze"
        }

    Returns:
        {"result": "structured analysis", "agent": "analyst"}
    """
    query = input_data.get("query", "")
    research = input_data.get("research", "")
    focus_areas = input_data.get("focus_areas", "")

    if not research:
        return {"result": "Error: no research data provided for analysis", "agent": "analyst"}

    logger.info("Analyst Agent: analyzing research for '%s'", query)

    user_message = (
        f"Original Question: {query}\n\n"
        f"Research Findings:\n{research}\n\n"
        f"{'Focus Areas: ' + focus_areas + chr(10) + chr(10) if focus_areas else ''}"
        f"Please analyze these research findings. Create comparisons, identify patterns, "
        f"and provide data-driven insights. Use tables for comparisons where appropriate."
    )

    result = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        temperature=0.3,
    )

    logger.info("Analyst Agent: analysis complete")

    return {
        "result": result,
        "agent": "analyst",
    }


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_research = """
    ## Key Findings
    - LangGraph has 15k+ GitHub stars, backed by LangChain team [Source 1]
    - CrewAI focuses on role-based agents, 20k+ stars [Source 2]
    - AWS Strands Agents is newest, deep AWS integration [Source 3]
    - AutoGen by Microsoft supports multi-agent conversations [Source 4]

    ## Details
    LangGraph uses a graph-based approach where nodes are agents/tools and edges
    define the flow. CrewAI uses a crew metaphor with roles and tasks. Strands
    focuses on simplicity with tool-use agents. AutoGen emphasizes conversational
    patterns between agents.
    """

    output = run({
        "query": "Compare AI agent frameworks",
        "research": sample_research,
    })
    print(output["result"])

"""Orchestrator Agent - Coordinates the multi-agent research team.

The orchestrator is the supervisor. It:
1. Receives a user request
2. Breaks it down into subtasks
3. Delegates to specialist agents (Research -> Analyst -> Writer)
4. Combines outputs into a final response

Local mode: calls agents directly as Python functions.
Deployed mode: calls agents via AgentCore A2A protocol (when ARNs are set).
"""

import json
import time
import argparse
import logging

from agents.config import call_llm, logger, MAX_AGENT_ITERATIONS
from agents import research_agent, analyst_agent, writer_agent

# ---------------------------------------------------------------------------
# AgentCore ARNs (set after deploying individual agents)
# Leave empty for local mode (direct function calls)
# ---------------------------------------------------------------------------

AGENT_ARNS = {
    "research": "",  # Set after: agentcore launch (research_agent)
    "analyst": "",   # Set after: agentcore launch (analyst_agent)
    "writer": "",    # Set after: agentcore launch (writer_agent)
}

# ---------------------------------------------------------------------------
# Orchestrator system prompt
# ---------------------------------------------------------------------------

PLANNING_PROMPT = """You are an orchestrator that coordinates a team of AI agents to fulfill
user requests. You have access to these specialist agents:

1. **Research Agent** - Searches the web and summarizes findings. Use for any topic
   that needs current information, facts, or data gathering.
2. **Analyst Agent** - Analyzes data, creates comparisons, identifies patterns.
   Use when research needs to be compared, ranked, or distilled into insights.
3. **Writer Agent** - Produces polished reports from research and analysis.
   Use as the final step to create a readable deliverable.

Given the user's request, create a plan with 2-4 steps. For each step, specify:
- Which agent to use (research, analyst, or writer)
- What to ask that agent
- What context from previous steps to pass along

Respond in this exact JSON format:
{
  "plan": [
    {"agent": "research", "task": "what to research", "needs_previous": false},
    {"agent": "analyst", "task": "what to analyze", "needs_previous": true},
    {"agent": "writer", "task": "what to write", "needs_previous": true}
  ]
}

Rules:
- Always start with research (unless the user explicitly says they don't need it)
- Analysis should come after research
- Writing should be the last step (to compile everything)
- Keep tasks specific and actionable
- Return ONLY valid JSON, no extra text
"""


# ---------------------------------------------------------------------------
# Local agent dispatch
# ---------------------------------------------------------------------------

AGENTS = {
    "research": research_agent,
    "analyst": analyst_agent,
    "writer": writer_agent,
}


def _call_agent_local(agent_name: str, input_data: dict) -> dict:
    """Call an agent directly as a Python function (local mode)."""
    agent_module = AGENTS.get(agent_name)
    if not agent_module:
        return {"result": f"Error: unknown agent '{agent_name}'", "agent": agent_name}
    return agent_module.run(input_data)


def _call_agent_remote(agent_name: str, input_data: dict) -> dict:
    """Call an agent via AgentCore Runtime (deployed mode).

    Uses boto3 to invoke the agent's AgentCore endpoint.
    """
    arn = AGENT_ARNS.get(agent_name, "")
    if not arn:
        logger.warning("No ARN for agent '%s' - falling back to local mode", agent_name)
        return _call_agent_local(agent_name, input_data)

    from agents.config import get_bedrock_agent_client

    client = get_bedrock_agent_client()

    response = client.invoke_agent_runtime(
        agentRuntimeArn=arn,
        runtimeSessionId=f"session-{int(time.time())}",
        payload=input_data,
    )

    return json.loads(response["body"].read().decode("utf-8"))


def call_agent(agent_name: str, input_data: dict) -> dict:
    """Route to local or remote agent based on whether ARNs are configured."""
    if AGENT_ARNS.get(agent_name):
        return _call_agent_remote(agent_name, input_data)
    return _call_agent_local(agent_name, input_data)


# ---------------------------------------------------------------------------
# Orchestrator logic
# ---------------------------------------------------------------------------

def create_plan(user_query: str) -> list[dict]:
    """Ask the LLM to create an execution plan for the given query."""
    response = call_llm(
        system_prompt=PLANNING_PROMPT,
        user_message=user_query,
        temperature=0.1,  # Very low temperature for structured planning
    )

    # Parse the JSON plan from the LLM response
    try:
        # Handle cases where LLM wraps JSON in markdown code blocks
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]  # Remove first line
            cleaned = cleaned.rsplit("```", 1)[0]  # Remove last ```
        plan_data = json.loads(cleaned)
        return plan_data.get("plan", [])
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM plan - using default research->analyze->write flow")
        return [
            {"agent": "research", "task": user_query, "needs_previous": False},
            {"agent": "analyst", "task": f"Analyze the research findings about: {user_query}", "needs_previous": True},
            {"agent": "writer", "task": f"Write a comprehensive report about: {user_query}", "needs_previous": True},
        ]


def run(query: str) -> dict:
    """Execute the full multi-agent pipeline for a given query.

    Returns:
        {
            "query": original query,
            "plan": execution plan,
            "steps": list of step results,
            "final_report": the writer's output,
            "total_time": execution time in seconds,
        }
    """
    start_time = time.time()
    logger.info("Orchestrator: received query '%s'", query)

    # Step 1: Create a plan
    logger.info("Orchestrator: creating execution plan...")
    plan = create_plan(query)
    logger.info("Orchestrator: plan has %d steps", len(plan))

    # Safety: limit iterations
    if len(plan) > MAX_AGENT_ITERATIONS:
        plan = plan[:MAX_AGENT_ITERATIONS]
        logger.warning("Orchestrator: plan truncated to %d steps", MAX_AGENT_ITERATIONS)

    # Step 2: Execute each step, passing context forward
    steps = []
    accumulated_research = ""
    accumulated_analysis = ""

    for i, step in enumerate(plan):
        agent_name = step["agent"]
        task = step["task"]
        needs_previous = step.get("needs_previous", False)

        logger.info("Orchestrator: step %d/%d - calling %s agent", i + 1, len(plan), agent_name)

        # Build input for this agent
        input_data = {"query": task}

        if needs_previous and agent_name == "analyst":
            input_data["research"] = accumulated_research
        elif needs_previous and agent_name == "writer":
            input_data["research"] = accumulated_research
            input_data["analysis"] = accumulated_analysis
        elif needs_previous and agent_name == "research":
            input_data["context"] = accumulated_research

        # Execute
        result = call_agent(agent_name, input_data)
        steps.append({
            "step": i + 1,
            "agent": agent_name,
            "task": task,
            "result": result.get("result", ""),
        })

        # Accumulate context for next steps
        if agent_name == "research":
            accumulated_research += f"\n\n{result.get('result', '')}"
        elif agent_name == "analyst":
            accumulated_analysis += f"\n\n{result.get('result', '')}"

        logger.info("Orchestrator: step %d complete", i + 1)

    # Step 3: Determine final output
    total_time = time.time() - start_time

    # The final report is the last writer output, or the last step if no writer
    final_report = ""
    for step in reversed(steps):
        if step["agent"] == "writer":
            final_report = step["result"]
            break
    if not final_report and steps:
        final_report = steps[-1]["result"]

    logger.info("Orchestrator: pipeline complete in %.1fs", total_time)

    return {
        "query": query,
        "plan": plan,
        "steps": steps,
        "final_report": final_report,
        "total_time": round(total_time, 1),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent AI Research Team",
        epilog="Example: python -m agents.orchestrator --query 'Compare AI agent frameworks'",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="Compare the top AI agent frameworks in 2026: LangGraph, CrewAI, Strands Agents, and OpenAI Agents SDK",
        help="Research query to process",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed step-by-step output",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  Multi-Agent AI Research Team")
    print(f"{'=' * 70}")
    print(f"\n  Query: {args.query}\n")

    result = run(args.query)

    if args.verbose:
        print(f"\n{'=' * 70}")
        print(f"  Execution Plan ({len(result['plan'])} steps)")
        print(f"{'=' * 70}")
        for step in result["plan"]:
            print(f"  [{step['agent'].upper()}] {step['task']}")

        print(f"\n{'=' * 70}")
        print(f"  Step-by-Step Results")
        print(f"{'=' * 70}")
        for step in result["steps"]:
            print(f"\n--- Step {step['step']}: {step['agent'].upper()} ---")
            print(step["result"][:500] + "..." if len(step["result"]) > 500 else step["result"])

    print(f"\n{'=' * 70}")
    print(f"  FINAL REPORT")
    print(f"{'=' * 70}\n")
    print(result["final_report"])

    print(f"\n{'=' * 70}")
    print(f"  Completed in {result['total_time']}s | Steps: {len(result['steps'])}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

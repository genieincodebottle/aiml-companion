"""AgentCore Runtime entrypoint for the Analyst Agent.

Deploy: agentcore configure -e analyst_agent.py && agentcore launch
"""

from bedrock_agentcore import BedrockAgentCoreApp, PingStatus

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.analyst_agent import run as analyst_run

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Handle incoming requests from the Orchestrator."""
    query = payload.get("query", payload.get("prompt", ""))
    research = payload.get("research", "")
    focus_areas = payload.get("focus_areas", "")

    result = analyst_run({
        "query": query,
        "research": research,
        "focus_areas": focus_areas,
    })

    return {
        "result": result["result"],
        "agent": "analyst",
    }


@app.ping
def health():
    return PingStatus.HEALTHY


if __name__ == "__main__":
    app.run(port=8082)

"""AgentCore Runtime entrypoint for the Orchestrator Agent.

The orchestrator is deployed as its own AgentCore service and communicates
with the specialist agents via their AgentCore endpoints.

Deploy: agentcore configure -e orchestrator.py && agentcore launch
"""

from bedrock_agentcore import BedrockAgentCoreApp, PingStatus

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.orchestrator import run as orchestrator_run

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Handle incoming research requests."""
    query = payload.get("query", payload.get("prompt", ""))

    if not query:
        return {"error": "No query provided. Send {\"query\": \"your research question\"}"}

    result = orchestrator_run(query)

    return {
        "final_report": result["final_report"],
        "total_time": result["total_time"],
        "steps_count": len(result["steps"]),
    }


@app.ping
def health():
    return PingStatus.HEALTHY


if __name__ == "__main__":
    app.run(port=8080)

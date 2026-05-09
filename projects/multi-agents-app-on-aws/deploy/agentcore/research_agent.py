"""AgentCore Runtime entrypoint for the Research Agent.

This file is deployed to AWS Bedrock AgentCore. It wraps the research agent
as an AgentCore-compatible service with health checks and session management.

Deploy: agentcore configure -e research_agent.py && agentcore launch
"""

from bedrock_agentcore import BedrockAgentCoreApp, PingStatus

# Import the actual agent logic
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.research_agent import run as research_run

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Handle incoming requests from the Orchestrator or direct invocation."""
    query = payload.get("query", payload.get("prompt", ""))
    context = payload.get("context", "")

    result = research_run({
        "query": query,
        "context": context,
    })

    return {
        "result": result["result"],
        "sources": result.get("sources", []),
        "agent": "research",
    }


@app.ping
def health():
    """Health check endpoint."""
    return PingStatus.HEALTHY


if __name__ == "__main__":
    app.run(port=8081)

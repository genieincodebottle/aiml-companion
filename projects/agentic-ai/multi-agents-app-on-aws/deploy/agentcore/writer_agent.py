"""AgentCore Runtime entrypoint for the Writer Agent.

Deploy: agentcore configure -e writer_agent.py && agentcore launch
"""

from bedrock_agentcore import BedrockAgentCoreApp, PingStatus

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.writer_agent import run as writer_run

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Handle incoming requests from the Orchestrator."""
    query = payload.get("query", payload.get("prompt", ""))
    research = payload.get("research", "")
    analysis = payload.get("analysis", "")

    result = writer_run({
        "query": query,
        "research": research,
        "analysis": analysis,
    })

    return {
        "result": result["result"],
        "agent": "writer",
    }


@app.ping
def health():
    return PingStatus.HEALTHY


if __name__ == "__main__":
    app.run(port=8083)

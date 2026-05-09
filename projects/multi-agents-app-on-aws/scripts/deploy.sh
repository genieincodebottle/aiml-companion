#!/bin/bash
# ===========================================================================
# Deploy All Agents to AWS Bedrock AgentCore
# ===========================================================================

set -e

echo "========================================="
echo "  Deploying Multi-Agent App to AgentCore"
echo "========================================="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check agentcore CLI
if ! command -v agentcore &> /dev/null; then
    echo -e "${YELLOW}AgentCore CLI not found. Installing...${NC}"
    uv pip install bedrock-agentcore-starter-toolkit
fi

cd deploy/agentcore

# ---------------------------------------------------------------------------
# Deploy each agent
# ---------------------------------------------------------------------------

AGENTS=("research_agent" "analyst_agent" "writer_agent" "orchestrator")

for agent in "${AGENTS[@]}"; do
    echo ""
    echo -e "${YELLOW}Deploying ${agent}...${NC}"
    echo "-------------------------------------------"

    # Configure
    agentcore configure -e "${agent}.py"

    # Deploy
    agentcore launch

    echo -e "${GREEN}✓ ${agent} deployed${NC}"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "========================================="
echo -e "  ${GREEN}All agents deployed!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Test: agentcore invoke '{\"query\": \"Compare AI agent frameworks\"}'"
echo "  2. Run UI: cd ui && streamlit run app.py"
echo "  3. View logs: aws logs tail /agentcore/multi-agents/dev --follow"
echo ""
echo "To tear down: bash scripts/cleanup.sh"
echo ""

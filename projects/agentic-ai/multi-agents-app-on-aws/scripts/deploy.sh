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

# Runtime role. AGENTCORE_ROLE_ARN wins; otherwise use the role Terraform created
# (output agentcore_runtime_role_arn in deploy/terraform). Empty means the toolkit
# creates its own role, as before.
ROLE_ARN="${AGENTCORE_ROLE_ARN:-}"
if [ -z "$ROLE_ARN" ] && command -v terraform &> /dev/null; then
    ROLE_ARN="$(terraform -chdir=deploy/terraform output -raw agentcore_runtime_role_arn 2>/dev/null || true)"
fi
if [ -n "$ROLE_ARN" ]; then
    echo "Using runtime role: $ROLE_ARN"
else
    echo -e "${YELLOW}No runtime role found; agentcore configure will create one.${NC}"
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

    # Configure. --execution-role (-er) is the starter toolkit flag
    # (bedrock-agentcore-starter-toolkit 0.3.x, cli/runtime/commands.py).
    if [ -n "$ROLE_ARN" ]; then
        agentcore configure -e "${agent}.py" --execution-role "$ROLE_ARN"
    else
        agentcore configure -e "${agent}.py"
    fi

    # Deploy
    agentcore launch

    echo -e "${GREEN}[OK] ${agent} deployed${NC}"
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

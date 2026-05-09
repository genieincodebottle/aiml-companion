#!/bin/bash
# ===========================================================================
# Test Deployed Agents
# ===========================================================================

set -e

echo "========================================="
echo "  Testing Multi-Agent App"
echo "========================================="
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Test 1: Local mode (direct Python)
# ---------------------------------------------------------------------------

echo "Test 1: Local mode (Research Agent)"
echo "-------------------------------------------"
RESULT=$(python -m agents.research_agent "Latest AI agent frameworks" 2>/dev/null | head -5)
if [ -n "$RESULT" ]; then
    echo -e "${GREEN}✓ Research Agent works locally${NC}"
else
    echo -e "${RED}✗ Research Agent failed${NC}"
fi

echo ""
echo "Test 2: Local mode (Analyst Agent)"
echo "-------------------------------------------"
RESULT=$(python -m agents.analyst_agent 2>/dev/null | head -5)
if [ -n "$RESULT" ]; then
    echo -e "${GREEN}✓ Analyst Agent works locally${NC}"
else
    echo -e "${RED}✗ Analyst Agent failed${NC}"
fi

echo ""
echo "Test 3: Local mode (Full Orchestrator Pipeline)"
echo "-------------------------------------------"
RESULT=$(python -m agents.orchestrator --query "What is AWS Bedrock AgentCore?" 2>/dev/null | tail -10)
if [ -n "$RESULT" ]; then
    echo -e "${GREEN}✓ Full pipeline works locally${NC}"
    echo ""
    echo "Sample output (last 10 lines):"
    echo "$RESULT"
else
    echo -e "${RED}✗ Orchestrator pipeline failed${NC}"
fi

# ---------------------------------------------------------------------------
# Test 4: Deployed mode (if agents are deployed)
# ---------------------------------------------------------------------------

echo ""
echo "Test 4: Deployed mode (AgentCore)"
echo "-------------------------------------------"

if command -v agentcore &> /dev/null; then
    DEPLOYED=$(agentcore invoke '{"query": "Hello, are you working?"}' 2>/dev/null || echo "")
    if [ -n "$DEPLOYED" ]; then
        echo -e "${GREEN}✓ Deployed agent responds${NC}"
    else
        echo -e "${YELLOW}⊘ No deployed agents found (use 'bash scripts/deploy.sh' to deploy)${NC}"
    fi
else
    echo -e "${YELLOW}⊘ agentcore CLI not installed (uv pip install bedrock-agentcore-starter-toolkit)${NC}"
fi

echo ""
echo "========================================="
echo "  Tests complete"
echo "========================================="
echo ""

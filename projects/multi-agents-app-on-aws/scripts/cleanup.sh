#!/bin/bash
# ===========================================================================
# Cleanup - Remove all deployed AgentCore resources
# ===========================================================================

set -e

echo "========================================="
echo "  Cleaning Up Multi-Agent App Resources"
echo "========================================="
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}WARNING: This will delete all deployed agents and associated resources.${NC}"
read -p "Are you sure? (y/N): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

# ---------------------------------------------------------------------------
# Delete AgentCore agents
# ---------------------------------------------------------------------------

echo ""
echo "Deleting AgentCore agents..."

# List and delete all agents with our project prefix
AGENTS=$(aws bedrock-agent-runtime list-agent-runtimes --query "agentRuntimes[?contains(name, 'multi-agents')].[name,arn]" --output text 2>/dev/null || echo "")

if [ -z "$AGENTS" ]; then
    echo -e "  ${GREEN}✓${NC} No AgentCore agents found to delete"
else
    while IFS=$'\t' read -r name arn; do
        echo "  Deleting: $name"
        aws bedrock-agent-runtime delete-agent-runtime --agent-runtime-arn "$arn" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Deleted $name"
    done <<< "$AGENTS"
fi

# ---------------------------------------------------------------------------
# Terraform cleanup (if used)
# ---------------------------------------------------------------------------

if [ -f "deploy/terraform/terraform.tfstate" ]; then
    echo ""
    echo "Destroying Terraform resources..."
    cd deploy/terraform
    terraform destroy -auto-approve -var="aws_region=$(aws configure get region || echo us-east-1)"
    cd ../..
    echo -e "  ${GREEN}✓${NC} Terraform resources destroyed"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "========================================="
echo -e "  ${GREEN}Cleanup complete!${NC}"
echo "========================================="
echo ""

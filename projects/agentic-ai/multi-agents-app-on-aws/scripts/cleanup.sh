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

# ---------------------------------------------------------------------------
# Usage: bash scripts/cleanup.sh [--dry-run]
#   --dry-run   list what would be deleted, delete nothing
#
# Two cases:
#   1. deploy/multiagents/ exists: deploy.sh created it with the AgentCore CLI
#      (@aws/agentcore). The runtimes belong to a CloudFormation stack, so they
#      are removed the CLI way: `agentcore remove all --yes`, then
#      `agentcore deploy --yes`, which deletes the stack.
#   2. No such folder: the runtimes came from the old starter toolkit
#      (`agentcore configure -e <file>.py`), which named each runtime after the
#      entrypoint file. They are found and deleted by name with the AWS CLI.
#      AgentCore runtime names match [a-zA-Z][a-zA-Z0-9_]{0,47}, i.e. NO
#      hyphens, so a name filter such as "multi-agents" can never match.
#      Override the list with e.g.
#        AGENT_RUNTIME_NAMES="research_agent analyst_agent" bash scripts/cleanup.sh
# ---------------------------------------------------------------------------

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

CLI_PROJECT="deploy/multiagents"
RUNTIME_NAMES="${AGENT_RUNTIME_NAMES:-research_agent analyst_agent writer_agent orchestrator}"

REGION_ARGS=()
if [ -n "${AWS_REGION:-}" ]; then
    REGION_ARGS=(--region "$AWS_REGION")
fi

TARGETS=""

if [ -d "$CLI_PROJECT/agentcore" ]; then
    echo "AgentCore CLI project found: $CLI_PROJECT"
    echo "It will run: agentcore remove all --yes, then agentcore deploy --yes"
else
    # -----------------------------------------------------------------------
    # Find AgentCore runtimes (control-plane API: bedrock-agentcore-control)
    # -----------------------------------------------------------------------

    echo "Looking for AgentCore runtimes: ${RUNTIME_NAMES}"

    # Output is "<agentRuntimeName>	<agentRuntimeId>" per runtime (auto-paginated).
    # A failure here (no credentials, no permission, old AWS CLI) must stop the
    # script, not be reported as "nothing to delete".
    ALL_RUNTIMES=$(aws bedrock-agentcore-control list-agent-runtimes --query "agentRuntimes[].[agentRuntimeName,agentRuntimeId]" --output text)

    while IFS=$'\t' read -r name id; do
        [ -z "$name" ] && continue
        for wanted in $RUNTIME_NAMES; do
            if [ "$name" == "$wanted" ]; then
                TARGETS+="${name}"$'\t'"${id}"$'\n'
            fi
        done
    done <<< "$ALL_RUNTIMES"

    if [ -z "$TARGETS" ]; then
        echo -e "  ${GREEN}[OK]${NC} No matching AgentCore runtimes found to delete"
    else
        echo "Runtimes that will be deleted:"
        while IFS=$'\t' read -r name id; do
            [ -z "$name" ] && continue
            echo "  - $name ($id)"
        done <<< "$TARGETS"
    fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "Dry run: nothing deleted."
    exit 0
fi

echo ""
echo -e "${YELLOW}WARNING: This will delete the runtimes listed above and any Terraform-managed resources.${NC}"
read -p "Are you sure? (y/N): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    exit 0
fi

# ---------------------------------------------------------------------------
# Delete AgentCore runtimes
# ---------------------------------------------------------------------------

FAILED=0
if [ -d "$CLI_PROJECT/agentcore" ]; then
    echo ""
    echo "Removing the AgentCore CLI project's resources and stack..."
    if (cd "$CLI_PROJECT" \
        && agentcore remove all --yes ${REGION_ARGS[@]+"${REGION_ARGS[@]}"} \
        && agentcore deploy --yes ${REGION_ARGS[@]+"${REGION_ARGS[@]}"}); then
        echo -e "  ${GREEN}[OK]${NC} CLI project stack removed"
    else
        echo -e "  ${RED}[FAIL]${NC} Could not remove the CLI project stack"
        FAILED=1
    fi
elif [ -n "$TARGETS" ]; then
    echo ""
    echo "Deleting AgentCore runtimes..."
    while IFS=$'\t' read -r name id; do
        [ -z "$name" ] && continue
        echo "  Deleting: $name ($id)"
        if aws bedrock-agentcore-control delete-agent-runtime --agent-runtime-id "$id" > /dev/null; then
            echo -e "  ${GREEN}[OK]${NC} Deleted $name"
        else
            echo -e "  ${RED}[FAIL]${NC} Could not delete $name"
            FAILED=1
        fi
    done <<< "$TARGETS"
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
    echo -e "  ${GREEN}[OK]${NC} Terraform resources destroyed"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "========================================="
if [ "$FAILED" -ne 0 ]; then
    echo -e "  ${RED}Cleanup finished with errors (see above)${NC}"
    echo "========================================="
    exit 1
fi
echo -e "  ${GREEN}Cleanup complete!${NC}"
echo "========================================="
echo ""

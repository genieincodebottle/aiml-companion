#!/bin/bash
# ===========================================================================
# Deploy All Agents to AWS Bedrock AgentCore (AgentCore CLI, @aws/agentcore)
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

# Check agentcore CLI. The new CLI is an npm package (Node 20.12 or later).
# The old Python starter toolkit also installs an `agentcore` command, so remove it first.
if ! command -v agentcore &> /dev/null; then
    echo -e "${RED}AgentCore CLI not found.${NC}"
    echo "  Install it:  npm install -g @aws/agentcore   (needs Node 20.12+ and uv)"
    echo "  If you used the old toolkit:  pip uninstall bedrock-agentcore-starter-toolkit"
    exit 1
fi
if ! agentcore add runtime --help 2>&1 | grep -q -- "--role-arn"; then
    echo -e "${RED}This agentcore is not the new AgentCore CLI (no 'add runtime').${NC}"
    echo "  Run: pip uninstall bedrock-agentcore-starter-toolkit"
    echo "  Then: npm install -g @aws/agentcore"
    exit 1
fi
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv not found.${NC} The AgentCore CLI uses it to install Python dependencies."
    exit 1
fi

# Runtime role. AGENTCORE_ROLE_ARN wins; otherwise use the role Terraform created
# (output agentcore_runtime_role_arn in deploy/terraform). Empty means the CLI
# creates a role for each runtime (CDK-managed).
ROLE_ARN="${AGENTCORE_ROLE_ARN:-}"
if [ -z "$ROLE_ARN" ] && command -v terraform &> /dev/null; then
    ROLE_ARN="$(terraform -chdir=deploy/terraform output -raw agentcore_runtime_role_arn 2>/dev/null || true)"
fi
if [ -n "$ROLE_ARN" ]; then
    echo "Using runtime role: $ROLE_ARN"
else
    echo -e "${YELLOW}No runtime role found; the CLI will create one per runtime.${NC}"
fi

# Region. --region wins over AWS_REGION, AWS_DEFAULT_REGION, the profile, then us-east-1.
REGION_ARGS=()
if [ -n "${AWS_REGION:-}" ]; then
    REGION_ARGS=(--region "$AWS_REGION")
fi

ROOT="$(pwd)"
PROJECT=multiagents
AGENTS=("research_agent" "analyst_agent" "writer_agent" "orchestrator")

# ---------------------------------------------------------------------------
# One AgentCore CLI project in deploy/multiagents/, generated on first run.
# Each agent becomes a runtime in app/<agent>/ with its own main.py, agents/,
# tools/ and pyproject.toml. The directory is a build output: do not edit it.
# ---------------------------------------------------------------------------

cd deploy
if [ ! -d "$PROJECT/agentcore" ]; then
    agentcore create --name "$PROJECT" --skip-git ${REGION_ARGS[@]+"${REGION_ARGS[@]}"}
fi
cd "$PROJECT"

# AgentCore serves HTTP on 8080. The wrappers read AGENT_PORT and default to
# their own local port, so set it here for the deployed runtimes.
ENV_JSON="${AGENTCORE_ENV_JSON:-}"
if [ -z "$ENV_JSON" ]; then
    ENV_JSON='{"AGENT_PORT": "8080"}'
fi

for agent in "${AGENTS[@]}"; do
    echo ""
    echo -e "${YELLOW}Preparing ${agent}...${NC}"
    echo "-------------------------------------------"

    # Register the runtime once. --role-arn is the existing execution role.
    if ! grep -q "\"name\": \"${agent}\"" agentcore/agentcore.json; then
        if [ -n "$ROLE_ARN" ]; then
            agentcore add runtime --name "$agent" --language python \
                --role-arn "$ROLE_ARN" --environment-variables "$ENV_JSON" \
                ${REGION_ARGS[@]+"${REGION_ARGS[@]}"}
        else
            agentcore add runtime --name "$agent" --language python \
                --environment-variables "$ENV_JSON" \
                ${REGION_ARGS[@]+"${REGION_ARGS[@]}"}
        fi
    fi

    # Copy the code in. The CLI uploads only app/<agent>/.
    dest="app/${agent}"
    rm -rf "$dest/agents" "$dest/tools"
    cp -r "$ROOT/agents" "$ROOT/tools" "$dest/"
    cp "$ROOT/deploy/agentcore/${agent}.py" "$dest/main.py"

    # The CLI reads dependencies only from pyproject.toml, so build it from
    # deploy/agentcore/requirements.txt. AgentCore starts the code with
    # opentelemetry-instrument, so aws-opentelemetry-distro is required.
    DEPS="$(tr -d '\r' < "$ROOT/deploy/agentcore/requirements.txt" | sed -e 's/#.*//' -e '/^[[:space:]]*$/d' -e 's/.*/    "&",/')"
    cat > "$dest/pyproject.toml" <<EOF
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "${agent}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
${DEPS}
    "aws-opentelemetry-distro~=0.21.0",
    "python-dotenv>=1.2",
]

[tool.hatch.build.targets.wheel]
packages = ["."]
EOF
    echo -e "${GREEN}[OK] ${agent} ready${NC}"
done

# ---------------------------------------------------------------------------
# Deploy all four runtimes in one step
# ---------------------------------------------------------------------------

echo ""
echo -e "${YELLOW}Deploying (first run bootstraps CDK in the account and region)...${NC}"
agentcore deploy ${REGION_ARGS[@]+"${REGION_ARGS[@]}"}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "========================================="
echo -e "  ${GREEN}All agents deployed!${NC}"
echo "========================================="
echo ""
echo "Next steps (run from deploy/multiagents):"
echo "  1. Status: agentcore status"
echo "  2. Test:   agentcore invoke --runtime orchestrator --payload '{\"query\": \"Compare AI agent frameworks\"}'"
echo "  3. Logs:   agentcore logs runtime --name orchestrator --since 1h"
echo "  4. Run UI: cd ui && streamlit run app.py"
echo ""
echo "To tear down: bash scripts/cleanup.sh"
echo ""

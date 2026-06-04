#!/bin/bash
# ===========================================================================
# Initial Setup Script
# Creates virtual environment, installs dependencies, validates AWS config
# ===========================================================================

set -e

echo "========================================="
echo "  Multi-Agent App - Initial Setup"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# 1. Check prerequisites
# ---------------------------------------------------------------------------

echo "Checking prerequisites..."

# Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python 3.10+ is required. Install from https://python.org/downloads${NC}"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON --version 2>&1 | grep -oP '\d+\.\d+')
echo -e "  ${GREEN}[OK]${NC} Python $PY_VERSION found"

# uv
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Reload PATH
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}ERROR: uv installation failed. Install manually: https://docs.astral.sh/uv/getting-started/installation/${NC}"
        exit 1
    fi
fi
echo -e "  ${GREEN}[OK]${NC} uv $(uv --version) found"

# AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}ERROR: AWS CLI v2 is required. Install from https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html${NC}"
    exit 1
fi
echo -e "  ${GREEN}[OK]${NC} AWS CLI found"

# AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${YELLOW}WARNING: AWS credentials not configured. Run 'aws configure' first.${NC}"
    echo "  You need: Access Key ID, Secret Access Key, Region (us-east-1)"
    echo "  See README.md 'Prerequisites Setup' for step-by-step instructions."
    exit 1
fi

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "us-east-1")
echo -e "  ${GREEN}[OK]${NC} AWS credentials valid (Account: $AWS_ACCOUNT, Region: $AWS_REGION)"

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------

echo ""
echo "Setting up Python virtual environment..."

if [ ! -d ".venv" ]; then
    uv venv
    echo -e "  ${GREEN}[OK]${NC} Virtual environment created (.venv/)"
else
    echo -e "  ${GREEN}[OK]${NC} Virtual environment already exists"
fi

# Activate
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
echo -e "  ${GREEN}[OK]${NC} Virtual environment activated"

# ---------------------------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------------------------

echo ""
echo "Installing Python dependencies..."
uv pip install -r requirements.txt -q
echo -e "  ${GREEN}[OK]${NC} Dependencies installed"

# ---------------------------------------------------------------------------
# 4. Create .env file
# ---------------------------------------------------------------------------

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "  ${GREEN}[OK]${NC} Created .env from template"
    echo -e "  ${YELLOW}NOTE: Edit .env to add your TAVILY_API_KEY (optional, for web search)${NC}"
else
    echo -e "  ${GREEN}[OK]${NC} .env already exists"
fi

# ---------------------------------------------------------------------------
# 5. Verify Bedrock access
# ---------------------------------------------------------------------------

echo ""
echo "Checking Bedrock model access..."

MODEL_ID="us.anthropic.claude-sonnet-4-6-v1:0"
if aws bedrock get-foundation-model --model-identifier "$MODEL_ID" --region "$AWS_REGION" &> /dev/null; then
    echo -e "  ${GREEN}[OK]${NC} Bedrock model access confirmed ($MODEL_ID)"
else
    echo -e "  ${YELLOW}WARNING: Could not verify Bedrock model access.${NC}"
    echo "  Go to https://console.aws.amazon.com/bedrock/home#/modelaccess"
    echo "  and enable 'Anthropic Claude Sonnet 4.6'"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "========================================="
echo -e "  ${GREEN}Setup complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv:  source .venv/bin/activate"
echo "  2. (Optional) Edit .env to add TAVILY_API_KEY"
echo "  3. Run locally:    python -m agents.orchestrator --query 'your question'"
echo "  4. Run UI:         streamlit run ui/app.py"
echo "  5. Deploy to AWS:  bash scripts/deploy.sh"
echo ""

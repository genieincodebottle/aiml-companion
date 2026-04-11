#!/bin/bash
#
# Smart Claims Processor - Pipeline Runner
#
# Usage:
#   ./scripts/run_pipeline.sh                    # Run demo (all sample claims)
#   ./scripts/run_pipeline.sh --claim FILE       # Process a single claim
#   ./scripts/run_pipeline.sh --test             # Run test suite only
#   ./scripts/run_pipeline.sh --dashboard        # Launch Streamlit dashboard
#   ./scripts/run_pipeline.sh --eval             # Run batch evaluation

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo -e "${GREEN}==============================${NC}"
echo -e "${GREEN} Smart Claims Processor${NC}"
echo -e "${GREEN}==============================${NC}"

# Check Python
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo -e "${RED}Error: Python not found. Install Python 3.11+${NC}"
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)

# Check .env
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env not found. Copying from .env.example${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env and add your GOOGLE_API_KEY${NC}"
fi

# Check dependencies
if ! $PYTHON -c "import langgraph" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    $PYTHON -m pip install -r requirements.txt -q
fi

# Create data directories
mkdir -p data/audit_logs data/sample_claims data/fraud_patterns

# Parse args
case "${1:-demo}" in
    --demo|demo)
        echo -e "\n${GREEN}Running all sample claims...${NC}\n"
        $PYTHON main.py --demo
        ;;
    --claim)
        if [ -z "${2:-}" ]; then
            echo -e "${RED}Error: --claim requires a file path${NC}"
            echo "Usage: $0 --claim data/sample_claims/auto_accident.json"
            exit 1
        fi
        echo -e "\n${GREEN}Processing claim: $2${NC}\n"
        $PYTHON main.py --claim "$2" --verbose
        ;;
    --test)
        echo -e "\n${GREEN}Running test suite...${NC}\n"
        $PYTHON -m pytest tests/ -v --tb=short
        ;;
    --dashboard)
        echo -e "\n${GREEN}Launching Streamlit dashboard...${NC}"
        echo -e "Open http://localhost:8501\n"
        $PYTHON -m streamlit run app.py --server.port 8501
        ;;
    --eval)
        echo -e "\n${GREEN}Running batch evaluation...${NC}\n"
        $PYTHON evaluation/run_eval.py
        ;;
    --help|-h)
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  --demo        Run all sample claims (default)"
        echo "  --claim FILE  Process a single claim JSON file"
        echo "  --test        Run the test suite"
        echo "  --dashboard   Launch Streamlit dashboard"
        echo "  --eval        Run batch evaluation"
        echo "  --help        Show this help"
        echo ""
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Run $0 --help for usage"
        exit 1
        ;;
esac

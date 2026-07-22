<div align="center">

# Multi-Agent AI App on AWS

### Build and deploy a production-ready multi-agent system on AWS using Bedrock AgentCore

**3 specialized AI agents + 1 orchestrator = Your AI research team**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![AWS AgentCore](https://img.shields.io/badge/AWS-Bedrock%20AgentCore-orange.svg)](https://aws.amazon.com/bedrock/agentcore/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Agents: 4](https://img.shields.io/badge/Agents-4-purple.svg)](#architecture)

</div>

> **Also available as a standalone repo:** [github.com/genieincodebottle/multi-agents-app-on-aws](https://github.com/genieincodebottle/multi-agents-app-on-aws). Both copies are kept in sync; this directory exists so the project is discoverable alongside the rest of `aiml-companion`.

---

## What You'll Build

A multi-agent system where 4 AI agents collaborate to research any topic and produce a comprehensive report:

| Agent | Role | What It Does |
|-------|------|-------------|
| **Orchestrator** | Supervisor | Breaks down your request, delegates to specialists, assembles final output |
| **Research Agent** | Web Researcher | Searches the web, gathers sources, summarizes findings |
| **Analyst Agent** | Data Analyst | Analyzes research data, identifies patterns, computes comparisons |
| **Writer Agent** | Report Writer | Takes research + analysis and produces a polished, structured report |

**Example prompt:**
> "Research the current state of AI agents in enterprise. Compare top frameworks (LangGraph, CrewAI, AutoGen, Strands). Include adoption trends and pricing."

**What you get:** A structured report with sourced research, comparative analysis, and actionable recommendations - all produced by collaborating agents.

---

## Architecture

<div align="center">
<img src="./images/architecture.svg" alt="Multi-Agent System Architecture" width="800"/>
</div>

---

## Why No Framework?

This project is intentionally built **from scratch without any agentic framework** (no LangGraph, no CrewAI, no AutoGen, no Strands).

**Why?** Because frameworks hide the core patterns behind abstractions. You learn the framework's API, not how agents actually work. This project shows you the raw mechanics:

| What You'll See | How It's Built |
|----------------|----------------|
| Agent orchestration | A Python function that generates a plan and loops through steps |
| Agent communication | Passing Python dicts between functions (local) or HTTPS calls (deployed) |
| Context threading | Manually accumulating each agent's output and passing it to the next |
| LLM calls | Direct API calls to Groq/Gemini/Bedrock - no wrapper layers |
| Tool use | Plain Python functions that return data |

**The entire multi-agent system is ~600 lines of Python.** Every line is readable and traceable.

> **Learn the patterns first, adopt a framework later.** Once you understand how orchestration, planning, context passing, and tool use work at the raw level, picking up LangGraph or CrewAI becomes much easier - you'll know exactly what the framework is doing under the hood.

---

## Two Learning Phases

This project has two phases. **You don't need AWS to start.**

| | Phase 1 - Learn Locally | Phase 2 - Deploy to AWS |
|--|------------------------|------------------------|
| **What** | Run all agents on your laptop as a single Python process | Deploy each agent to AWS Bedrock AgentCore (serverless microVMs) |
| **LLM Provider** | Groq or Gemini (free API key) | AWS Bedrock recommended (Claude Sonnet 4.6), but Groq/Gemini also work |
| **AgentCore** | Not used | Used - each agent runs in its own isolated microVM |
| **AWS Account** | Not needed | Required (with Bedrock model access) |
| **Agent Communication** | Direct Python function calls | A2A protocol over HTTPS |
| **Memory** | In-memory (lost on restart) | AgentCore Memory (persistent across sessions) |
| **Cost** | Free | ~$0.01-0.05 per request (mostly LLM token costs) |
| **Best for** | Learning, building, testing | Production, multi-user, auto-scaling |

> **Start with Phase 1.** Build and test your multi-agent system for free. When you're ready for production, Phase 2 adds AgentCore's auto-scaling, persistent memory, and enterprise-grade infrastructure.

---

## Quick Start (5 Minutes) - Phase 1

Get the multi-agent system running on your machine in 4 steps. **No AWS account needed** - we'll use Groq's free API to start.

### Step 1: Install Prerequisites

**Git** - Check if you have it:
```bash
git --version
```
If not installed:
- **Windows**: Download from [git-scm.com/downloads/win](https://git-scm.com/downloads/win). Run installer with default options. Reopen your terminal after.
- **Mac**: Run `xcode-select --install` in Terminal
- **Linux**: `sudo apt update && sudo apt install git`

**Python 3.10+** - Check if you have it:
```bash
python --version
```

If not installed, download from [python.org/downloads](https://python.org/downloads).
- **Windows**: Run installer. **Check "Add Python to PATH"** before clicking Install. Reopen your terminal after.
- **Mac**: Run the `.pkg` installer
- **Linux**: `sudo apt update && sudo apt install python3 python3-pip`

**uv** (fast Python package manager):
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip (any platform)
pip install uv
```

> **After installing each tool, close and reopen your terminal** so the new commands are available.

> **Which terminal to use?** Windows: open **PowerShell** (search "PowerShell" in Start menu). Mac: open **Terminal**. Linux: any terminal.

### Step 2: Clone and Install

```bash
git clone https://github.com/genieincodebottle/multi-agents-app-on-aws.git
cd multi-agents-app-on-aws
```

Create a virtual environment (keeps this project's packages separate from your system Python):
```bash
uv venv
```

Activate it:
```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Mac / Linux
source .venv/bin/activate
```

> You should see `(.venv)` at the start of your terminal prompt. This means the virtual environment is active.

Install dependencies:
```bash
uv pip install -r requirements.txt   # installs core dependencies
uv pip install groq                  # installs Groq LLM provider
```

### Step 3: Get a Free Groq API Key (2 Minutes)

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up with your Google or GitHub account (instant, no credit card)
3. Click **"Create API Key"**
4. Copy the key (starts with `gsk_`)

Now create your config file:
```bash
# Windows (PowerShell or Command Prompt)
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` in any text editor (Notepad, VS Code, etc.) and set these two lines:
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_paste_your_key_here
```

That's it. Save the file.

### Step 4: Run It

```bash
python -m agents.orchestrator --query "What is machine learning?"

# If "python" is not found, try "python3" (common on Mac/Linux)
python3 -m agents.orchestrator --query "What is machine learning?"
```

You should see output like:
```
Orchestrator: received query 'What is machine learning?'
Orchestrator: creating execution plan...
Orchestrator: step 1/2 - calling research agent
Research Agent: completed research with 3 sources
Orchestrator: step 2/2 - calling writer agent
Writer Agent: report complete
Orchestrator: pipeline complete in 4.5s

======================================================================
  FINAL REPORT
======================================================================
# Machine Learning: An Overview
...
```

**It works!** Try more queries:
```bash
# See each agent's detailed output
python -m agents.orchestrator --query "Compare Python vs Rust for AI" --verbose

# Run the web UI
uv pip install -r ui/requirements.txt
streamlit run ui/app.py
```

---

## Choose Your LLM Provider

This project supports 3 LLM providers. You already set up Groq above. Here's how to switch:

### Option 1: Groq (Default - Free)

The fastest option with a generous free tier. Already configured if you followed Quick Start.

| Detail | Value |
|--------|-------|
| **Cost** | Free (30 req/min, 14,400 req/day) |
| **Setup time** | 2 minutes |
| **Best model** | `openai/gpt-oss-120b` |
| **Get API key** | [console.groq.com/keys](https://console.groq.com/keys) |

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL_ID=openai/gpt-oss-120b
```

Other models: `openai/gpt-oss-120b` (Llama 3.3 workhorse), `llama-3.1-8b-instant` (fastest), `qwen/qwen3-32b`

### Option 2: Gemini (Free)

Google's Gemini models, also with a free tier.

| Detail | Value |
|--------|-------|
| **Cost** | Free (15 req/min, 1,500 req/day) |
| **Setup time** | 2 minutes |
| **Best model** | `gemini-3.6-flash` |
| **Get API key** | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

```bash
uv pip install google-genai
```

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL_ID=gemini-3.6-flash
```

Other models: `gemini-3.5-flash-lite` (faster, cheaper), `gemini-3.1-pro-preview` (best quality)

### Option 3: AWS Bedrock (Production-Grade)

Best quality with Claude Sonnet 4.6.6. Requires an AWS account with billing enabled.

| Detail | Value |
|--------|-------|
| **Cost** | ~$0.01-0.04 per request |
| **Setup time** | 15 minutes |
| **Best model** | Claude Sonnet 4.6 |
| **Cheapest model** | Nova Lite (~$0.001/request) |

```bash
uv pip install boto3 botocore
```

```env
LLM_PROVIDER=bedrock
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6
```

For Bedrock setup, see [AWS Setup Guide](#aws-setup-guide) below.

### Provider Comparison

| Provider | Cost | Speed | Best Model | Quality |
|----------|------|-------|------------|---------|
| **Groq** | Free | Fastest (~5s) | GPT-OSS 120B | Good |
| **Gemini** | Free | Medium (~17s) | Gemini 3.6 Flash | Good |
| **Bedrock (Nova Lite)** | ~$0.001/req | Medium (~15s) | Nova Lite | Good |
| **Bedrock (Claude Sonnet 4.6)** | ~$0.01-0.04/req | Medium (~15s) | Claude Sonnet 4.6 | Best |

> **Recommendation:** Start with **Groq** (free + fastest). Switch to **Bedrock + Claude Sonnet 4.6** when you want production quality.

---

## Deploy to AWS (AgentCore) - Phase 2

Once you've tested locally with Groq/Gemini, you can deploy to AWS AgentCore for production use. This gives you auto-scaling, persistent memory, and per-second billing.

> **Why AWS for deployment?** AgentCore runs each agent in an isolated microVM with auto-scaling, persistent conversation memory, and A2A (Agent-to-Agent) protocol support. These are production features that don't exist in local mode. You need an AWS account with Bedrock access for this phase.
>
> See [AWS Setup Guide](#aws-setup-guide) if you haven't set up AWS yet.

**AWS Services Used:**
- **Bedrock AgentCore Runtime** - Serverless agent hosting (auto-scales, per-second billing)
- **Amazon Bedrock** - Foundation models (Claude Sonnet 4.6)
- **AgentCore Memory** - Conversation history + long-term memory
- **AgentCore Gateway** - Tool management (web search, calculator)
- **IAM** - Permissions and security

**Estimated Cost:** ~$0.01-0.05 per research request (mostly LLM token costs)

### Option A: One-Command Deploy (Recommended)

```bash
uv pip install bedrock-agentcore-starter-toolkit
bash scripts/deploy.sh
```

The script will:
1. Configure IAM roles (if not exists)
2. Deploy each agent to AgentCore Runtime
3. Print the endpoint URLs
4. Run a health check

### Option B: Manual Deploy (Step by Step)

```bash
# 1. Deploy Research Agent
cd deploy/agentcore
agentcore configure -e research_agent.py
agentcore launch
# Note the ARN printed - you'll need it

# 2. Deploy Analyst Agent
agentcore configure -e analyst_agent.py
agentcore launch

# 3. Deploy Writer Agent
agentcore configure -e writer_agent.py
agentcore launch

# 4. Deploy Orchestrator (needs other agent ARNs)
# Edit orchestrator.py with the ARNs from steps 1-3
agentcore configure -e orchestrator.py
agentcore launch
```

### Option C: Terraform (Production-Grade)

```bash
cd deploy/terraform
terraform init
terraform plan -var="aws_region=us-east-1"
terraform apply -var="aws_region=us-east-1"
terraform output
```

### Cleanup

```bash
# Remove all deployed AgentCore resources
bash scripts/cleanup.sh

# Or with Terraform
cd deploy/terraform
terraform destroy -var="aws_region=us-east-1"
```

---

## AWS Setup Guide

<details>
<summary><strong>Click to expand - Only needed if you want to use AWS Bedrock or deploy to AgentCore</strong></summary>

### 1. Install AWS CLI v2

```bash
aws --version   # Check if already installed
```

If not installed:
- **Windows**: Download and run [AWSCLIV2.msi](https://awscli.amazonaws.com/AWSCLIV2.msi). Reopen your terminal after.
- **Mac**: Download and run [AWSCLIV2.pkg](https://awscli.amazonaws.com/AWSCLIV2.pkg)
- **Linux**:
  ```bash
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  unzip awscliv2.zip
  sudo ./aws/install
  ```

### 2. Create an AWS Account

1. Go to [aws.amazon.com/free](https://aws.amazon.com/free) and click **"Create a Free Account"**
2. Enter your email, choose an account name, and verify your email
3. Choose **"Personal"** account type
4. Enter your payment details (required, but Free Tier won't charge you for small usage)
5. Complete phone verification
6. Select the **"Basic Support - Free"** plan
7. Sign in to the [AWS Console](https://console.aws.amazon.com/)

### 3. Create an IAM User and Get Access Keys

1. Sign in to [AWS Console](https://console.aws.amazon.com/)
2. Search **"IAM"** in the top search bar, click it
3. Click **"Users"** in the left sidebar, then **"Create user"**
4. **User name**: `bedrock-agent-user` (or any name)
5. Click **"Next"**
6. Select **"Attach policies directly"**
7. Search and check these policies:
   - `AmazonBedrockFullAccess`
   - `IAMFullAccess` (needed for AgentCore deployment only)
8. Click **"Next"**, then **"Create user"**
9. Click on the user you just created
10. Go to the **"Security credentials"** tab
11. Scroll to **"Access keys"**, click **"Create access key"**
12. Select **"Command Line Interface (CLI)"**
13. Check the confirmation box, click **"Next"**, then **"Create access key"**
14. **Save both keys now** (you won't see the Secret again):
    - `Access key ID` (looks like: `AKIA...`)
    - `Secret access key` (looks like: `wJalrXUtnF...`)

### 4. Configure AWS CLI

```bash
aws configure
```

Enter your values one at a time:
```
AWS Access Key ID [None]: AKIA...YOUR_ACCESS_KEY...
AWS Secret Access Key [None]: wJalrXUtnF...YOUR_SECRET_KEY...
Default region name [None]: us-east-1
Default output format [None]: json
```

**Verify:**
```bash
aws sts get-caller-identity
```

You should see your account info. If you see an error, double-check your keys (trailing spaces are the most common mistake).

### 5. Enable Bedrock Models

1. Go to [Amazon Bedrock Model Access](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess)
2. Click **"Manage model access"** (or **"Modify model access"**)
3. Find **"Anthropic"** section, check **"Claude Sonnet 4.6"**
4. Click **"Request model access"** (or **"Save changes"**)
5. Wait for the status to show **"Access granted"** (usually instant)

> Make sure you're in the **us-east-1** region (check the top-right dropdown in AWS Console).

### 6. Update Your .env

```env
LLM_PROVIDER=bedrock
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6
```

> **Want to save money?** Use `BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0` (10x cheaper, great for testing).

> **Billing error?** If you see `INVALID_PAYMENT_INSTRUMENT`, add a payment method at [AWS Billing Console](https://console.aws.amazon.com/billing/) and retry after 2 minutes.

</details>

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | LLM provider: `groq`, `gemini`, or `bedrock` |
| `GROQ_API_KEY` | | Groq API key ([console.groq.com/keys](https://console.groq.com/keys)) |
| `GROQ_MODEL_ID` | `openai/gpt-oss-120b` | Groq model ID |
| `GEMINI_API_KEY` | | Gemini API key ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)) |
| `GEMINI_MODEL_ID` | `gemini-3.6-flash` | Gemini model ID |
| `AWS_REGION` | `us-east-1` | AWS region (Bedrock only) |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-sonnet-4-6` | Bedrock model ID |
| `TAVILY_API_KEY` | (optional) | Web search API key ([tavily.com](https://tavily.com) - free tier: 1000 searches/month) |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `MAX_RESEARCH_RESULTS` | `5` | Number of web search results per query |
| `MAX_AGENT_ITERATIONS` | `10` | Safety limit for agent reasoning loops |
| `AGENTCORE_MEMORY_ID` | (auto-created) | AgentCore Memory resource ID (for deployed mode) |

### Supported Bedrock Models

| Model | Model ID | Cost (Input/Output per 1M tokens) |
|-------|----------|-----------------------------------|
| Claude Sonnet 4.6 | `us.anthropic.claude-sonnet-4-6` | $3.00 / $15.00 |
| Claude Haiku 4.5 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | $0.80 / $4.00 |
| Amazon Nova 2 Lite | `us.amazon.nova-2-lite-v1:0` | $0.06 / $0.24 |
| Llama 4 Scout | `us.meta.llama4-scout-17b-instruct-v1:0` | $0.27 / $0.35 |

---

## How It Works

```
User: "Research AI agent frameworks and compare them"
  |
  v
Orchestrator receives request
  |
  |-- Orchestrator breaks down into subtasks:
  |     1. "Research current AI agent frameworks"
  |     2. "Analyze and compare the frameworks"
  |     3. "Write a comprehensive report"
  |
  |-- Step 1: Research Agent
  |     - Searches web, gathers sources, summarizes findings
  |
  |-- Step 2: Analyst Agent (receives research)
  |     - Compares frameworks, creates matrix, identifies trends
  |
  |-- Step 3: Writer Agent (receives research + analysis)
  |     - Structures into report with citations and recommendations
  |
  v
Orchestrator returns final report to user
```

---

## Project Structure

```
multi-agents-app-on-aws/
├── agents/                      # Agent source code
│   ├── config.py                # Shared config (LLM provider routing)
│   ├── orchestrator.py          # Supervisor that coordinates all agents
│   ├── research_agent.py        # Web research specialist
│   ├── analyst_agent.py         # Data analysis specialist
│   └── writer_agent.py          # Report writing specialist
│
├── tools/                       # Custom tools for agents
│   ├── web_search.py            # Tavily web search integration
│   └── calculator.py            # Math computation tool
│
├── deploy/                      # AWS deployment configs
│   ├── agentcore/               # AgentCore CLI deployment
│   └── terraform/               # Infrastructure as Code
│
├── ui/                          # Streamlit web frontend
│   └── app.py
│
├── scripts/                     # Helper scripts
│   ├── setup.sh                 # Initial setup
│   ├── deploy.sh                # Deploy all agents
│   ├── cleanup.sh               # Tear down resources
│   └── test_agents.sh           # Test agents
│
├── tests/                       # Unit tests (15 tests)
├── .env.example                 # Environment template
├── requirements.txt             # Python dependencies
└── pyproject.toml               # Project metadata
```

---

## Customization

### Adding a New Agent

1. Create the agent in `agents/`:

```python
# agents/my_custom_agent.py
from agents.config import call_llm

SYSTEM_PROMPT = """You are a specialist in [your domain].
Your job is to [specific task]."""

def run(input_data: dict) -> dict:
    result = call_llm(SYSTEM_PROMPT, input_data["query"])
    return {"result": result, "agent": "my_custom_agent"}
```

2. Register it in `agents/orchestrator.py`:

```python
AGENTS = {
    "research": research_agent,
    "analyst": analyst_agent,
    "writer": writer_agent,
    "my_custom": my_custom_agent,  # Add here
}
```

3. Update the orchestrator's system prompt to know about the new agent.

### Adding a New Tool

1. Create the tool in `tools/`:

```python
# tools/my_tool.py
def my_tool(param: str) -> str:
    """Description of what this tool does."""
    # Your tool logic
    return result
```

2. Add it to the agent that needs it (in `agents/research_agent.py` or whichever).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `GROQ_API_KEY NOT SET` | Get a free key at [console.groq.com/keys](https://console.groq.com/keys) and add to `.env` |
| `GEMINI_API_KEY NOT SET` | Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and add to `.env` |
| `groq: No module named 'groq'` | Run `uv pip install groq` |
| `google: No module named 'google.genai'` | Run `uv pip install google-genai` |
| `AccessDeniedException` on Bedrock | Enable model access in [Bedrock Console](https://console.aws.amazon.com/bedrock/home#/modelaccess) |
| `NoCredentialsError` | Run `aws configure` or set `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| `ThrottlingException` | You've hit rate limits. Wait a moment or request limit increase |
| Web search returns empty | Get a free Tavily API key at [tavily.com](https://tavily.com) and set `TAVILY_API_KEY` |
| `python: command not found` | Try `python3` instead (common on Mac/Linux). Or reinstall Python and check "Add to PATH" |
| `uv: command not found` | Install uv: see [Step 1](#step-1-install-prerequisites). Reopen terminal after installing |
| `.ps1 cannot be loaded because running scripts is disabled` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell, then try again |
| `(.venv) not showing in terminal` | You forgot to activate. Run `.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate` (Mac/Linux) |

---

## Learn More

- [AWS Bedrock AgentCore Docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/)
- [AgentCore SDK (GitHub)](https://github.com/aws/bedrock-agentcore-sdk-python)
- [AgentCore Samples](https://github.com/awslabs/amazon-bedrock-agentcore-samples)
- [Strands Agents Framework](https://github.com/strands-agents/sdk-python)
- [uv - Fast Python Package Manager](https://docs.astral.sh/uv/)
- [Build AI Systems Visually - AI/ML Companion](https://aimlcompanion.ai/)

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-agent`)
3. Commit your changes (`git commit -m 'Add new agent'`)
4. Push to the branch (`git push origin feature/new-agent`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built by [Rajesh Srivastava](https://github.com/genieincodebottle)**

[AI/ML Companion](https://aimlcompanion.ai/) | [YouTube](https://youtube.com/@genieincodebottle) | [Instagram](https://instagram.com/genieincodebottle)

</div>

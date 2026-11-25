# Agentic AI Projects

> **Learn AI/ML interactively at [AI-ML Companion](https://aimlcompanion.ai/)** - Guided walkthroughs, architecture decisions, hands-on challenges, and narrated overviews.

This folder holds eleven runnable agent projects. Each one builds a working system and is built around
one idea that is hard to get right in production: routing, disagreement between agents,
human approval, failure, trust, or measurement. Every project has its own README with
run steps and a full-system diagram at `docs/high-level-architecture.svg`.

The table follows the suggested order in [Where to start](#where-to-start).

| # | Project | Focus | Runs with no API key | Architecture | Learn |
|---|---|---|---|---|---|
| 1 | [Multi-Agent App on AWS](multi-agents-app-on-aws/) | Supervisor and specialists, then managed deployment | No, a free Groq or Gemini key | [Diagram](multi-agents-app-on-aws/docs/high-level-architecture.svg) | [Learn &rarr;](https://aimlcompanion.ai/module/cloudGenAI/multiAgentsOnAws) |
| 2 | [AI Agents Research System](ai-agents-project/) | Quality-gated routing and a revision loop | Yes, offline mode | [Diagram](ai-agents-project/docs/high-level-architecture.svg) | [Learn &rarr;](https://aimlcompanion.ai/module/aiAgents/agentsCapstone) |
| 3 | [Due Diligence Agent](due-diligence-agent/) | Parallel agents that disagree, fact-check and debate | Yes, offline mode | [Diagram](due-diligence-agent/docs/high-level-architecture.svg) | [Learn &rarr;](https://aimlcompanion.ai/module/aiAgents/dueDiligenceProject) |
| 4 | [Content Moderation](content-moderation-project/) | Full-stack moderation with a human review queue and memory | Yes, offline demo | [Diagram](content-moderation-project/docs/high-level-architecture.svg) | [Learn &rarr;](https://aimlcompanion.ai/module/aiAgents/contentModerationProject) |
| 5 | [Smart Claims Processor](smart-claims-processor/) | Pause and resume for human approval, durable checkpoints | No, a Gemini or Groq key | [Diagram](smart-claims-processor/docs/high-level-architecture.svg) | [Learn &rarr;](https://aimlcompanion.ai/module/aiAgents/smartClaimsProcessor) |
| 6 | [Multi-Agent Anatomy](multi-agent-anatomy/) | How a multi-agent request fails, broken on demand | Yes, replay mode | [Diagram](multi-agent-anatomy/docs/high-level-architecture.svg) | [Post &rarr;](https://aimlcompanion.ai/blog/production-multi-agent-genai-architecture-2026) |
| 7 | [AI Incident Commander](incident-commander/) | A bounded agent loop where code decides what is allowed | Yes, offline mode | [Diagram](incident-commander/docs/high-level-architecture.svg) | [README &rarr;](incident-commander/) |
| 8 | [Autonomous Migration Engineer](migration-engineer/) | Many agents in parallel, and catching reward hacking | Yes, stub mode | [Diagram](migration-engineer/docs/high-level-architecture.svg) | [README &rarr;](migration-engineer/) |
| 9 | [Mission Control](mission-control/) | The control plane under agents you did not write | Yes, offline mode | [Diagram](mission-control/docs/high-level-architecture.svg) | [README &rarr;](mission-control/) |
| 10 | [Hermes Ops Agent](hermes-ops-agent/) | Measuring whether an agent's learning loop pays off | Yes, recorded fixture | [Diagram](hermes-ops-agent/docs/high-level-architecture.svg) | [README &rarr;](hermes-ops-agent/) |
| 11 | [Dream-RSI Explorer](dream-rsi-explorer/) | Learning where a search loop should look next, by replay | No, a Gemini key (tests need none) | [Diagram](dream-rsi-explorer/docs/high-level-architecture.svg) | [Post &rarr;](https://aimlcompanion.ai/blog/dream-rsi-recursive-self-improvement-explained-2026) |

"Runs with no API key" means the project ships a deterministic stand-in for the model
(a rule engine, a replay of recorded runs, or a scripted worker) and labels its output
as such. It shows the orchestration, not model quality.

---

## The projects

### [Multi-Agent App on AWS](multi-agents-app-on-aws/)

An orchestrator hands a research request to three specialists (research, analyst,
writer) and assembles their report. It is plain Python with no agent framework, so you
see every handoff, and phase 2 deploys each agent to **Amazon Bedrock AgentCore**, where
they talk over the A2A protocol.

- **Stack:** Python, Groq, Gemini or Amazon Bedrock, Tavily search, Streamlit, Bedrock AgentCore
- **No API key:** no. Phase 1 needs a free Groq or Gemini key and no AWS account; without a Tavily key, search returns labelled placeholder results
- **Architecture:** [docs/high-level-architecture.svg](multi-agents-app-on-aws/docs/high-level-architecture.svg)

### [AI Agents Research System](ai-agents-project/)

A planner splits a topic into sub-topics, researchers run in parallel, a pure-Python
quality gate sends weak sources back, and a writer and reviewer loop until the report
passes or the revision cap is hit. It teaches **LangGraph routing and bounded loops**,
and its evaluation compares the same questions against a single agent, counting tokens
and latency as well as quality.

- **Stack:** Python, LangGraph, Gemini, Tavily search, Streamlit, SQLite cache
- **No API key:** yes. Offline mode runs the real graph against a rule-based stand-in for the model and for search
- **Architecture:** [docs/high-level-architecture.svg](ai-agents-project/docs/high-level-architecture.svg)

### [Due Diligence Agent](due-diligence-agent/)

Four analysts (financial, news, competitive, risk) research a company in parallel, a
fact-check pass verifies their claims, and a lead analyst resolves the contradictions
into a report with a risk rating. The key idea is **what to do when agents disagree**:
the offline fixtures disagree on purpose, so the debate runs and one claim stays
unverifiable.

- **Stack:** Python, LangGraph, Gemini, Streamlit, Docker
- **No API key:** yes. An offline provider with fixtures that contradict each other
- **Architecture:** [docs/high-level-architecture.svg](due-diligence-agent/docs/high-level-architecture.svg)

### [Content Moderation](content-moderation-project/)

A full-stack moderation platform: six agents review each post, short posts take a
single-pass fast path, and edge cases go to a priority queue for human moderators, with
appeals and six user roles. It teaches **human-in-the-loop review, agent memory and
guardrails** in one app.

- **Stack:** Python, FastAPI, LangGraph, Gemini, ChromaDB, SQLite, React + MUI
- **No API key:** yes for the CLI demo, which runs the six-agent graph against a rule-based stand-in. The full stack needs a Gemini key
- **Architecture:** [docs/high-level-architecture.svg](content-moderation-project/docs/high-level-architecture.svg)

### [Smart Claims Processor](smart-claims-processor/)

Insurance claims pass through seven agents, with a CrewAI crew for fraud inside one
LangGraph node. Risky claims **pause mid-pipeline** with LangGraph's `interrupt()`, and
a human reviewer's decision resumes the run from a checkpoint saved in SQLite. It also
covers token caps, PII masking, LLM-as-judge evaluation and US and India rule profiles.

- **Stack:** Python, FastAPI, LangGraph, CrewAI, Gemini or Groq, ChromaDB, SQLite, React + MUI
- **No API key:** no, the app needs a Gemini or Groq key. The tests mock the model and need none
- **Architecture:** [docs/high-level-architecture.svg](smart-claims-processor/docs/high-level-architecture.svg)

### [Multi-Agent Anatomy](multi-agent-anatomy/)

An order-support request in 8 stages and 5 agents, with six toggles that break it:
kill an agent, slow a tool, fail a saga step, serve a stale policy passage, turn off
prompt caching, attempt a cross-tenant read. The key idea is that **a green trace does
not mean a right answer**. The main view is a trace waterfall, and there is no agent
framework.

- **Stack:** Python, FastAPI, SQLite, Gemini, React + Vite
- **No API key:** yes. Seven recorded traces, and a replay client that builds answers from the real tool output
- **Architecture:** [docs/high-level-architecture.svg](multi-agent-anatomy/docs/high-level-architecture.svg)

### [AI Incident Commander](incident-commander/)

Seven agents diagnose and fix faults in a simulated microservice stack inside one
bounded loop, and every high-risk action stops for a human. The rule it teaches is
**the model decides what to do, code decides what is allowed**, and its ten scenarios
include an unfixable fault and a false alarm, where the right answer is to hand over or
do nothing.

- **Stack:** Python, FastAPI, LangGraph, Gemini, React + Vite + MUI, optional Postgres, Docker Compose
- **No API key:** yes. Offline mode is a rule-based agent that reads the same telemetry, with a banner in the console
- **Architecture:** [docs/high-level-architecture.svg](incident-commander/docs/high-level-architecture.svg)

### [Autonomous Migration Engineer](migration-engineer/)

An agent applies one code migration across many repositories: it edits, runs the tests,
fixes what breaks, prepares a pull request and waits for approval. The key idea is
**reward hacking**: scored on "tests pass", an agent can weaken or delete a test, and
three independent checks catch that.

- **Stack:** Python, FastAPI, LangGraph, Claude Agent SDK or Gemini, React + MUI, Docker Compose
- **No API key:** yes. Stub mode is a scripted worker that uses the same tools, budget, reviewer and approval gate
- **Architecture:** [docs/high-level-architecture.svg](migration-engineer/docs/high-level-architecture.svg)

### [Mission Control](mission-control/)

The platform under agents you did not write: a gateway every model call goes through, a
policy engine for tool calls, a runtime that checkpoints and survives restarts, a
reviewer gate with **maker-checker**, and an evaluation gate for prompt changes. It is
proven on a six-agent anti-money-laundering investigation squad.

- **Stack:** Python, FastAPI, LangChain with Gemini, React, Docker Compose
- **No API key:** yes. Offline mode runs the whole platform; a Gemini key makes it live
- **Architecture:** [docs/high-level-architecture.svg](mission-control/docs/high-level-architecture.svg)

### [Hermes Ops Agent](hermes-ops-agent/)

Runs [Hermes Agent](https://github.com/NousResearch/hermes-agent) with hardened config
and skills, then checks its claim that a task takes less work once Hermes has written a
skill for it, by reading Hermes' own SQLite store. It teaches **measuring an agent instead of
trusting its claims**, and includes a repeatable prompt-injection probe.

- **Stack:** Python standard library CLI, SQLite, Hermes Agent for real runs
- **No API key:** yes. The measurement runs on a recorded fixture database, with no Hermes install
- **Architecture:** [docs/high-level-architecture.svg](hermes-ops-agent/docs/high-level-architecture.svg)

### [Dream-RSI Explorer](dream-rsi-explorer/)

Implements Dream-RSI on one of the paper's benchmarks, packing 26 circles in a square.
Gemini proposes layouts, and the rule that decides **where to search next** is improved
by replaying recorded searches instead of paying for new ones.

- **Stack:** Python, Gemini (google-genai), uv
- **No API key:** no, the search calls Gemini. The tests use a fake client and need no key
- **Architecture:** [docs/high-level-architecture.svg](dream-rsi-explorer/docs/high-level-architecture.svg)

---

## Where to start

From the simplest shape to the hardest questions about trust:

1. **[Multi-Agent App on AWS](multi-agents-app-on-aws/)**: the basic supervisor and
   specialists shape in plain Python, where every handoff is a function call.
2. **[AI Agents Research System](ai-agents-project/)**: the same kind of task on
   LangGraph, adding parallel fan-out, a routing gate and a bounded revision loop.
3. **[Due Diligence Agent](due-diligence-agent/)**: parallel agents whose answers
   conflict, and how to settle a conflict instead of hiding it.
4. **[Content Moderation](content-moderation-project/)**: the first full-stack app,
   with a human review queue, memory and guardrails.
5. **[Smart Claims Processor](smart-claims-processor/)**: human approval as a paused,
   checkpointed run that resumes, plus evaluation with an LLM judge.
6. **[Multi-Agent Anatomy](multi-agent-anatomy/)**: what goes wrong once the happy path
   works: partial failure, budgets, undos, caching and silent wrong answers.
7. **[AI Incident Commander](incident-commander/)**: an agent that acts on a system,
   with hard limits in code and an evaluation that includes cases it must not fix.
8. **[Autonomous Migration Engineer](migration-engineer/)**: many agents at once, and an
   agent that games its own score.
9. **[Mission Control](mission-control/)**: the platform layer that governs a fleet of
   agents built by someone else.

Two projects stand to the side and fit any time after the first two:

- **[Hermes Ops Agent](hermes-ops-agent/)**: operating and measuring an existing agent
  rather than building one.
- **[Dream-RSI Explorer](dream-rsi-explorer/)**: a research idea, an agent loop that
  improves how it searches.

---

## Shared conventions

- **Keys live in `.env`, which every project git-ignores.** Most projects ship a
  `.env.example` to copy; Dream-RSI Explorer has you create `.env` with one line.
- **Most projects run with no key.** Eight of the eleven ship an offline, replay or
  stub mode, as listed in the table. The three that need a key mock or fake the model
  in their tests.
- **Gemini is the default model provider** in most projects. Each README names the
  environment variable it reads.
- **Python is managed with [uv](https://docs.astral.sh/uv/) in most projects**; a few
  use `pip install -r requirements.txt` in a virtual environment. Follow the commands in
  each README.

## License

Most projects are MIT licensed. Check each project's LICENSE file or README.

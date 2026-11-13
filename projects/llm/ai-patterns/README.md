# AI Reasoning Patterns

23 hands-on Jupyter notebooks demonstrating advanced AI reasoning and agentic patterns. Each notebook is self-contained and runs on Google Colab (no local setup needed).

## Quick Start

Click any notebook below to open it in Google Colab, or run locally with `jupyter notebook`.

## Patterns

| # | Pattern | Notebook | Description |
|---|---------|----------|-------------|
| 1 | Chain-of-Thought | `Chain-of-Thought.ipynb` | Step-by-step reasoning before answering |
| 2 | Tree of Thought | `Tree-of-Thought.ipynb` | Explore multiple reasoning branches |
| 3 | Graph of Thoughts | `Graph-of-Thoughts.ipynb` | Non-linear reasoning with graph structure |
| 4 | Skeleton of Thought | `Skeleton-of-Thought.ipynb` | Generate outline first, then fill in details |
| 5 | Chain of Verification | `Chain-of-Verification.ipynb` | Verify answers with follow-up questions |
| 6 | ReAct | `ReAct.ipynb` | Reason + Act loop with tool use |
| 7 | Reflexion | `Reflexion.ipynb` | Self-reflection to improve responses |
| 8 | Self-Refine | `Self-Refine.ipynb` | Iterative self-improvement of outputs |
| 9 | Recursive Criticism | `Recursive-Criticism-and-Improvement.ipynb` | Critique and revise in a loop |
| 10 | Least-to-Most | `Least-to-Most-Prompting.ipynb` | Decompose into sub-problems, solve incrementally |
| 11 | Decomposed Prompting | `Decomposed-Prompting.ipynb` | Break complex tasks into simpler sub-tasks |
| 12 | Plan and Solve | `Plan-and-Solve.ipynb` | Create a plan, then execute step-by-step |
| 13 | Reasoning via Planning | `Reasoning-via-Planning.ipynb` | Use planning as a reasoning strategy |
| 14 | Meta-Prompting | `Meta-Prompting.ipynb` | LLM generates its own prompts |
| 15 | RAG | `RAG.ipynb` | Retrieval-Augmented Generation |
| 16 | Toolformer | `Toolformer.ipynb` | LLM learns when and how to use tools |
| 17 | Auto Reasoning + Tools | `Automatic-Reasoning-and-Tool-Use.ipynb` | Automatic tool selection and reasoning |
| 18 | Multi-Agent Debate | `Multi-Agent-Debate.ipynb` | Multiple agents argue to reach better answers |
| 19 | Orchestrator-Worker | `Orchestrator-Worker.ipynb` | One agent coordinates, others execute |
| 20 | Generative Agents | `Generative-Agents.ipynb` | Simulated agents with memory and reflection |
| 21 | Self-Evolving Agent | `Self-Evolving-Agent.ipynb` | Agent that improves itself over time |
| 22 | Language Agent Tree Search | `Language-Agent-Tree-Search.ipynb` | Tree search for optimal agent actions |
| 23 | Model Context Protocol | `Model-Context-Protocol.ipynb` | MCP standard for LLM tool interoperability |

## Reference

See `ai-patterns.pdf` for a visual guide covering all patterns.

## Run Locally (Optional)

```bash
pip install jupyter google-genai chromadb
jupyter notebook
```

You'll need a free Google Gemini API key: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Set it once as an environment variable and every notebook picks it up:

```bash
export GOOGLE_API_KEY="your-key"     # or GEMINI_API_KEY
```

Each notebook still prompts with `getpass` if the variable is unset, so nothing
breaks in Colab. The fallback exists because prompting *only* meant the
notebooks could not be run non-interactively (nbconvert, papermill, CI) and you
retyped the key once per notebook.

### Model and SDK notes

These notebooks call **`gemini-flash-latest`**, a rolling alias rather than a
pinned version. That is a deliberate trade, and the reason is instructive.

They previously pinned `gemini-2.0-flash` and `gemini-2.0-flash-exp`, both of
which Google has since **retired**. Every one of the 23 notebooks failed with:

```
404 This model models/gemini-2.0-flash is no longer available.
    Please update your code to ...
```

A pinned version is the right default for a published result you need to
reproduce exactly. For a teaching repo that has to keep *running*, it is a
maintenance liability with a silent fuse: nothing warns you, and one day every
notebook 404s at once. The alias trades exact reproducibility for staying alive.

The trade has a second side: the alias moves to newer, larger models, and speed
moves with it. These patterns are multi-call by construction - debate rounds,
tree-search branches, verification chains - so a notebook makes 50-150
sequential requests and per-call latency decides how long it runs. Measured on
one identical prompt (median of 3 calls, 2026-11-13):

| model | resolves to | latency per call |
|---|---|---|
| `gemini-flash-latest` | `gemini-3.8-flash` | 10.1s |
| `gemini-3.6-flash` | `gemini-3.6-flash` | 6.6s |
| `gemini-flash-lite-latest` | `gemini-3.5-flash-lite` | 1.6s |

`Multi-Agent-Debate.ipynb` makes about 140 calls, so on the default alias it
takes around 25 minutes; most notebooks finish in a few. To trade some answer
quality for speed, change the model string in the agent class to
`gemini-flash-lite-latest`. If you need reproducible outputs, pin whatever
`gemini-flash-latest` currently resolves to and accept the maintenance.

All notebooks use the **`google-genai`** SDK (`from google import genai`).
They were moved off `google.generativeai`, which Google deprecated and no longer
updates. The call shape is `client.models.generate_content(model=name, contents=...)`
in place of the old `genai.GenerativeModel(name).generate_content(...)`, and
`RAG.ipynb` embeds with `client.models.embed_content(...)`.

The client is created with `http_options={"api_version": "v1beta"}`. From
google-genai 3.0 the SDK talks to the stable `v1` API by default, and `v1` does
not serve the `-latest` aliases or preview models: `gemini-flash-latest` returns
`404 ... is not found for API version v1`. Pinning `v1beta` keeps the alias working.

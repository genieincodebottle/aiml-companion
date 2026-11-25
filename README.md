# AI/ML Learning Companion

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Projects](https://img.shields.io/badge/Projects-33-orange)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

> **Learn AI/ML interactively at [AI-ML Companion](https://aimlcompanion.ai/)** - Guided walkthroughs, architecture decisions, hands-on challenges, and narrated overviews.

<a href="https://aimlcompanion.ai/"><img src="https://raw.githubusercontent.com/genieincodebottle/generative-ai/main/images/aiml-companion-tour.gif" width="80%" alt="AI-ML Companion tour"></a>

> 33 end-to-end projects covering the full spectrum, from classical ML to production LLM and agent systems. Most are the runnable companion to a module or blog post on the platform, and most follow a common **industry best-practice structure**.

---

## Contents

- [About the AI-ML Companion Platform](#about-the-ai-ml-companion-platform)
- [Projects](#projects) - all 33, grouped by domain, each with its app module
- [Projects used in more than one module](#projects-used-in-more-than-one-module)
- [Project Details](#project-details) - what each one actually does
- [Industry Best-Practice Project Structure](#industry-best-practice-project-structure)
- [Key Principles](#key-principles)
- [Quick Start](#quick-start)
- [Learning Path (Recommended Order)](#learning-path-recommended-order)
- [Repository Structure](#repository-structure)

---

## About the AI-ML Companion Platform

These 33 projects are the hands-on companion to **[AI-ML Companion](https://aimlcompanion.ai)**, an interactive platform for learning AI and machine learning by watching it work. Instead of passive video, every concept is taught through **live, animated visualizations** and runnable code: gradient descent stepping down a loss surface, attention weights lighting up across tokens, a network training in real time.

- **32 tracks, 500+ modules** from Python and math foundations to deep learning, LLMs, agentic AI, MLOps, and cloud GenAI.
- **Interactive visualizations**, not slides. See the mechanism move, then experiment with it.
- **Runnable Python** in the browser, quizzes, and progress tracking.
- **Trilingual narration** (English, Hindi, Spanish) with technical terms kept in English.
- **PWA**, works on phone and desktop, installable and offline-capable.
- **Free tier:** 8 complete tracks (Foundations, Python for ML, Jupyter, Linear Algebra, ML Algorithms, Git, AI Tools, Interview Q&A) plus the first module of every Premium track. Premium unlocks the full catalog.

**Curated learning paths:** Data Science and AI Engineering guide you end to end.

**Explore:** [Curriculum](https://aimlcompanion.ai/curriculum) · [Roadmap](https://aimlcompanion.ai/roadmap) · [Blog](https://aimlcompanion.ai/blog)

### FAQ

**Is AI-ML Companion free?** Yes, there is a free tier: 8 complete tracks plus the opening module of every Premium track. Premium unlocks all 32 tracks and 500+ modules.

**How is it different from video courses?** It is interactive. Concepts are taught through live visualizations and runnable code rather than passive video, so you can watch algorithms operate and experiment directly.

**Who is it for?** Beginners wanting an intuition-first on-ramp, engineers moving into ML / MLOps / LLM / agentic AI roles, and anyone preparing for AI/ML interviews.

**What languages are supported?** English, Hindi, and Spanish, with technical terminology kept in English.

---

## Projects

The last column is the page on the platform that teaches the project. A project
without its own module links to the blog post or lesson it accompanies.

### Machine Learning

| # | Project | Domain | Difficulty | Key Tech | App module |
|---|---|---|---|---|---|
| 1 | [IPL Match Predictor](projects/machine-learning/ipl-match-predictor/) | Data Science / EDA | Beginner-Intermediate | Pandas, Plotly, Scikit-learn | ML Algorithms: [Project: IPL Match Predictor](https://aimlcompanion.ai/module/mlAlgorithms/iplProject) |
| 2 | [Algorithm Showdown](projects/machine-learning/algorithm-showdown/) | Classical ML / Interpretability | Intermediate | Scikit-learn, XGBoost, SHAP | ML Algorithms: [Project: Algorithm Showdown](https://aimlcompanion.ai/module/mlAlgorithms/mlAlgorithmsCapstone) |
| 3 | [Rent Price Explainer](projects/machine-learning/rent-price-explainer/) | Regression Diagnostics | Intermediate | Statsmodels, Scikit-learn, SHAP | ML Algorithms: [Project: Rent Price Explainer](https://aimlcompanion.ai/module/mlAlgorithms/rentPriceExplainer) |
| 4 | [Support Ticket Triage](projects/machine-learning/support-ticket-triage/) | Text Classification / Calibration | Intermediate | Scikit-learn, SciPy, Pandas | ML Algorithms: [Project: Support Ticket Triage](https://aimlcompanion.ai/module/mlAlgorithms/supportTicketTriage) |
| 5 | [Cross-Validation Traps](projects/machine-learning/cross-validation-traps/) | Model Validation | Intermediate-Advanced | Scikit-learn, Pandas, NumPy | No project module yet. Related lesson: ML Algorithms: [Cross-Validation](https://aimlcompanion.ai/module/mlAlgorithms/crossVal) |
| 6 | [Lapse Prediction](projects/machine-learning/lapse-prediction/) | Ordinal Modelling / Insurance | Advanced | LightGBM, XGBoost, Statsmodels | No app module yet, see the [README](projects/machine-learning/lapse-prediction/) |
| 7 | [Credit Risk Pipeline](projects/machine-learning/credit-risk-pipeline/) | Feature Engineering / Production ML | Advanced | Scikit-learn, FastAPI, Docker | ML Pipeline: [Project: Credit Risk Pipeline](https://aimlcompanion.ai/module/mlPipeline/mlPipelineCapstone) |

### Computer Vision

| # | Project | Domain | Difficulty | Key Tech | App module |
|---|---|---|---|---|---|
| 8 | [Visual Defect Triage](projects/computer-vision/visual-defect-triage/) | Computer Vision / Calibration | Advanced | ViT, NumPy, Pydantic | Computer Vision: [Project: Visual Defect Triage](https://aimlcompanion.ai/module/computerVision/cvVitCapstone) |
| 9 | [Site Safety Monitor](projects/computer-vision/site-safety-monitor/) | Computer Vision / Edge Inference | Advanced | YOLO, ByteTrack, NumPy | Computer Vision: [Project: Site Safety Monitor](https://aimlcompanion.ai/module/computerVision/cvYoloCapstone) |

### Deep Learning

| # | Project | Domain | Difficulty | Key Tech | App module |
|---|---|---|---|---|---|
| 10 | [Deep Learning Classifier](projects/deep-learning/deep-learning-project/) | Computer Vision / DL | Intermediate-Advanced | PyTorch, TorchVision | Deep Learning: [Project: From Pixels to Production](https://aimlcompanion.ai/module/deepLearning/dlCapstone) |

### LLM

| # | Project | Domain | Difficulty | Key Tech | App module |
|---|---|---|---|---|---|
| 11 | [GraphRAG Supply Chain](projects/llm/graphrag-supply-chain/) | Graph + Vector Retrieval | Advanced | Neo4j, Gemini, FastAPI, Streamlit | LLM: [Project: GraphRAG Supply Chain](https://aimlcompanion.ai/module/llm/graphRagSupplyChain) |
| 12 | [RAG Expert Assistant](projects/llm/rag-expert-assistant/) | Retrieval-Augmented Generation | Advanced | LangChain, ChromaDB | LLM: [Project: RAG Expert Assistant](https://aimlcompanion.ai/module/llm/ragExpertAssistant) |
| 13 | [AI Reasoning Patterns](projects/llm/ai-patterns/) | Prompting / Reasoning Patterns | Intermediate | Jupyter, Colab, Gemini | Blog: [The Top 16 GenAI Patterns, agentic vs non-agentic](https://aimlcompanion.ai/blog/top-25-genai-patterns-agentic-non-agentic-2026) |
| 14 | [Muse Glimmer Lab](projects/llm/muse-glimmer-lab/) | Open-Weight Model Internals | Advanced | vLLM, OpenAI-compatible API | Blog: [Meta's Muse Glimmer, an agentic AI model](https://aimlcompanion.ai/blog/meta-muse-glimmer-explained-2026) |
| 15 | [LLM Judge Lifecycle](projects/llm/llm-judge-lifecycle/) | LLM Evaluation / EvalOps | Advanced | Gemini, FastAPI, Streamlit, Pytest | Blog: [Implementing the LLM judge lifecycle from the Netflix paper](https://aimlcompanion.ai/blog/implementing-the-llm-judge-lifecycle-2026) |
| 16 | [Jev Decision Lab](projects/llm/jev-decision-lab/) | Typed Decision Models / Calibration | Intermediate-Advanced | TypeSafe Jev (simulated or live), Pytest | Blog: [Jev and the comeback of classifiers](https://aimlcompanion.ai/blog/jev-system-one-model-classifier-returns-2026) |

### Agentic AI

All 11 agentic projects, what each one teaches and a suggested order: [projects/agentic-ai/README.md](projects/agentic-ai/README.md).
Every agentic project has a README with run steps and a full-system diagram at `docs/high-level-architecture.svg`.

| # | Project | Domain | Difficulty | Key Tech | App module |
|---|---|---|---|---|---|
| 17 | [AI Agents Research System](projects/agentic-ai/ai-agents-project/) | LLM Agent Orchestration | Advanced | LangGraph, Gemini, Tavily, Streamlit | Agentic AI: [Project: Multi-Agent Research System](https://aimlcompanion.ai/module/aiAgents/agentsCapstone) |
| 18 | [Content Moderation](projects/agentic-ai/content-moderation-project/) | Multi-Agent Moderation / Human Review | Advanced | LangGraph, FastAPI, ChromaDB, React | Agentic AI: [Project: Content Moderation System](https://aimlcompanion.ai/module/aiAgents/contentModerationProject) |
| 19 | [Due Diligence Agent](projects/agentic-ai/due-diligence-agent/) | Multi-Agent Research | Advanced | LangGraph, Gemini, Streamlit | Agentic AI: [Project: Due Diligence Analyst](https://aimlcompanion.ai/module/aiAgents/dueDiligenceProject) |
| 20 | [Smart Claims Processor](projects/agentic-ai/smart-claims-processor/) | Multi-Agent Insurance Claims | Advanced | LangGraph, CrewAI, Gemini or Groq, FastAPI, React | Agentic AI: [Project: Smart Claims Processor](https://aimlcompanion.ai/module/aiAgents/smartClaimsProcessor) |
| 21 | [Multi-Agent Anatomy](projects/agentic-ai/multi-agent-anatomy/) | Multi-Agent Failure Modes | Advanced | FastAPI, React, Gemini, no framework | Blog: [Inside a Production Multi-Agent GenAI System](https://aimlcompanion.ai/blog/production-multi-agent-genai-architecture-2026) |
| 22 | [Multi-Agents App on AWS](projects/agentic-ai/multi-agents-app-on-aws/) | Cloud GenAI / Managed Agents | Advanced | Groq, Gemini or Bedrock, Bedrock AgentCore | AI on Cloud: [Project: Multi-Agents App on AWS](https://aimlcompanion.ai/module/cloudGenAI/multiAgentsOnAws) |
| 23 | [Hermes Ops Agent](projects/agentic-ai/hermes-ops-agent/) | Agent Operations / Measurement | Intermediate-Advanced | Hermes, SQLite, Pytest | AI Tools: [Hermes Agent: Self-Hosted Agent with Memory and Skills](https://aimlcompanion.ai/module/aiTools/hermesAgent) |
| 24 | [AI Incident Commander](projects/agentic-ai/incident-commander/) | Incident Response / Guardrails in Code | Advanced | FastAPI, LangGraph, React, Gemini or offline mode | Agentic AI: [Project: AI Incident Commander](https://aimlcompanion.ai/module/aiAgents/incidentCommanderProject) |
| 25 | [Mission Control](projects/agentic-ai/mission-control/) | Agent Control Plane / Governance | Advanced | FastAPI, React, Gemini or offline mode | Agentic AI: [Project: Agent Control Plane (Mission Control)](https://aimlcompanion.ai/module/aiAgents/missionControlProject) |
| 26 | [Autonomous Migration Engineer](projects/agentic-ai/migration-engineer/) | Coding Agents / Reward Hacking | Advanced | LangGraph, Claude Agent SDK or Gemini, FastAPI, React | Agentic AI: [Project: Autonomous Migration Engineer](https://aimlcompanion.ai/module/aiAgents/migrationEngineerProject) |
| 27 | [Dream-RSI Explorer](projects/agentic-ai/dream-rsi-explorer/) | Search Policy / Self-Improvement | Advanced | Gemini, uv, Pytest | Blog: [How Dream-RSI lets an AI agent improve how it searches](https://aimlcompanion.ai/blog/dream-rsi-recursive-self-improvement-explained-2026) |

### Inference Engineering

| # | Project | Domain | Difficulty | Key Tech | App module |
|---|---|---|---|---|---|
| 28 | [Inference Serving Starter](projects/inference-engineering/inference-serving-starter/) | Serving / Benchmarking / Capacity | Advanced | FastAPI engine simulator, NumPy, Pandas, Pytest | Inference Engineering: [Project: Serve, Benchmark and Cost an Open Model](https://aimlcompanion.ai/module/inferenceEngineering/inferenceServingCapstone) |

### MLOps

| # | Project | Domain | Difficulty | Key Tech | App module |
|---|---|---|---|---|---|
| 29 | [Model Serving Platform](projects/mlops/model-serving-platform/) | Model Deployment / Infrastructure | Advanced | FastAPI, Docker, Prometheus, GitHub Actions | MLOps & Deployment: [Project: Model Serving Platform](https://aimlcompanion.ai/module/mlOps/mlopsCapstone) |

### Forward Deployment

| # | Project | Domain | Difficulty | Key Tech | App module |
|---|---|---|---|---|---|
| 30 | [FDE Engagement Starter](projects/forward-deployment/fde-engagement-starter/) | Forward Deployment Engineering | Intermediate-Advanced | FastAPI, Pandas, Pytest, Docker | Forward Deployment Engineering: [Playbook: FDE Portfolio Minimum Viable Artifact](https://aimlcompanion.ai/module/forwardDeployment/fdePortfolioMVA) |

### Data Engineering

| # | Project | Domain | Difficulty | Key Tech | App module |
|---|---|---|---|---|---|
| 31 | [Star Schema Modelling](projects/data-engineering/star-schema-modelling/) | Dimensional Modelling / Data Tests | Intermediate | dbt, DuckDB, Pytest | Data Engineering: [Project: Model a Star Schema](https://aimlcompanion.ai/module/dataEngineering/deStarSchemaProject) |
| 32 | [Event Log Replay](projects/data-engineering/event-log-replay/) | Streaming / Delivery Semantics | Advanced | DuckDB, Kafka API (Redpanda, optional), Pytest | Data Engineering: [Project: Replay the Log](https://aimlcompanion.ai/module/dataEngineering/deReplayProject) |
| 33 | [Incremental Warehouse](projects/data-engineering/incremental-warehouse/) | Incremental Pipelines / Orchestration | Advanced | SQLite, DuckDB, Parquet, Pytest | Data Engineering: [Capstone: Build a Warehouse](https://aimlcompanion.ai/module/dataEngineering/dataEngCapstone) |

---

## Projects used in more than one module

The Forward Deployment Engineering track reuses several projects as engagement
material, on top of the module listed in the table above.

| Project | Also used in |
|---|---|
| [RAG Expert Assistant](projects/llm/rag-expert-assistant/) | [FDE Portfolio Minimum Viable Artifact](https://aimlcompanion.ai/module/forwardDeployment/fdePortfolioMVA), [7-Day Embedded Engagement Simulation](https://aimlcompanion.ai/module/forwardDeployment/fdeEmbeddedSim) |
| [Due Diligence Agent](projects/agentic-ai/due-diligence-agent/) | [FDE Portfolio Minimum Viable Artifact](https://aimlcompanion.ai/module/forwardDeployment/fdePortfolioMVA) |
| [Smart Claims Processor](projects/agentic-ai/smart-claims-processor/) | [FDE Portfolio Minimum Viable Artifact](https://aimlcompanion.ai/module/forwardDeployment/fdePortfolioMVA), [7-Day Embedded Engagement Simulation](https://aimlcompanion.ai/module/forwardDeployment/fdeEmbeddedSim) |
| [Multi-Agents App on AWS](projects/agentic-ai/multi-agents-app-on-aws/) | [FDE Portfolio Minimum Viable Artifact](https://aimlcompanion.ai/module/forwardDeployment/fdePortfolioMVA) |
| [Model Serving Platform](projects/mlops/model-serving-platform/) | [FDE Portfolio Minimum Viable Artifact](https://aimlcompanion.ai/module/forwardDeployment/fdePortfolioMVA) |
| [FDE Engagement Starter](projects/forward-deployment/fde-engagement-starter/) | [7-Day Embedded Engagement Simulation](https://aimlcompanion.ai/module/forwardDeployment/fdeEmbeddedSim) |

---

## Project Details

### Machine Learning

#### IPL Dataset Analysis - End-to-End EDA

Comprehensive analysis of 17 IPL seasons with interactive visualizations, hypothesis testing, feature engineering, and predictive modeling.

**Highlights**

- 1000+ matches
- Plotly interactive charts
- Hypothesis testing
- RF + GB models

[Interactive Walkthrough](https://aimlcompanion.ai/module/mlAlgorithms/iplProject)

---

#### ML Algorithms - Medical Diagnostic Classifier

Compare 6 ML algorithms on real clinical data with cost-sensitive threshold tuning (~95% malignant recall) and SHAP explainability for regulatory review.

**Highlights**

- 6 algorithms compared
- XGBoost AUC ~0.994
- SHAP reports
- Threshold tuning

[Interactive Walkthrough](https://aimlcompanion.ai/module/mlAlgorithms/mlAlgorithmsCapstone)

---

#### Rent Price Explainer - Specification Beats Algorithm

A rent model where the usual instinct, reach for a stronger learner, is the wrong move. OLS diagnostics (RESET, Breusch-Pagan, VIF, Cook's distance) locate a missing interaction and a log-scale problem; fixing the specification cuts median absolute percentage error from 8.04% to 3.77% while R-squared barely moves, from 0.774 to 0.784.

A gradient booster on the bad specification does not close the gap.

**Highlights**

- Full OLS diagnostic battery
- Omitted-interaction bias measured at 28.5% on one coefficient
- HC3 robust errors + Duan's smearing
- SHAP shown to be exactly `coef * (x - mean(x))` for a linear model
- 32 tests

[Interactive Walkthrough](https://aimlcompanion.ai/module/mlAlgorithms/rentPriceExplainer)

---

#### Support Ticket Triage - The Independence Assumption, Measured

The Naive Bayes assumption is provably false on ticket text, and the model works anyway. This project measures both halves. Redundancy families are planted at a known strength so the violation is known in advance, then found again blind with no answer key.

Fitting Naive Bayes beside multinomial logistic regression on identical data separates what the assumption costs (2.3 accuracy points) from the data simply carrying less information (6.4 points).

**Highlights**

- Planted violation recovered blind, with the precision and recall trade shown
- All four multiclass strategies (native, OvR, OvO, softmax)
- Why a chi-square p-value is the wrong tool at n=9000
- Isotonic calibration cuts ECE 0.0255 to 0.0097 while keeping 95.4% of decisions
- 33 tests

[Project README](projects/machine-learning/support-ticket-triage/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/mlAlgorithms/supportTicketTriage)

---

#### Cross-Validation Trap Lab - Measuring How Much CV Lies

Everyone knows cross-validation can be optimistic. Almost nobody measures by how much, because on real data the truth is never observable. Here it is: a panel is split so that unseen customers in future periods are held back, and five validation traps are each scored by their distance from that.

The result that reorganised the project is that the traps come in two families. Statistical traps, from how often you look, collapse from 0.18 to 0.0075 AUC as rows grow. Structural traps, from how you cut, do not shrink at all.

**Highlights**

- Every scheme scored against a real holdout rather than against each other
- Fixing time alone recovers only half the error, because customers are still shared
- No careful scheme lands on the truth, and the project says so
- Controls that switch the traps off, so the claims are falsifiable
- 36 tests

[Project README](projects/machine-learning/cross-validation-traps/) · Related lesson: [Cross-Validation](https://aimlcompanion.ai/module/mlAlgorithms/crossVal)

---

#### Lapse Prediction - One Ordered Target, Not Two Chained Models

An insurance renewal ledger where the lapse decision and the payment-timing distribution are modelled as a single ordered target rather than a classifier chained to a regression. Eleven model families sit behind one interface so the comparison is measured, not asserted.

The finding is that the algorithm barely matters (every legitimate model lands in AUC 0.802 to 0.818) and the data discipline does.

**Highlights**

- Ordinal cumulative chain over bucketed days-to-payment
- The two-stage hurdle baseline finishes 10th of 11
- Leakage guard via `groupby.shift(1)` plus a test that tampers with an outcome and asserts no feature moves
- Cohort maturity and out-of-time splits
- Release gate that fails the pipeline on regression
- 57 tests

[Project README](projects/machine-learning/lapse-prediction/)

---

#### ML Pipeline - Credit Risk with Monitoring

End-to-end pipeline from messy bank data to deployed model with KNN imputation, domain feature engineering, and PSI drift monitoring.

**Highlights**

- Feature engineering
- 10:1 cost-sensitive
- PSI drift detection
- FastAPI + Docker

[Interactive Walkthrough](https://aimlcompanion.ai/module/mlPipeline/mlPipelineCapstone)

---

### Computer Vision

#### Visual Defect Triage - Calibration Is What Makes a Gate Possible

A frozen vision transformer with a linear probe over 3,000 labels, and
the machinery that decides which predictions a human should see. Fits a
single temperature on validation, which cannot reorder logits and so
leaves accuracy untouched while halving expected calibration error, then
sets a confidence gate against a review budget.

The result worth reading is the slice report. Overall accuracy 0.975 hides a hairline-crack class at 0.681, and because that class is 2.6 per cent of traffic its entire improvement ceiling is 0.0083 against the healthy class's 0.0167.

The run asserts the ceilings sum to the error budget, so the report says where work is worth doing rather than only where the model is weak.

Runs offline on numpy alone. Torch and timm sit behind lazy imports.

**Highlights**

- One backbone pass feeding a classifier, a retrieval index and a drift monitor
- Temperature scaling that cannot change a prediction, ECE 0.013 to 0.009
- Slice ceilings that sum to the error budget
- Mined review decisions tagged so they can never enter the evaluation set
- 29 tests

[Project README](projects/computer-vision/visual-defect-triage/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/computerVision/cvVitCapstone) · [Kaggle Notebook](https://www.kaggle.com/code/genieincodebottle/visual-defect-triage-vit)

#### Site Safety Monitor - The Frame Budget Decides the Architecture

Two cameras at 15 fps into one edge device is 33.33 ms a frame. A frame that runs the detector costs 33.50 ms, so it does not fit, and inference is 53.7 per cent of it, which caps what optimising the network can ever be worth at 2.16x.

Detecting on every second frame and letting ByteTrack carry the gap averages 22.50 ms.

The pipeline turns 6,091 raw violation detections into 59 alerts, about
103 to one, and none of that comes from the detector. It comes from a
majority class vote, a point-in-polygon test on the worker's feet rather
than their box centre, a three second dwell timer, and deduplication by
zone and violation.

Runs offline on numpy alone. OpenCV and TensorRT are optional, and the geometry
is numpy rather than shapely so the pipeline carries no extra dependency.

**Highlights**

- A frame budget derived from camera count, not typed
- Amdahl ceiling of 2.16x on the obvious optimisation
- ByteTrack's low-confidence second pass
- A CI gate written in false alerts per shift
- 53 tests

[Project README](projects/computer-vision/site-safety-monitor/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/computerVision/cvYoloCapstone) · [Kaggle Notebook](https://www.kaggle.com/code/genieincodebottle/site-safety-monitor-yolo)

### Deep Learning

#### Deep Learning - CIFAR-10 Progressive Classifier

Systematically improve a CIFAR-10 image classifier from 60% to 93%+ accuracy across 6 documented experiments with a full diagnostics toolkit.

**Highlights**

- 6 progressive experiments
- ResNet + CutMix
- LR Finder
- Per-class analysis

[Interactive Walkthrough](https://aimlcompanion.ai/module/deepLearning/dlCapstone)

---

### LLM

#### GraphRAG Supply Chain Intelligence - When a Join Beats a Search

Multi-tier supply chain risk on a real Neo4j graph. Answers questions whose
answer exists in no single document, because it is a traversal across five
relationships rather than a passage in a chunk. Ships a 12-question benchmark
that deliberately includes cases where GraphRAG loses to plain RAG.

**Highlights**

- Neo4j graph + vector index in one store
- hybrid vector/BM25/graph retrieval
- LLM extraction with provenance and evidence quotes on every edge
- three-stage guardrails against graph poisoning
- measured comparison, including its own losses

[Project README](projects/llm/graphrag-supply-chain/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/llm/graphRagSupplyChain)

---

#### LLM/RAG - Expert Assistant

Production RAG system with chunking, security defense, and evaluation framework.

**Highlights**

- RAG pipeline
- PII defense
- A/B testing

[Interactive Walkthrough](https://aimlcompanion.ai/module/llm/ragExpertAssistant)

---

#### AI Reasoning Patterns - 23 Runnable Notebooks

23 self-contained Jupyter notebooks demonstrating advanced reasoning and agentic patterns, one pattern per notebook. Each runs on Google Colab with no local setup.

**Highlights**

- One pattern per notebook
- Colab-ready, zero install
- Runs locally under `jupyter notebook` just as well

[Project README](projects/llm/ai-patterns/) · [Blog post](https://aimlcompanion.ai/blog/top-25-genai-patterns-agentic-non-agentic-2026)

---

#### Muse Glimmer Lab - Open-Weight Model Internals

A runnable companion to the post on Meta's Muse Glimmer, a 30B open-weight agentic model that fits on a single 24 GB consumer GPU. Five experiments take the design decisions that make that possible and turn each into something you can run and change.

Two of the five are fully real with no model download at all.

**Highlights**

- Channel-scoped output parsed into reasoning, tool call and answer
- What the reasoning-strength knob costs
- A full agentic tool loop
- Why the KV cache is 1.7 GiB instead of 104 GiB
- Why a diffusion drafter beats an autoregressive one

[Project README](projects/llm/muse-glimmer-lab/) · [Blog Deep Dive](https://aimlcompanion.ai/blog/meta-muse-glimmer-explained-2026)

---

#### LLM Judge Lifecycle - The Judge Is a Service, Not a Score

A runnable companion to the post on the Netflix LLM judge paper (Kong et al., COLM 2026). Most teams treat an LLM judge as a benchmark score, built once and validated once. Here it is **a service in the request path that drifts**, and all four phases of its lifecycle are implemented on invented data.

**Highlights**

- All four lifecycle phases, from rubric tuning to gating and drift
- Every phase runs with no API key (`python run.py --offline benchmark`)
- A Colab notebook and a terminal quickstart
- 192 tests

[Project README](projects/llm/llm-judge-lifecycle/) · [Blog post](https://aimlcompanion.ai/blog/implementing-the-llm-judge-lifecycle-2026)

---

#### Jev Decision Lab - Are the Numbers Good Enough to Threshold?

A typed decision model takes unstructured state and a set of questions, and returns typed values with probabilities: no text, no JSON to parse, no retry when the model invents a field. This harness tests TypeSafe's Jev on the question that matters, **whether its probabilities are good enough to put a threshold on**, which is a question about your data, not the model.

**Highlights**

- One holistic question compared against five narrow ones
- Calibration and ECE measured, then the band you can automate and what it covers
- A simulated backend with no dependencies, no key and no network; the live SDK is optional

[Project README](projects/llm/jev-decision-lab/) · [Blog post](https://aimlcompanion.ai/blog/jev-system-one-model-classifier-returns-2026)

---

### Agentic AI

#### AI Agents - Multi-Agent Research System

A planner splits a topic into sub-topics, researchers run in parallel, a pure-Python quality gate sends weak sources back, and a writer and reviewer loop until the report passes or the revision cap is hit. It teaches **LangGraph routing and bounded loops**.

**Highlights**

- Seven agents in an eight-node graph, with two edges that point backwards
- Guardrails in code: input and snippet screening, a per-run token cap, a rate limit and PII handling
- The reviewer's pass is decided in code from its score, not by the model's own flag
- An evaluation that compares one model call with a three-call chain, counting tokens and latency as well as quality
- Offline mode runs the real graph with no API key

[Project README](projects/agentic-ai/ai-agents-project/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/agentsCapstone)

---

#### Content Moderation - Multi-Agentic System

A full-stack moderation platform. Six agents review each post, short posts can take a single-pass fast path, and hard cases go to a priority queue for human moderators, with appeals and agent memory. It teaches **human-in-the-loop review, agent memory and guardrails** in one app.

**Highlights**

- Plain code scores the words, and a model reads the context
- The account behind a post changes the answer: the same sentence can be approved, suspended or held
- Seven review triggers and a priority score decide which posts a person sees first
- The fast path is measured against the full graph, including the outcomes it gets wrong
- An offline CLI demo runs the six-agent graph with no key

[Project README](projects/agentic-ai/content-moderation-project/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/contentModerationProject)

---

#### Due Diligence Agent - Multi-Agent Company Research

Four analysts (financial, news, competitive, risk) research a company in parallel, a fact-check pass verifies their claims, and a lead analyst resolves the contradictions into a report with a risk rating. The key idea is **what to do when agents disagree**: the offline fixtures disagree on purpose, so the debate runs and one claim stays unverifiable.

**Highlights**

- Parallel fan-out via LangGraph `Send()` into one shared state
- A guard around every step: token, usage, call, repeat-entry and time limits
- Fact-checking and debate, with weak findings flagged rather than dropped
- Streamlit dashboard and an offline mode with no key

[Project README](projects/agentic-ai/due-diligence-agent/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/dueDiligenceProject)

---

#### Smart Claims Processor - Multi-Agent Insurance System

Insurance claims pass through seven agents, with a CrewAI crew for fraud inside one LangGraph node. Risky claims **pause mid-pipeline** with LangGraph's `interrupt()`, and a human reviewer's decision resumes the run from a checkpoint saved in SQLite.

**Highlights**

- LangGraph + CrewAI hybrid
- Human-in-the-loop with durable checkpointing
- A fast path only for small claims on a policy with a clean claim history
- Token caps, PII masking and LLM-as-judge evaluation
- Country-aware rule profiles (US / India) and pluggable LLMs (Gemini / Groq)
- React UI with an Agent Trace panel; self-registration creates claimant accounts only

[Project README](projects/agentic-ai/smart-claims-processor/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/smartClaimsProcessor)

---

#### Multi-Agent Anatomy - Production Failure Modes, Runnable

A runnable companion to the blog post "Inside a Production Multi-Agent GenAI System": an order-support request in 8 stages and 5 agents, built with **no agent framework on purpose** so every mechanism is visible. Six toggles break it: kill an agent, slow a tool, fail a saga step, serve a stale policy passage, turn off prompt caching, attempt a cross-tenant read.

The main view is a trace waterfall, not a chat window, because **a green trace does not mean a right answer**.

**Highlights**

- 5 agents, no framework
- Failure injection as a first-class feature
- Budget propagation + saga undos
- Token usage shown per span in the trace view
- Replay mode runs with no API key

[Project README](projects/agentic-ai/multi-agent-anatomy/) · [Blog deep dive](https://aimlcompanion.ai/blog/production-multi-agent-genai-architecture-2026)

---

#### Multi-Agents App on AWS - Bedrock AgentCore

An orchestrator hands a research request to three specialists (research, analyst, writer) and assembles their report. It is plain Python with no agent framework, so you see every handoff. Phase 2 deploys each agent to **Amazon Bedrock AgentCore**. Also published as a standalone repo; both copies are kept in sync.

**Highlights**

- 4 agents, first locally with a free Groq or Gemini key, then on managed AWS infrastructure
- Terraform for the runtime role, used by the deploy script
- AgentCore primitives and lifecycle, with a note on the move from the starter toolkit to the new AgentCore CLI

[Project README](projects/agentic-ai/multi-agents-app-on-aws/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/cloudGenAI/multiAgentsOnAws)

---

#### Hermes Ops Agent - Does the Learning Loop Pay Off?

Runs [Hermes Agent](https://github.com/NousResearch/hermes-agent) with hardened config and skills, then checks its claim that a task takes less work once Hermes has written a skill for it, by reading Hermes' own SQLite store. It teaches **measuring an agent instead of trusting its claims**.

**Highlights**

- Measures the learning loop instead of assuming it
- A repeatable prompt-injection probe
- Runs on a recorded fixture database with no Hermes install

[Project README](projects/agentic-ai/hermes-ops-agent/) · [AI Tools module](https://aimlcompanion.ai/module/aiTools/hermesAgent)

---

#### AI Incident Commander - The Model Decides, Code Decides What Is Allowed

Seven agents diagnose and fix faults in a simulated microservice stack inside one bounded loop, and every high-risk action stops for a human. The rule it teaches is **the model decides what to do, code decides what is allowed**.

**Highlights**

- Ten scenarios, including an unfixable fault and a false alarm, where the right answer is to hand over or do nothing
- Guardrails and approvals enforced in code, not in the prompt
- Offline mode reads the same telemetry with a rule-based agent, with a banner in the console
- FastAPI + LangGraph backend, React console, optional Postgres, Docker Compose

[Project README](projects/agentic-ai/incident-commander/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/incidentCommanderProject)

---

#### Mission Control - The Control Plane Under Agents You Did Not Write

A gateway every model call goes through, a policy engine for tool calls, a runtime that checkpoints and survives restarts, a reviewer gate with **maker-checker**, and an evaluation gate for prompt changes. It is proven on a six-agent anti-money-laundering investigation squad.

**Highlights**

- One gateway for every model call, with usage limits
- Policy checks on tool calls, and a second reviewer before an action is released
- Durable runs that resume after a restart
- An evaluation gate before a prompt change ships; offline mode runs the whole platform

[Project README](projects/agentic-ai/mission-control/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/missionControlProject)

---

#### Autonomous Migration Engineer - Catching Reward Hacking

An agent applies one code migration across many repositories: it edits, runs the tests, fixes what breaks, prepares a pull request and waits for approval. The key idea is **reward hacking**: scored on "tests pass", an agent can weaken or delete a test, and three independent checks catch that.

**Highlights**

- Many agents in parallel across repositories
- Three independent checks against weakened or deleted tests
- A human approval gate before any pull request
- Stub mode, a scripted worker with the same tools, budget, reviewer and gate, runs with no key

[Project README](projects/agentic-ai/migration-engineer/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/migrationEngineerProject)

---

#### Dream-RSI Explorer - Learning Where to Search Next

Implements Dream-RSI on one of the paper's benchmarks, packing 26 circles in a square. Gemini proposes layouts, and the rule that decides **where to search next** is improved by replaying recorded searches instead of running new ones.

**Highlights**

- A search policy improved by replay
- One benchmark from the paper, end to end
- Tests use a fake Gemini client and need no key

[Project README](projects/agentic-ai/dream-rsi-explorer/) · [Blog post](https://aimlcompanion.ai/blog/dream-rsi-recursive-self-improvement-explained-2026)

---

### Inference Engineering

#### Inference Serving Starter - Benchmark It Properly

The scaffold behind the Inference Engineering capstone. **It is deliberately incomplete: five of the six gates fail when you clone it.** The hard-to-fake parts are given (a deterministic vLLM-style serving engine with an OpenAI-compatible API and `/metrics`, a dataset, and recorded incidents), and the definition of done is written as failing tests. You write the load generator, the analysis and the report.

**Highlights**

- Open-loop and closed-loop load generators, written by you
- Percentiles, goodput, the knee of the load curve and a fleet size under two SLOs
- Quantisation and speculative decoding compared on the same engine
- Three recorded incidents to diagnose from metrics, plus an optional fourth
- Runs on a CPU

[Project README](projects/inference-engineering/inference-serving-starter/) · [Interactive Walkthrough](https://aimlcompanion.ai/module/inferenceEngineering/inferenceServingCapstone)

---

### MLOps

#### MLOps - Model Serving Platform

Production ML infrastructure: FastAPI with graceful shutdown, CI/CD pipeline, Prometheus metrics, Locust load testing, and operational runbook.

**Highlights**

- CI/CD (GitHub Actions)
- P95 < 45ms
- 161.7 RPS
- Kubernetes-ready

[Interactive Walkthrough](https://aimlcompanion.ai/module/mlOps/mlopsCapstone)

---

### Forward Deployment

#### FDE Engagement Starter - A Scaffold, Not a Solution

The repository behind the Forward Deployed Engineer path: the portfolio artifact and the 7-day embedded engagement simulation both run on it. Deliberately a starter, so the engineering judgment stays with you.

**Highlights**

- Portfolio artifact plus a 7-day engagement simulation
- FastAPI, pandas, pytest, Docker
- Eval workflow wired into CI

[Interactive Walkthrough](https://aimlcompanion.ai/module/forwardDeployment/fdePortfolioMVA) · [Engagement Simulation](https://aimlcompanion.ai/module/forwardDeployment/fdeEmbeddedSim)

---

### Data Engineering

#### Star Schema Modelling - Which Tests Catch a Broken Model

A dbt star schema on DuckDB for a generated online store with five planted source problems. Customers move region, shipping is charged per order, order lines are re-sent, some customer rows arrive late and some orders land on the second of a change. The generator keeps the true answer, so the reports are scored to the cent, and the correct model matches it exactly.

Then the model is broken 13 ways and all 64 tests run against each break. Six breaks pass every generic and singular test while the report is wrong, and one of those six fails no test at all.

**Highlights**

- Type 2 customer dimension joined by validity window, two facts at two grains
- Joining on `is_current` moves 17.53% of revenue between regions
- A mutation matrix that records which test tier catches each break
- 75 tests, including a check that the notebook matches the dbt results

[Interactive Walkthrough](https://aimlcompanion.ai/module/dataEngineering/deStarSchemaProject)

---

#### Event Log Replay - An Idempotent Sink You Can Prove

A 51,000-record wallet event log with 1,000 producer retries, 400 out-of-order profile updates and 52 seeded consumer crashes, written into DuckDB five ways and scored against an answer key. Committing the offset first loses 2,442 events. A dedup table in its own transaction still applies 2,175 events twice. Only one transaction for the effect, the event id and the offset gets all 2,000 accounts right.

A full replay from offset zero then shows which sinks are idempotent, and that an identical checksum is not the same as a correct one.

**Highlights**

- File-backed log with Kafka semantics, and an optional Redpanda backend with identical results
- State events take a guarded upsert, delta events need effect-level dedup
- Batch size changes how much is redelivered, never whether the result is correct
- 60 tests, plus 2 that run against a live broker

[Interactive Walkthrough](https://aimlcompanion.ai/module/dataEngineering/deReplayProject)

---

#### Incremental Warehouse - Rerun, Backfill and Fail Before Publish

The Data Engineering track capstone. A simulated shop database changes for 30 days while a daily pipeline extracts only what changed, lands Parquet partitions, builds a DuckDB warehouse and publishes it behind gating checks. A strict `updated_at` watermark misses all 111 late commits, and without a key snapshot all 27 hard-deleted orders stay published.

The pipeline proves three properties with commands and tests. A rerun leaves identical checksums, a late backfill equals a clean pass, and broken input stops at `check_raw` with the published tables unchanged.

**Highlights**

- Lookback window plus dedup for late commits, key reconciliation for hard deletes
- Partition overwrite by logical date, SCD type 2 customers, atomic publish
- Six naive switches measured against the correct pipeline
- A small in-repo DAG runner with retries and skips, and a runbook
- 56 tests

[Interactive Walkthrough](https://aimlcompanion.ai/module/dataEngineering/dataEngCapstone)

---

## Industry Best-Practice Project Structure

Most of the Python projects follow a consistent structure adapted from top ML teams:

```
project/
├── configs/                # Experiment configuration (YAML)
├── data/                   # Datasets the project reads
├── notebooks/              # Exploration & communication
├── src/                    # Production source code
├── tests/                  # Testing pyramid (unit/integration/load)
├── artifacts/              # Versioned outputs (models, results, figures)
├── docs/                   # Model cards, architecture docs, experiment logs
├── scripts/                # One-command automation scripts
├── docker/                 # Containerization (where applicable)
├── .gitignore
├── Makefile                # make train | make test | make serve
├── requirements.txt
└── README.md
```

## Key Principles

| Principle | What It Means |
|---|---|
| **Separation of Concerns** | Code (`src/`), config (`configs/`), data (`data/`), and artifacts (`artifacts/`) never mix |
| **Reproducibility First** | Configs are YAML, seeds are explicit, environments are containerized |
| **Notebook = Communication** | Notebooks prototype and communicate; `src/` is the production code |
| **Testing Pyramid** | Unit tests catch logic bugs, integration tests catch pipeline bugs, load tests catch scaling bugs |
| **Security by Default** | Input sanitization, PII detection, injection defense (critical for LLM projects) |
| **Observable from Day 1** | Monitoring, structured logging, metrics export built-in |

## Quick Start

Each project is self-contained and its own README is the authoritative setup
guide. Use Python 3.12: most projects require it, and the few that accept an
older version (3.10 or 3.11) say so in their README.

Most projects install and run like this:

```bash
cd projects/machine-learning/algorithm-showdown    # or any other project
pip install -r requirements.txt
make all                                           # train -> evaluate -> test
```

Where a project has no `all` target, run `make help` or follow the README, which
lists the commands for that project.

Many agentic projects run with no API key, against a deterministic stand-in for
the model that labels its output as such. The
[agentic index](projects/agentic-ai/README.md) shows which ones.

These projects are shaped differently and do not take the steps above:

| Project | How it runs |
|---|---|
| [AI Reasoning Patterns](projects/llm/ai-patterns/) | 23 standalone notebooks. Open one in Colab, or run `jupyter notebook` locally |
| [Muse Glimmer Lab](projects/llm/muse-glimmer-lab/) | `uv sync`, then `uv run python scripts/01_hello.py` |
| [Jev Decision Lab](projects/llm/jev-decision-lab/) | `uv sync`, then `uv run python run.py compare`, with no key or network |
| [LLM Judge Lifecycle](projects/llm/llm-judge-lifecycle/) | [Colab notebook](https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/projects/llm/llm-judge-lifecycle/notebooks/LLM_Judge_Lifecycle.ipynb), or `uv venv && uv pip install -r requirements.txt` then `python run.py --offline benchmark` - all four phases run with no API key |
| [Content Moderation](projects/agentic-ai/content-moderation-project/) | `uv pip install -r requirements.txt` inside `backend/`, then `python run.py demo` from the project folder |
| [Multi-Agent Anatomy](projects/agentic-ai/multi-agent-anatomy/) | `uv sync` in `backend/`, `npm install` in `frontend/`, run both |
| [Inference Serving Starter](projects/inference-engineering/inference-serving-starter/) | A starter: most gates fail on purpose until you write the missing parts. Its README lists the gates |

## Learning Path (Recommended Order)

Read in this order to build up. The `#` is the project's number in the
[Projects](#projects) table above, so it always points at the same project even
though the recommended order is not the table order.

```
MACHINE LEARNING
  #1  IPL Match Predictor      -> Data wrangling, EDA, visualization fundamentals
  #2  Algorithm Showdown       -> Classical ML, model comparison, interpretability
  #3  Rent Price Explainer     -> Regression diagnostics, why specification beats algorithm
  #4  Support Ticket Triage    -> Naive Bayes, multiclass strategies, calibration, routing
  #5  Cross-Validation Traps   -> Validation schemes, leakage, nested CV, what a fold allows
  #6  Lapse Prediction         -> Ordinal targets, leakage, cohort maturity, out-of-time splits
  #7  Credit Risk Pipeline     -> Feature engineering, end-to-end pipelines, monitoring
       |
DATA ENGINEERING
  #31 Star Schema Modelling    -> Grain, Type 2 history, which tests catch a broken model
  #32 Event Log Replay         -> Delivery semantics, idempotent sinks, the replay proof
  #33 Incremental Warehouse    -> Incremental extraction, gating checks, rerun and backfill
       |
DEEP LEARNING
  #10 Deep Learning Classifier -> Neural networks, progressive experimentation
       |
COMPUTER VISION
  #8  Visual Defect Triage     -> Vision transformers, calibration, slice analysis
  #9  Site Safety Monitor      -> Object detection, tracking, edge latency budgets
       |
MLOPS
  #29 Model Serving Platform   -> Deployment, CI/CD, load testing, infrastructure
       |
LLM
  #12 RAG Expert Assistant     -> Retrieval-augmented generation, evaluation, security
  #11 GraphRAG Supply Chain    -> Knowledge graphs, multi-hop retrieval, when a join beats a search
  #13 AI Reasoning Patterns    -> Reasoning and agentic patterns, one per notebook
  #14 Muse Glimmer Lab         -> Open-weight model internals, KV cache, drafters
  #15 LLM Judge Lifecycle      -> Evaluating the evaluator: rubric tuning, gating, drift
  #16 Jev Decision Lab         -> Typed decisions, calibration, where a threshold is safe
       |
INFERENCE ENGINEERING
  #28 Inference Serving Starter -> Open and closed loop benchmarks, SLOs, incidents, fleet size
       |
AGENTIC AI
  #22 Multi-Agents on AWS      -> Supervisor and specialists in plain Python, then managed deployment
  #17 AI Agents Research       -> LangGraph routing, a quality gate, a bounded revision loop
  #19 Due Diligence Agent      -> Parallel agents that disagree, fact-checking, debate
  #18 Content Moderation       -> Full stack, human review queue, memory, guardrails
  #20 Smart Claims Processor   -> Paused and resumed runs for human approval, LLM judge
  #21 Multi-Agent Anatomy      -> Production failure modes: partial failure, budgets, sagas
  #24 AI Incident Commander    -> A bounded loop where code decides what is allowed
  #26 Migration Engineer       -> Parallel coding agents, catching reward hacking
  #25 Mission Control          -> The control plane under agents you did not write
  #23 Hermes Ops Agent         -> Operating an agent, measuring its learning loop
  #27 Dream-RSI Explorer       -> Improving a search policy by replay
       |
FORWARD DEPLOYMENT
  #30 FDE Engagement Starter   -> Embedded engagement simulation, portfolio artifact
```

## Repository Structure

```
aiml-companion/
├── projects/
│   ├── machine-learning/
│   │   ├── ipl-match-predictor/        # EDA + Predictive Modeling
│   │   ├── algorithm-showdown/         # Classical ML + SHAP
│   │   ├── rent-price-explainer/       # Regression Diagnostics + SHAP
│   │   ├── support-ticket-triage/      # Naive Bayes + Multiclass + Calibration
│   │   ├── cross-validation-traps/     # Validation + Leakage + Nested CV
│   │   ├── lapse-prediction/           # Ordinal Targets + Leakage Discipline
│   │   └── credit-risk-pipeline/       # Credit Risk + Monitoring
│   ├── deep-learning/
│   │   └── deep-learning-project/      # CIFAR-10 + PyTorch
│   ├── computer-vision/
│   │   ├── visual-defect-triage/       # ViT + Calibration + Slice Analysis
│   │   └── site-safety-monitor/        # YOLO + Tracking + Edge Latency
│   ├── llm/
│   │   ├── graphrag-supply-chain/      # GraphRAG + Neo4j + Multi-Hop Retrieval
│   │   ├── rag-expert-assistant/       # RAG + Security
│   │   ├── ai-patterns/                # 23 Reasoning Pattern Notebooks
│   │   ├── muse-glimmer-lab/           # Open-Weight Model Internals
│   │   ├── llm-judge-lifecycle/        # LLM-as-a-Judge + Rubric Tuning + Drift
│   │   └── jev-decision-lab/           # Typed Decisions + Calibration + Thresholds
│   ├── agentic-ai/
│   │   ├── README.md                   # Index of the agentic projects + suggested order
│   │   ├── ai-agents-project/          # Multi-Agent Research + LangGraph
│   │   ├── content-moderation-project/ # Multi-Agent Moderation + Human Review
│   │   ├── due-diligence-agent/        # Multi-Agent Company Research
│   │   ├── smart-claims-processor/     # Multi-Agent Insurance Claims
│   │   ├── multi-agent-anatomy/        # Production Multi-Agent Failure Modes
│   │   ├── multi-agents-app-on-aws/    # Bedrock AgentCore
│   │   ├── hermes-ops-agent/           # Agent Operations + Learning Loops
│   │   ├── incident-commander/         # Bounded Agent Loop + Guardrails in Code
│   │   ├── mission-control/            # Agent Control Plane + Maker-Checker
│   │   ├── migration-engineer/         # Coding Agents + Reward Hacking Checks
│   │   └── dream-rsi-explorer/         # Search Policy Improved by Replay
│   ├── inference-engineering/
│   │   └── inference-serving-starter/  # Serving Benchmarks + Incidents + Fleet Size
│   ├── data-engineering/
│   │   ├── star-schema-modelling/      # dbt + DuckDB + Mutation Matrix
│   │   ├── event-log-replay/           # Idempotent Sinks + Replay Proof
│   │   └── incremental-warehouse/      # Incremental Pipeline + Gating Checks
│   ├── mlops/
│   │   └── model-serving-platform/     # Model Serving + CI/CD
│   └── forward-deployment/
│       └── fde-engagement-starter/     # FDE Engagement Scaffold
└── README.md                           # This file
```

---

**Author:** [Rajesh Srivastava](https://github.com/genieincodebottle)

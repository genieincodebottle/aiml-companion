# AI/ML Learning Companion

> **Learn AI/ML interactively at [AI-ML Companion](https://aimlcompanion.ai/)** - Guided walkthroughs, architecture decisions, hands-on challenges, and narrated overviews.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Projects](https://img.shields.io/badge/Projects-23-orange)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

> A complete AI/ML learning platform with 23 end-to-end projects covering the full spectrum, from classical ML to production LLM systems. Most projects follow a common **industry best-practice structure**.

---

## Contents

- [About the AI-ML Companion Platform](#about-the-ai-ml-companion-platform)
- [Projects](#projects) - all 23, grouped by domain
- [Project Details](#project-details) - what each one actually does
- [Industry Best-Practice Project Structure](#industry-best-practice-project-structure)
- [Key Principles](#key-principles)
- [Quick Start](#quick-start)
- [Learning Path (Recommended Order)](#learning-path-recommended-order)
- [Repository Structure](#repository-structure)

---

## About the AI-ML Companion Platform

These 24 projects are the hands-on companion to **[AI-ML Companion](https://aimlcompanion.ai)**, an interactive platform for learning AI and machine learning by watching it work. Instead of passive video, every concept is taught through **live, animated visualizations** and runnable code: gradient descent stepping down a loss surface, attention weights lighting up across tokens, a network training in real time.

- **28 tracks, 400+ modules** from Python and math foundations to deep learning, LLMs, agentic AI, MLOps, and cloud GenAI.
- **Interactive visualizations**, not slides. See the mechanism move, then experiment with it.
- **Runnable Python** in the browser, quizzes, and progress tracking.
- **Trilingual narration** (English, Hindi, Spanish) with technical terms kept in English.
- **PWA**, works on phone and desktop, installable and offline-capable.
- **Free tier:** 8 complete tracks (Foundations, Python for ML, Jupyter, Linear Algebra, ML Algorithms, Git, AI Tools, Interview Q&A) plus the first module of every Premium track. Premium unlocks the full catalog.

**Curated learning paths:** Data Science and AI Engineering guide you end to end.

**Explore:** [Curriculum](https://aimlcompanion.ai/curriculum) · [Roadmap](https://aimlcompanion.ai/roadmap) · [Blog](https://aimlcompanion.ai/blog)

### FAQ

**Is AI-ML Companion free?** Yes, there is a free tier: 8 complete tracks plus the opening module of every Premium track. Premium unlocks all 28 tracks and 400+ modules.

**How is it different from video courses?** It is interactive. Concepts are taught through live visualizations and runnable code rather than passive video, so you can watch algorithms operate and experiment directly.

**Who is it for?** Beginners wanting an intuition-first on-ramp, engineers moving into ML / MLOps / LLM / agentic AI roles, and anyone preparing for AI/ML interviews.

**What languages are supported?** English, Hindi, and Spanish, with technical terminology kept in English.

---

## Projects

### Machine Learning

| # | Project | Domain | Difficulty | Key Tech | Learn / Docs |
|---|---|---|---|---|---|
| 1 | [IPL Match Predictor](projects/machine-learning/ipl-match-predictor/) | Data Science / EDA | Beginner-Intermediate | Pandas, Plotly, Scikit-learn | [Learn &rarr;](https://aimlcompanion.ai/module/mlAlgorithms/iplProject) |
| 2 | [Algorithm Showdown](projects/machine-learning/algorithm-showdown/) | Classical ML / Interpretability | Intermediate | Scikit-learn, XGBoost, SHAP | [Learn &rarr;](https://aimlcompanion.ai/module/mlAlgorithms/mlAlgorithmsCapstone) |
| 3 | [Rent Price Explainer](projects/machine-learning/rent-price-explainer/) | Regression Diagnostics | Intermediate | Statsmodels, Scikit-learn, SHAP | [Learn &rarr;](https://aimlcompanion.ai/module/mlAlgorithms/rentPriceExplainer) |
| 4 | [Support Ticket Triage](projects/machine-learning/support-ticket-triage/) | Text Classification / Calibration | Intermediate | Scikit-learn, SciPy, Pandas | [Learn &rarr;](https://aimlcompanion.ai/module/mlAlgorithms/supportTicketTriage) |
| 5 | [Cross-Validation Traps](projects/machine-learning/cross-validation-traps/) | Model Validation | Intermediate-Advanced | Scikit-learn, Pandas, NumPy | [README &rarr;](projects/machine-learning/cross-validation-traps/) |
| 6 | [Lapse Prediction](projects/machine-learning/lapse-prediction/) | Ordinal Modelling / Insurance | Advanced | LightGBM, XGBoost, Statsmodels | [README &rarr;](projects/machine-learning/lapse-prediction/) |
| 7 | [Credit Risk Pipeline](projects/machine-learning/credit-risk-pipeline/) | Feature Engineering / Production ML | Advanced | Scikit-learn, FastAPI, Docker | [Learn &rarr;](https://aimlcompanion.ai/module/mlPipeline/mlPipelineCapstone) |

### Computer Vision

| # | Project | Domain | Difficulty | Key Tech | Learn / Docs |
|---|---|---|---|---|---|
| 8 | [Visual Defect Triage](projects/computer-vision/visual-defect-triage/) | Computer Vision / Calibration | Advanced | ViT, NumPy, Pydantic | [Learn &rarr;](https://aimlcompanion.ai/module/computerVision/cvVitCapstone) |
| 9 | [Site Safety Monitor](projects/computer-vision/site-safety-monitor/) | Computer Vision / Edge Inference | Advanced | YOLO, ByteTrack, NumPy | [Learn &rarr;](https://aimlcompanion.ai/module/computerVision/cvYoloCapstone) |

### Deep Learning

| # | Project | Domain | Difficulty | Key Tech | Learn / Docs |
|---|---|---|---|---|---|
| 10 | [Deep Learning Classifier](projects/deep-learning/deep-learning-project/) | Computer Vision / DL | Intermediate-Advanced | PyTorch, TorchVision | [Learn &rarr;](https://aimlcompanion.ai/module/deepLearning/dlCapstone) |

### LLM

| # | Project | Domain | Difficulty | Key Tech | Learn / Docs |
|---|---|---|---|---|---|
| 11 | [GraphRAG Supply Chain](projects/llm/graphrag-supply-chain/) | Graph + Vector Retrieval | Advanced | Neo4j, Gemini, FastAPI, Streamlit | [Learn &rarr;](https://aimlcompanion.ai/module/llm/graphRagSupplyChain) |
| 12 | [RAG Expert Assistant](projects/llm/rag-expert-assistant/) | Retrieval-Augmented Generation | Advanced | LangChain, ChromaDB | [Learn &rarr;](https://aimlcompanion.ai/module/llm/ragExpertAssistant) |
| 13 | [AI Reasoning Patterns](projects/llm/ai-patterns/) | Prompting / Reasoning Patterns | Intermediate | Jupyter, Colab, Gemini | [README &rarr;](projects/llm/ai-patterns/) |
| 14 | [Muse Glimmer Lab](projects/llm/muse-glimmer-lab/) | Open-Weight Model Internals | Advanced | vLLM, OpenAI-compatible API | [README &rarr;](projects/llm/muse-glimmer-lab/) |
| 15 | [LLM Judge Lifecycle](projects/llm/llm-judge-lifecycle/) | LLM Evaluation / EvalOps | Advanced | Gemini, FastAPI, Streamlit, Pytest | [Notebook &rarr;](projects/llm/llm-judge-lifecycle/notebooks/LLM_Judge_Lifecycle.ipynb) |

### Agentic AI

| # | Project | Domain | Difficulty | Key Tech | Learn / Docs |
|---|---|---|---|---|---|
| 16 | [AI Agents Research System](projects/agentic-ai/ai-agents-project/) | LLM Agent Orchestration | Advanced | LangGraph, OpenAI, Tavily | [Learn &rarr;](https://aimlcompanion.ai/module/aiAgents/agentsCapstone) |
| 17 | [Content Moderation](projects/agentic-ai/content-moderation-project/) | Multi-Agent Moderation | Advanced | LangGraph, Multi-Agent | [Learn &rarr;](https://aimlcompanion.ai/module/aiAgents/contentModerationProject) |
| 18 | [Due Diligence Agent](projects/agentic-ai/due-diligence-agent/) | Multi-Agent Research | Advanced | LangGraph, Gemini, Streamlit | [Learn &rarr;](https://aimlcompanion.ai/module/aiAgents/dueDiligenceProject) |
| 19 | [Smart Claims Processor](projects/agentic-ai/smart-claims-processor/) | Multi-Agent Insurance Claims | Advanced | LangGraph, CrewAI, Gemini, FastAPI, React | [Learn &rarr;](https://aimlcompanion.ai/module/aiAgents/smartClaimsProcessor) |
| 20 | [Multi-Agent Anatomy](projects/agentic-ai/multi-agent-anatomy/) | Multi-Agent Failure Modes | Advanced | FastAPI, React, Gemini, no framework | [README &rarr;](projects/agentic-ai/multi-agent-anatomy/) |
| 21 | [Multi-Agents App on AWS](projects/agentic-ai/multi-agents-app-on-aws/) | Cloud GenAI / Managed Agents | Advanced | AWS Bedrock AgentCore | [Learn &rarr;](https://aimlcompanion.ai/module/cloudGenAI/multiAgentsOnAws) |
| 22 | [Hermes Ops Agent](projects/agentic-ai/hermes-ops-agent/) | Agent Operations / Measurement | Intermediate-Advanced | Hermes, Pytest, PyYAML | [README &rarr;](projects/agentic-ai/hermes-ops-agent/) |

### MLOps

| # | Project | Domain | Difficulty | Key Tech | Learn / Docs |
|---|---|---|---|---|---|
| 23 | [Model Serving Platform](projects/mlops/model-serving-platform/) | Model Deployment / Infrastructure | Advanced | FastAPI, Docker, Prometheus, GitHub Actions | [Learn &rarr;](https://aimlcompanion.ai/module/mlOps/mlopsCapstone) |

### Forward Deployment

| # | Project | Domain | Difficulty | Key Tech | Learn / Docs |
|---|---|---|---|---|---|
| 24 | [FDE Engagement Starter](projects/forward-deployment/fde-engagement-starter/) | Forward Deployment Engineering | Intermediate-Advanced | FastAPI, Pandas, Pytest, Docker | [Learn &rarr;](https://aimlcompanion.ai/module/forwardDeployment/fdePortfolioMVA) |

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

[Project README](projects/machine-learning/support-ticket-triage/)

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

[Project README](projects/machine-learning/cross-validation-traps/)

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

[Project README](projects/llm/graphrag-supply-chain/)

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

[Project README](projects/llm/ai-patterns/)

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

### Agentic AI

#### AI Agents - Multi-Agent Research System

4-agent orchestrated research pipeline (researcher, analyst, writer, fact-checker) with guardrails, evaluation, and cost tracking.

**Highlights**

- LangGraph orchestration
- +33% completeness vs single-agent
- Budget tracking
- LLM-as-judge

[Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/agentsCapstone)

---

#### Content Moderation - Multi-Agentic System

Multi-agent content moderation pipeline with specialized agents for different content types.

[Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/contentModerationProject)

---

#### Due Diligence Agent - Multi-Agent Company Research

Enterprise-grade company research powered by 6 AI agents with parallel execution, fact-checking, contradiction resolution, and comprehensive guardrails.

**Highlights**

- 6 specialist agents
- Parallel execution via LangGraph Send()
- Fact-checking + debate
- Streamlit dashboard

[Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/dueDiligenceProject)

---

#### Smart Claims Processor - Multi-Agent Insurance System

Production-style multi-agent insurance claims system built with LangGraph (orchestration) and CrewAI (fraud detection). 7 specialist agents handle intake validation, fraud detection, damage assessment, policy compliance, settlement calculation, LLM-as-judge evaluation, and claimant notification.

**Highlights**

- LangGraph + CrewAI hybrid
- Human-in-the-Loop with durable checkpointing
- Per-agent confidence gates
- Country-aware (US/India)
- Pluggable LLMs (Gemini/Groq)
- React UI with Agent Trace panel

[Interactive Walkthrough](https://aimlcompanion.ai/module/aiAgents/smartClaimsProcessor)

---

#### Multi-Agent Anatomy - Production Failure Modes, Runnable

A runnable companion to the blog post "Inside a Production Multi-Agent GenAI System": an ecommerce order-support assistant with 8 stages and 5 agents, built with **no agent framework on purpose** so every mechanism is visible. Answering correctly is the least interesting part - the point is breaking it and watching what happens: partial failure, budget propagation, saga undos, per-agent timeouts, cache-aware prompt ordering, and observability that stays green while the answer is wrong.

The main view is a trace waterfall, not a chat window.

**Highlights**

- 5 agents, no framework
- Failure injection as a first-class feature
- Budget propagation + saga undos
- Trace-waterfall UI
- [Blog deep dive](https://aimlcompanion.ai/blog/production-multi-agent-genai-architecture-2026)

[Project README](projects/agentic-ai/multi-agent-anatomy/)

---

#### Multi-Agents App on AWS - Bedrock AgentCore

Three specialist agents plus one orchestrator collaborating to research a topic and produce a report, deployed on AWS Bedrock AgentCore. Also published as a standalone repo; both copies are kept in sync so the project is discoverable alongside the rest of this repository.

**Highlights**

- 4 agents on managed AWS infrastructure
- AgentCore primitives and lifecycle
- Deployable end to end

[Interactive Walkthrough](https://aimlcompanion.ai/module/cloudGenAI/multiAgentsOnAws)

---

#### Hermes Ops Agent - Does the Learning Loop Pay Off?

Operating an off-the-shelf agent properly, and measuring whether its central claim survives contact with your machine. Hermes claims that once it has written a skill for a task, doing that task again is cheaper. This project runs that experiment rather than repeating the claim.

**Highlights**

- Measures the learning loop instead of assuming it
- Runs on your own hardware
- Operations focus rather than another agent build

[Project README](projects/agentic-ai/hermes-ops-agent/)

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

[Interactive Walkthrough](https://aimlcompanion.ai/module/forwardDeployment/fdePortfolioMVA)

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
guide. Python 3.10 or newer; a few projects ask for 3.11+ and say so.

Most projects install and run like this:

```bash
cd projects/machine-learning/algorithm-showdown    # or any other project
pip install -r requirements.txt
make all                                           # train -> evaluate -> test
```

15 projects ship a Makefile and 10 of those define `all`. Where there is no
`all` target, run `make help` or follow the README, which lists the commands
for that project.

Four projects are shaped differently and do not take the steps above:

| Project | How it runs |
|---|---|
| [AI Reasoning Patterns](projects/llm/ai-patterns/) | 23 standalone notebooks. Open one in Colab, or run `jupyter notebook` locally |
| [Muse Glimmer Lab](projects/llm/muse-glimmer-lab/) | `uv sync`, then `uv run python scripts/01_hello.py` |
| [Content Moderation](projects/agentic-ai/content-moderation-project/) | `cd backend`, `uv pip install -r requirements.txt`, then `python run.py demo` |
| [Multi-Agent Anatomy](projects/agentic-ai/multi-agent-anatomy/) | `uv sync` in `backend/`, `npm install` in `frontend/`, run both |
| [LLM Judge Lifecycle](projects/llm/llm-judge-lifecycle/) | [Colab notebook](https://colab.research.google.com/github/genieincodebottle/aiml-companion/blob/main/projects/llm/llm-judge-lifecycle/notebooks/LLM_Judge_Lifecycle.ipynb), or `uv venv && uv pip install -r requirements.txt` then `python run.py --offline benchmark` - all four phases run with no API key |

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
DEEP LEARNING
  #10 Deep Learning Classifier -> Neural networks, progressive experimentation
       |
COMPUTER VISION
  #8  Visual Defect Triage     -> Vision transformers, calibration, slice analysis
  #9  Site Safety Monitor      -> Object detection, tracking, edge latency budgets
       |
MLOPS
  #23 Model Serving Platform   -> Deployment, CI/CD, load testing, infrastructure
       |
LLM
  #12 RAG Expert Assistant     -> Retrieval-augmented generation, evaluation, security
  #11 GraphRAG Supply Chain    -> Knowledge graphs, multi-hop retrieval, when a join beats a search
  #13 AI Reasoning Patterns    -> Reasoning and agentic patterns, one per notebook
  #14 Muse Glimmer Lab         -> Open-weight model internals, KV cache, drafters
  #15 LLM Judge Lifecycle      -> Evaluating the evaluator: rubric tuning, gating, drift
       |
AGENTIC AI
  #16 AI Agents Research       -> Multi-agent orchestration, guardrails, cost optimization
  #17 Content Moderation       -> Multi-agentic content pipelines
  #18 Due Diligence Agent      -> Enterprise multi-agent research, fact-checking, debate
  #19 Smart Claims Processor   -> Multi-agent insurance, HITL, hybrid orchestration
  #21 Multi-Agents on AWS      -> Managed multi-agent infrastructure, Bedrock AgentCore
  #22 Hermes Ops Agent         -> Operating an agent, measuring its learning loop
  #20 Multi-Agent Anatomy      -> Production failure modes: partial failure, budgets, sagas
       |
FORWARD DEPLOYMENT
  #24 FDE Engagement Starter   -> Embedded engagement simulation, portfolio artifact
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
│   │   └── llm-judge-lifecycle/        # LLM-as-a-Judge + Rubric Tuning + Drift
│   ├── agentic-ai/
│   │   ├── ai-agents-project/          # Multi-Agent + LangGraph
│   │   ├── content-moderation-project/ # Multi-Agentic Content Moderation
│   │   ├── due-diligence-agent/        # Multi-Agent Company Research
│   │   ├── smart-claims-processor/     # Multi-Agent Insurance Claims
│   │   ├── multi-agent-anatomy/        # Production Multi-Agent Failure Modes
│   │   ├── multi-agents-app-on-aws/    # Bedrock AgentCore
│   │   └── hermes-ops-agent/           # Agent Operations + Learning Loops
│   ├── mlops/
│   │   └── model-serving-platform/     # Model Serving + CI/CD
│   └── forward-deployment/
│       └── fde-engagement-starter/     # FDE Engagement Scaffold
└── README.md                           # This file
```

---

**Author:** [Rajesh Srivastava](https://github.com/genieincodebottle)

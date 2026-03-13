# AI/ML Learning Companion

> **Learn AI/ML interactively at [AI-ML Companion](https://aimlcompanion.ai/)** - Guided walkthroughs, architecture decisions, hands-on challenges, and narrated overviews for every project.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Projects](https://img.shields.io/badge/Projects-9-orange)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

> A comprehensive AI/ML learning platform with 9 end-to-end projects covering the full spectrum, from classical ML to production LLM systems. Every project follows **industry best-practice structure**.

---

## Projects

| # | Project | Domain | Difficulty | Key Tech |
|---|---|---|---|---|
| 1 | [ML Algorithms](projects/algorithm-showdown/) | Classical ML / Interpretability | Intermediate | Scikit-learn, XGBoost, SHAP |
| 2 | [Deep Learning](projects/deep-learning-project/) | Computer Vision / DL | Intermediate-Advanced | PyTorch, TorchVision |
| 3 | [ML Pipeline](projects/credit-risk-pipeline/) | Feature Engineering / Production ML | Advanced | Scikit-learn, FastAPI, Docker |
| 4 | [MLOps](projects/model-serving-platform/) | Model Deployment / Infrastructure | Advanced | FastAPI, Docker, Prometheus, GitHub Actions |
| 5 | [LLM/RAG](projects/rag-expert-assistant/) | Retrieval-Augmented Generation | Advanced | LangChain, ChromaDB |
| 6 | [AI Agents](projects/ai-agents-project/) | LLM Agent Orchestration | Advanced | LangGraph, OpenAI, Tavily |
| 7 | [IPL Analysis](projects/ipl-match-predictor/) | Data Science / EDA | Beginner-Intermediate | Pandas, Plotly, Scikit-learn |
| 8 | [Content Moderation](projects/content-moderation-project/) | Multi-Agentic AI | Advanced | LangGraph, Multi-Agent |
| 9 | [Due Diligence Agent](projects/due-diligence-agent/) | Multi-Agent Research | Advanced | LangGraph, Gemini, Streamlit |

---

## Project Details

### 1. ML Algorithms - Medical Diagnostic Classifier

Compare 6 ML algorithms on real clinical data with cost-sensitive threshold tuning (~95% malignant recall) and SHAP explainability for regulatory review.

**Highlights:** 6 algorithms compared | XGBoost AUC ~0.994 | SHAP reports | Threshold tuning

---

### 2. Deep Learning - CIFAR-10 Progressive Classifier

Systematically improve a CIFAR-10 image classifier from 60% to 93%+ accuracy across 6 documented experiments with a full diagnostics toolkit.

**Highlights:** 6 progressive experiments | ResNet + CutMix | LR Finder | Per-class analysis

---

### 3. ML Pipeline - Credit Risk with Monitoring

End-to-end pipeline from messy bank data to deployed model with KNN imputation, domain feature engineering, and PSI drift monitoring.

**Highlights:** Feature engineering | 10:1 cost-sensitive | PSI drift detection | FastAPI + Docker

---

### 4. MLOps - Model Serving Platform

Production ML infrastructure: FastAPI with graceful shutdown, CI/CD pipeline, Prometheus metrics, Locust load testing, and operational runbook.

**Highlights:** CI/CD (GitHub Actions) | P95 < 45ms | 161.7 RPS | Kubernetes-ready

---

### 5. LLM/RAG - Expert Assistant

Production RAG system with chunking, security defense, and evaluation framework.

**Highlights:** RAG pipeline | PII defense | A/B testing

---

### 6. AI Agents - Multi-Agent Research System

4-agent orchestrated research pipeline (researcher, analyst, writer, fact-checker) with guardrails, evaluation, and cost tracking.

**Highlights:** LangGraph orchestration | +33% completeness vs single-agent | Budget tracking | LLM-as-judge

---

### 7. IPL Dataset Analysis - End-to-End EDA

Comprehensive analysis of 17 IPL seasons with interactive visualizations, hypothesis testing, feature engineering, and predictive modeling.

**Highlights:** 1000+ matches | Plotly interactive charts | Hypothesis testing | RF + GB models

---

### 8. Content Moderation - Multi-Agentic System

Multi-agent content moderation pipeline with specialized agents for different content types.

---

### 9. Due Diligence Agent - Multi-Agent Company Research

Enterprise-grade company research powered by 6 AI agents with parallel execution, fact-checking, contradiction resolution, and comprehensive guardrails.

**Highlights:** 6 specialist agents | Parallel execution via LangGraph Send() | Fact-checking + debate | Streamlit dashboard

---

## Industry Best-Practice Project Structure

Every project follows a consistent structure adapted from top ML teams:

```
project/
├── configs/                # Experiment configuration (YAML)
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

Each project is self-contained. Pick one and follow its README:

```bash
cd projects/algorithm-showdown    # or any other project
pip install -r requirements.txt
make all                          # train -> evaluate -> test
```

## Learning Path (Recommended Order)

```
1. IPL Analysis          -> Data wrangling, EDA, visualization fundamentals
       |
2. ML Algorithms         -> Classical ML, model comparison, interpretability
       |
3. Deep Learning         -> Neural networks, progressive experimentation
       |
4. ML Pipeline           -> Feature engineering, end-to-end pipelines, monitoring
       |
5. MLOps                 -> Deployment, CI/CD, load testing, infrastructure
       |
6. LLM/RAG              -> Retrieval-augmented generation, evaluation, security
       |
7. AI Agents             -> Multi-agent orchestration, guardrails, cost optimization
       |
8. Content Moderation    -> Multi-agentic content pipelines
       |
9. Due Diligence Agent   -> Enterprise multi-agent research, fact-checking, debate
```

## Repository Structure

```
aiml-companion/
├── projects/
│   ├── algorithm-showdown/         # Classical ML + SHAP
│   ├── deep-learning-project/      # CIFAR-10 + PyTorch
│   ├── credit-risk-pipeline/       # Credit Risk + Monitoring
│   ├── model-serving-platform/     # Model Serving + CI/CD
│   ├── rag-expert-assistant/       # RAG + Security
│   ├── ai-agents-project/          # Multi-Agent + LangGraph
│   ├── ipl-match-predictor/        # EDA + Predictive Modeling
│   ├── content-moderation-project/ # Multi-Agentic Content Moderation
│   └── due-diligence-agent/        # Multi-Agent Company Research
└── README.md                       # This file
```

---

**Author:** [Rajesh Srivastava](https://github.com/genieincodebottle)
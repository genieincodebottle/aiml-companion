# Model Serving Platform: Production ML Infrastructure

> **Learn how to build this project step-by-step on [AI-ML Companion](https://aimlcompanion.ai/)** - Interactive ML learning platform with guided walkthroughs, architecture decisions, and hands-on challenges.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-brightgreen)

Complete ML model deployment with CI/CD, monitoring, load testing, and graceful shutdown.

## Architecture

```
+---------------------------------------------+
|  GitHub Actions CI/CD                        |
|  lint -> test -> build -> push -> smoke test |
+---------------------------------------------+
    |
    v
+---------------------------------------------+
|  Docker Container                            |
|  +------------------------------------------+
|  |  FastAPI Server                          |
|  |  /health (warm model check)             |
|  |  /predict (Pydantic + inference)        |
|  |  /metrics (Prometheus)                  |
|  |  Graceful shutdown (SIGTERM)            |
|  +------------------------------------------+
|  Pre-loaded sklearn model at startup        |
+---------------------------------------------+
    |                              |
    v                              v
+--------------------+  +------------------------+
|  Locust Load Test  |  |  Prometheus/Grafana     |
|  100 RPS target    |  |  Latency P50/P95/P99   |
|  <200ms P95 SLA    |  |  Error rate, drift      |
+--------------------+  +------------------------+
```

## Problem Statement

A trained ML model is worthless without production infrastructure. This project builds the operational layer: containerized serving with health checks, CI/CD pipeline, Prometheus metrics, load testing, and documented operational procedures.

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/model-serving-platform

# Install uv (if you don't have it)
pip install uv

# Create virtual environment and install dependencies
uv venv
# Activate it
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

uv pip install -r requirements.txt
```

### 2. Create Demo Model and Start Server

```bash
python scripts/create_demo_model.py
uvicorn src.app.main:app --port 8000
```

### 3. Test the Endpoints

Open a new terminal:

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Prometheus metrics
curl http://localhost:8000/metrics
```

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Docker Deployment (optional)

```bash
docker build -t model-server -f docker/Dockerfile .
docker run --rm -p 8000:8000 model-server
```

### 6. Load Test (optional)

```bash
uv pip install locust
locust -f tests/load/locustfile.py --headless -u 10 -r 2 -t 30s --host http://localhost:8000
```

> **Shortcut**: If you have `make` installed, `make serve` runs steps 2-3 automatically.

## Project Structure

```
model-serving-platform/
|-- .github/workflows/
|   +-- ci.yml                  # Full CI/CD: lint -> test -> build -> push
|-- configs/
|   +-- base.yaml               # Server settings, model path, SLA thresholds
|-- src/
|   |-- app/
|   |   |-- main.py             # FastAPI with lifespan, graceful shutdown
|   |   +-- metrics.py          # Prometheus latency/count/error metrics
|   +-- monitoring/
|       +-- metrics.py          # Latency tracking, drift detection
|-- tests/
|   |-- test_api.py             # Request validation, health, metrics tests
|   +-- load/
|       +-- locustfile.py       # Realistic load test (10% health + 90% predict)
|-- artifacts/
|   |-- models/                 # Model artifacts (.gitkeep)
|   +-- results/
|       +-- load_test_report.md # Sample P50/P95/P99 results
|-- docker/
|   |-- Dockerfile              # Multi-stage build with health check
|   +-- docker-compose.yml      # Service definition with build context
|-- docs/
|   |-- model_card.md           # Model documentation
|   +-- runbook.md              # Deployment, rollback, incident response
|-- scripts/
|   |-- run_server.sh           # Start uvicorn server
|   +-- run_load_test.sh        # Run Locust load tests
|-- notebooks/
|   +-- Model_Serving_Platform.ipynb
|-- Makefile                    # serve, test, load-test, docker-build, docker-run, all
|-- requirements.txt
+-- README.md
```

## Load Test Results

| Metric | Value | SLA | Status |
|---|---|---|---|
| P50 latency | 12ms | - | - |
| P95 latency | 45ms | < 200ms | PASS |
| P99 latency | 89ms | - | - |
| Error rate | 0.02% | < 1% | PASS |
| Throughput | 161.7 RPS | > 100 RPS | PASS |

> Sample results from 100 concurrent users, 5-minute run.

## Interview Guide: How to Talk About This Project

### "Walk me through this project."

"I built the production infrastructure around an ML model: a FastAPI server with graceful shutdown, a multi-stage Docker build, GitHub Actions CI/CD pipeline, Prometheus metrics, and Locust load testing. The model itself is intentionally simple (Iris classifier) because the infrastructure is the deliverable."

### "What was the hardest part?"

"Getting graceful shutdown right. When Kubernetes sends SIGTERM, the server needs to finish in-flight requests before exiting. I used FastAPI's lifespan context manager for startup/shutdown and signal handlers for SIGTERM. The health endpoint also serves as a readiness probe for load balancers."

### "What would you do differently?"

"Add A/B testing infrastructure for safe model rollouts, implement canary deployments with traffic splitting, and add prediction logging to a data warehouse for offline analysis and retraining triggers."

### "How does this scale?"

"Horizontally via Kubernetes: the Docker container is stateless (model loaded at startup from shared volume), health checks enable rolling updates, and Prometheus metrics feed into auto-scaling policies based on latency P95."

### "Explain CI/CD to a non-technical person."

"Every time a developer pushes code, an automated system checks for mistakes (linting), runs tests, builds a shipping container, and deploys it. It's like having a quality inspector who checks every product before it leaves the factory, but running in seconds instead of hours."

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

## Three failures that look like features

Operational code fails differently from modelling code: it does not throw, it
reassures. All three of these passed their tests and read correctly.

### The drift monitor was silent when drift was total

PSI bins are quantiles of the reference distribution, and `np.histogram`
**discards values outside the bin range**. So a production distribution that had
moved far enough to share no support with reference produced an all-zero
histogram, which after the epsilon floor is uniform — the same shape as a
reference histogram that is uniform by construction. PSI came out at ~0.

| current, vs reference ~ N(0,1) | PSI before | verdict before | PSI after |
|---|---|---|---|
| N(0, 1) — no drift | 0.0055 | stable | 0.0054 stable |
| N(0.5, 1) | 0.2492 | moderate | 0.2531 |
| N(2, 1) | 3.1664 | significant | 3.3246 |
| **N(10, 1) — catastrophic** | **0.0000** | **stable** | **8.2811 significant** |
| **N(100, 1)** | **0.0000** | **stable** | **8.2811 significant** |
| **constant 999** | **0.0000** | **stable** | **8.2811 significant** |

The alarm was loudest for mild drift and completely silent for total drift. The
fix is one line — open the outer bins to ±∞ so out-of-range mass is counted —
and PSI is now monotone in drift severity.

### The health check reported 200 while saying "unhealthy"

`/health` returned HTTP 200 with `status: "unhealthy"` in the body. But `curl
-f` — what the Dockerfile and docker-compose healthchecks actually run — fails
only on HTTP ≥ 400, and load balancers route on the status code, not on JSON.

A server with no model was therefore reported **healthy** and kept receiving
traffic, with every request failing 503 at `/predict`. The one component whose
job was to notice the outage was the one insisting nothing was wrong. It now
returns **503** when the model is not loaded.

Worth separating while you are here: this is a **readiness** probe ("can I serve
right now"). **Liveness** ("is the process wedged, restart me") is a different
question and needs an endpoint that does *not* depend on the model — otherwise a
pod that cannot load a model crash-loops forever instead of being marked unready
and left alone.

### The graceful shutdown code prevented graceful shutdown

`main.py` registered SIGTERM/SIGINT handlers at import time that called
`sys.exit(0)`. Uvicorn installs its own handlers, which stop accepting new
connections, drain in-flight requests, and *then* run the lifespan shutdown.
Overriding them with an immediate exit kills the process mid-request and skips
the "Shutting down gracefully" path entirely.

Registering handlers at import is also unsound on its own terms: signal handlers
can only be set from the main thread, so importing the module from a worker
thread or some test runners raises `ValueError`. The handlers are gone; cleanup
belongs after the `yield` in `lifespan()`.

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/genieincodebottle/aiml-companion.git
cd aiml-companion/projects/mlops/model-serving-platform

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

"Realising that operational code fails by reassuring you. Three things here looked correct and were actively harmful. The drift monitor returned PSI 0.0 'stable' for a distribution that had moved a hundred standard deviations, because the histogram silently dropped out-of-range values — it was loudest for mild drift and mute for total drift. The health endpoint returned HTTP 200 with 'unhealthy' in the body, so `curl -f` and every load balancer saw a healthy server with no model and kept routing to it. And the SIGTERM handlers I was proudest of called sys.exit(0) at import time, which pre-empted uvicorn's own draining logic — the graceful-shutdown code was the thing preventing graceful shutdown. Now shutdown is left to uvicorn plus the lifespan block, /health answers 503 when it cannot serve, and PSI counts out-of-range mass. The lesson I actually took: for infrastructure, test the failure path, because the success path tests all passed the whole time."

### "What would you do differently?"

"Add A/B testing infrastructure for safe model rollouts, implement canary deployments with traffic splitting, and add prediction logging to a data warehouse for offline analysis and retraining triggers."

### "How does this scale?"

"Horizontally via Kubernetes: the Docker container is stateless (model loaded at startup from shared volume), health checks enable rolling updates, and Prometheus metrics feed into auto-scaling policies based on latency P95."

### "Explain CI/CD to a non-technical person."

"Every time a developer pushes code, an automated system checks for mistakes (linting), runs tests, builds a shipping container, and deploys it. It's like having a quality inspector who checks every product before it leaves the factory, but running in seconds instead of hours."

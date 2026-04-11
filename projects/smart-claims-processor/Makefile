.PHONY: setup install demo test lint clean serve-hitl

# Setup project
setup:
	cp -n .env.example .env || true
	pip install -r requirements.txt
	mkdir -p data/audit_logs data/sample_claims data/fraud_patterns

# Install dependencies only
install:
	pip install -r requirements.txt

# Run demo with all sample claims
demo:
	python main.py --demo

# Run a single test claim
test-claim:
	python main.py --claim data/sample_claims/auto_accident.json --verbose

# Run test suite
test:
	pytest tests/ -v --tb=short

# Run with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Launch Streamlit dashboard
dashboard:
	streamlit run app.py --server.port 8501

# Launch HITL review API server
serve-hitl:
	uvicorn src.hitl.queue:router --host 0.0.0.0 --port 8001 --reload

# Run batch evaluation
eval:
	python evaluation/run_eval.py

# Lint
lint:
	ruff check src/ tests/ --fix || true

# Clean generated files
clean:
	rm -rf data/audit_logs/*.ndjson
	rm -f data/*.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

# Show HITL queue stats
queue-stats:
	python -c "from src.hitl.queue import queue_stats; import json; print(json.dumps(queue_stats(), indent=2))"

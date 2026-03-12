.PHONY: help install test lint analyze ui evaluate clean docker-build docker-run

## Display this help message
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed -e 's/## //' | head -20
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt

test: ## Run all tests with pytest
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint: ## Run code quality checks
	python -m py_compile src/config.py
	python -m py_compile src/llm.py
	python -m py_compile src/agents/graph.py
	python -m py_compile src/guardrails/manager.py
	@echo "All files compile successfully"

analyze: ## Run analysis (interactive prompt for company name)
	python main.py --stage analyze

ui: ## Launch Streamlit web interface
	streamlit run app.py

evaluate: ## Run evaluation framework
	python main.py --stage evaluate

clean: ## Remove generated files and caches
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	rm -rf src/__pycache__ src/agents/__pycache__ src/tools/__pycache__
	rm -rf src/models/__pycache__ src/guardrails/__pycache__ src/memory/__pycache__
	rm -rf tests/__pycache__ evaluation/__pycache__
	rm -rf data/cache.db logs/
	find . -name "*.pyc" -delete 2>/dev/null || true

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t due-diligence-agent .

docker-run: ## Run Docker container
	docker run -p 8501:8501 --env-file .env due-diligence-agent

all: install test lint ## Install, test, and lint

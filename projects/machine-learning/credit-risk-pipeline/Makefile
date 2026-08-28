.PHONY: setup run clean eda features train evaluate serve test lint docker

# Setup virtual environment and install dependencies
setup:
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt
	@echo "[OK] Setup complete. Activate with: source .venv/Scripts/activate"

# Run full pipeline
run:
	python main.py --verbose

# Run individual stages
clean:
	python main.py --stage clean

eda:
	python main.py --stage eda

features:
	python main.py --stage features

train:
	python main.py --stage train

evaluate:
	python main.py --stage evaluate

# Serve model via FastAPI
serve:
	uvicorn src.serve:app --reload --port 8000

# Run tests
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Clean artifacts
clean-artifacts:
	rm -rf artifacts/figures/*.png artifacts/results/*.joblib artifacts/results/*.md

# Docker
docker-build:
	docker build -t credit-risk-pipeline .

docker-run:
	docker run --rm -p 8000:8000 credit-risk-pipeline
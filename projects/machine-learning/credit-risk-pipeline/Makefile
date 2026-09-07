# Convenience wrapper only. `make` is NOT installed by default on Windows, so
# nothing here is the documented way in -- `python run.py` is, and it does
# everything these targets do. See README.md.
#
# The setup target used to read `.venv/Scripts/pip`, a Windows-only path inside
# a Unix-only tool, so it could not work on either platform. It detects now.

.PHONY: setup run clean eda features train evaluate serve test test-cov clean-artifacts docker-build docker-run

ifeq ($(OS),Windows_NT)
VENV_BIN := .venv/Scripts
RM_ARTIFACTS := del /Q artifacts\figures\*.png artifacts\results\*.joblib artifacts\results\*.md
else
VENV_BIN := .venv/bin
RM_ARTIFACTS := rm -f artifacts/figures/*.png artifacts/results/*.joblib artifacts/results/*.md
endif

# Create a virtual environment and install dependencies
setup:
	python -m venv .venv
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/python -m pip install -r requirements.txt
	@echo "[OK] Setup complete. Activate with:"
	@echo "     Windows: .venv\Scripts\activate"
	@echo "     macOS/Linux: source .venv/bin/activate"

# Run the full pipeline
run:
	python run.py --verbose

# Individual stages (each runs the stages before it)
clean:
	python run.py clean

eda:
	python run.py eda

features:
	python run.py features

train:
	python run.py train

evaluate:
	python run.py evaluate

# Serve the model via FastAPI
serve:
	python run.py serve

# Tests
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Remove generated artifacts
clean-artifacts:
	$(RM_ARTIFACTS)

# Docker
docker-build:
	docker build -t credit-risk-pipeline .

docker-run:
	docker run --rm -p 8000:8000 credit-risk-pipeline

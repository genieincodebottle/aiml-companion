# Portable entry point: the `lapse` console script is installed by `pip install -e .`,
# but `python -m` always works regardless of PATH.
LAPSE := python -m lapse_prediction.cli

.PHONY: install dev data train benchmark score models test test-all lint clean all

install:
	pip install -e .

dev:
	pip install -e ".[dev,neural]"

data:
	$(LAPSE) data --n 20000

train:
	$(LAPSE) train --model ordinal_chain

benchmark:
	$(LAPSE) benchmark --n 12000

score:
	$(LAPSE) score

models:
	$(LAPSE) models

test:
	pytest -m "not slow" -p no:warnings

test-all:
	pytest -p no:warnings

lint:
	ruff check src tests

clean:
	rm -rf artifacts models data .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

all: data train score

.PHONY: help test demo sweep lint clean

help:
	@echo "make test   - run the suite (numpy only, no GPU, no network)"
	@echo "make demo   - simulate a shift and write artifacts/"
	@echo "make sweep  - confidence sweep, false alerts per shift"
	@echo "make lint   - ruff, if it is installed"

test:
	python -m pytest

demo:
	python scripts/run_demo.py

sweep:
	python scripts/run_demo.py --sweep-only

lint:
	ruff check . || true

clean:
	rm -rf .pytest_cache artifacts/clips data
	find . -name __pycache__ -type d -exec rm -rf {} +

# `make` is absent on stock Windows. Every target is a single command you can
# run directly. RUN needs no install of this project at all.
RUN := python run.py

.PHONY: install source daily score rerun backfill break naive runs notebook test clean all

install:
	uv venv
	uv pip install -r requirements.txt

source:
	$(RUN) source

daily:
	$(RUN) daily

score:
	$(RUN) score

rerun:
	$(RUN) rerun

backfill:
	$(RUN) backfill

break:
	$(RUN) break

naive:
	$(RUN) naive

runs:
	$(RUN) runs

notebook:
	python notebooks/_build_notebook.py
	python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1500 notebooks/incremental_warehouse_standalone.ipynb

test:
	pytest -p no:warnings

clean:
	rm -rf data artifacts .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

all:
	$(RUN) all

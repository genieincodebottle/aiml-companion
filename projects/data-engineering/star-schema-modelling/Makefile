# `make` is absent on stock Windows. Every target below is a single command you
# can run directly. RUN needs no install of this project at all.
RUN := python run.py

.PHONY: install data build check naive break all notebook test clean

install:
	uv pip install -r requirements.txt

data:
	$(RUN) data

build:
	$(RUN) build

check:
	$(RUN) check

naive:
	$(RUN) naive

break:
	$(RUN) break

all:
	$(RUN) all

notebook:
	python notebooks/_build_notebook.py
	python -m nbconvert --to notebook --execute --inplace notebooks/star_schema_modelling_standalone.ipynb

test:
	pytest -p no:warnings

clean:
	rm -rf data artifacts .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

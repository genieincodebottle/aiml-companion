# `make` is absent on stock Windows -- every target below is a single command
# you can run directly. RUN needs no install of this project at all.
RUN := python run.py

.PHONY: install dev data diagnose compare recover explain notebook test clean all

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt && pip install -e .

data:
	$(RUN) data

diagnose:
	$(RUN) diagnose

compare:
	$(RUN) compare

recover:
	$(RUN) recover

explain:
	$(RUN) explain

notebook:
	jupyter lab notebooks/rent_price_explainer_standalone.ipynb

test:
	pytest -p no:warnings

clean:
	rm -rf data artifacts models .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

all: data compare explain

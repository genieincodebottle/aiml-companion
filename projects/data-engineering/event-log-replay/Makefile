# `make` is absent on stock Windows -- every target below is a single command
# you can run directly. RUN needs no install of this project at all.
RUN := python run.py

.PHONY: install produce compare replay batches inspect kafka-up kafka-down notebook build-notebook test clean all

install:
	pip install -r requirements.txt

produce:
	$(RUN) produce

compare:
	$(RUN) compare

replay:
	$(RUN) replay

batches:
	$(RUN) batches

inspect:
	$(RUN) inspect

# Optional real broker. Needs: pip install -r requirements-kafka.txt
kafka-up:
	docker compose up -d

kafka-down:
	docker compose down

notebook:
	jupyter lab notebooks/event_log_replay_standalone.ipynb

build-notebook:
	python notebooks/_build_notebook.py
	python -m nbconvert --to notebook --execute --inplace notebooks/event_log_replay_standalone.ipynb

test:
	pytest -p no:warnings

clean:
	rm -rf data artifacts .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

all: produce compare replay batches inspect

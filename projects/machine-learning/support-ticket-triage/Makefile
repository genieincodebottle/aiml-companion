# Thin wrappers around run.py. `make` is not installed by default on Windows,
# so every target here is a one-line python command you can run directly.
.PHONY: install data independence strategies calibrate sweep route test notebook clean all

install:
	pip install -r requirements.txt

data:
	python run.py data

independence:
	python run.py independence

strategies:
	python run.py strategies

calibrate:
	python run.py calibrate

sweep:
	python run.py sweep

route:
	python run.py route

test:
	pytest -p no:warnings

notebook:
	jupyter lab notebooks/support_ticket_triage_standalone.ipynb

clean:
	rm -rf data artifacts models

all: data independence strategies calibrate sweep route

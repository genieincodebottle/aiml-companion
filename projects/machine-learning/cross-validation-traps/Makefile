# Thin wrappers around run.py. `make` is not installed by default on Windows,
# so every target here is a one-line python command you can run directly.
.PHONY: install data truth preprocessing grouped temporal selection variance sweep test notebook clean all

install:
	pip install -r requirements.txt

data:
	python run.py data

truth:
	python run.py truth

preprocessing:
	python run.py preprocessing

grouped:
	python run.py grouped

temporal:
	python run.py temporal

selection:
	python run.py selection

variance:
	python run.py variance

sweep:
	python run.py sweep

test:
	pytest -p no:warnings

notebook:
	jupyter lab notebooks/cross_validation_traps_standalone.ipynb

clean:
	rm -rf data artifacts

all: data truth preprocessing grouped temporal selection variance sweep

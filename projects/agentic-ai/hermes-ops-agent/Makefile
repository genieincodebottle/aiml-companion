.PHONY: help install fixture demo test test-cov lint sessions compare probe \
        install-skills verify-gateway clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | sed -e 's/:.*## /~/' \
	  | awk -F'~' '{printf "  %-16s %s\n", $$1, $$2}'

install:  ## Install dependencies (the harness itself is stdlib-only)
	pip install -r requirements.txt

fixture:  ## Build fixtures/state.fixture.db from the checked-in SQL
	python scripts/build_fixture.py

demo: fixture  ## Run the comparison on the fixture. No Hermes, no API key.
	python main.py demo

test: fixture  ## Run the test suite
	python -m pytest tests/ -v --tb=short

test-cov: fixture  ## Run tests with a coverage report
	python -m pytest tests/ --cov=src --cov-report=term-missing

lint:  ## Byte-compile the sources and parse the shell script
	python -m compileall -q src main.py scripts
	bash -n deploy/verify-allowlist.sh
	python -c "import yaml; yaml.safe_load(open('configs/config.yaml')); yaml.safe_load(open('deploy/docker-compose.yml'))"
	@echo "ok"

sessions:  ## List sessions from your local Hermes state.db
	python main.py sessions

compare:  ## Compare two of your sessions: make compare BASE=<id> CAND=<id>
	@test -n "$(BASE)" -a -n "$(CAND)" \
	  || { echo "usage: make compare BASE=<id> CAND=<id>"; exit 2; }
	python main.py compare $(BASE) $(CAND)

probe:  ## Write prompt-injection payloads into ./probe for the agent to read
	python main.py probe --write ./probe

install-skills:  ## Copy the skill library into $$HERMES_HOME/skills
	@dest="$${HERMES_HOME:-$$HOME/.hermes}/skills"; \
	 mkdir -p "$$dest"; \
	 cp -R skills/. "$$dest"/; \
	 echo "installed skills into $$dest"; \
	 echo "verify with: hermes skills browse"

verify-gateway:  ## Run the allowlist and secrets checks against your install
	bash deploy/verify-allowlist.sh $${BOT:-your-bot-handle}

clean:  ## Remove build artefacts and caches
	rm -f fixtures/state.fixture.db
	rm -rf .pytest_cache htmlcov .coverage
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

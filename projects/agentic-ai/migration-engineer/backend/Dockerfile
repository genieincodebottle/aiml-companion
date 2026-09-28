# Backend image: FastAPI + the migration orchestrator, installed with uv.
FROM python:3.12-slim

# git is required at runtime: the worker clones, branches, commits, and pushes real
# repositories (both the GitHub provider and the local-git demo provider use it).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Bring in the uv binary (pinned for reproducibility).
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Resolve + install dependencies first (cached layer), then copy the app code.
# The gemini extra ships in the image so a GEMINI_API_KEY alone flips the container
# to live mode; the Claude SDK extra is opt-in (add --extra sdk + ANTHROPIC_API_KEY).
COPY pyproject.toml ./
RUN uv sync --no-dev --extra gemini

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

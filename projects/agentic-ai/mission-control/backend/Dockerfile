# Backend image: the Mission Control platform (FastAPI), installed reproducibly with uv.
FROM python:3.14-slim

# Bring in the uv binary (pinned for reproducibility).
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    MC_DATA_DIR=/data

# Install dependencies first (cached layer), then the app code. --frozen: the image
# gets exactly the versions in uv.lock, the ones the tests ran against.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000
# 0.0.0.0 inside the container network; docker-compose publishes it on 127.0.0.1 only.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
